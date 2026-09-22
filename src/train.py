#!/usr/bin/env python
"""Train one model, or a bundle of them in sequence.

    torchrun --nproc_per_node=8 train.py --bundle block.json --data /data \
             --ckpt /opt/ml/checkpoints --out /opt/ml/output

Replicated data parallel on one node. No sharding: a 400M model with Adam
states is ~7 GB and these run on 80 GB cards.

Per run, under <ckpt>/<run_id>/:
    env.json    the config, world, versions -- written once
    log.jsonl   one line per log interval, and every validation
    latest/     model + optimizer, for resuming after a spot reclaim

Resume is by run_id, so resubmitting the same bundle continues rather than
restarts, and a run already at its final step is skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data import TokenShards
from model import Transformer, set_init_seed
from runs import Train

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"
RANK = int(os.environ.get("RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))


def log(*a):
    if RANK == 0:
        print(*a, flush=True)


def lr_at(cfg: Train, step: int) -> float:
    """Linear warmup then cosine to min_lr_ratio of peak."""
    warm = max(1, int(cfg.warmup_frac * cfg.steps))
    if step < warm:
        return cfg.lr * (step + 1) / warm
    t = (step - warm) / max(1, cfg.steps - warm)
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def validate(model, shards, cfg) -> float:
    model.eval()
    tot = torch.zeros((), device=DEVICE)
    for i in range(cfg.val_batches):
        x, y = shards.val_batch(i, cfg.micro, cfg.seq)
        tot += model(x.to(DEVICE), y.to(DEVICE)).float()
    model.train()
    if WORLD > 1:
        dist.all_reduce(tot, op=dist.ReduceOp.SUM)
        tot /= WORLD
    return float(tot / cfg.val_batches)


def run_one(cfg: Train, shards: TokenShards, ckpt_root: Path):
    d = ckpt_root / cfg.run_id
    d.mkdir(parents=True, exist_ok=True)
    accum = cfg.accum(WORLD)

    torch.manual_seed(cfg.seed)
    set_init_seed(cfg.seed)
    model = Transformer(cfg.shape, cfg.seq, dtype=torch.bfloat16, device=DEVICE)
    if cfg.compile:
        model = torch.compile(model, dynamic=False)
    if WORLD > 1:
        model = DDP(model, device_ids=[RANK % torch.cuda.device_count()] if CUDA else None)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, cfg.beta2),
                            weight_decay=cfg.weight_decay, fused=CUDA)

    start = 0
    if (d / "latest" / "model.pt").exists():
        st = torch.load(d / "latest" / "model.pt", map_location=DEVICE, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["step"]
        log(f"  resumed at step {start}")
    if start >= cfg.steps:
        log(f"  {cfg.label()} already complete"); return

    if RANK == 0 and not (d / "env.json").exists():
        (d / "env.json").write_text(json.dumps({
            "config": json.loads(cfg.to_json()), "run_id": cfg.run_id, "world": WORLD,
            "accum": accum, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if CUDA else "cpu",
            "job": os.environ.get("TRAINING_JOB_NAME", "")}, indent=1))

    logf = open(d / "log.jsonl", "a") if RANK == 0 else None
    last_ckpt = time.time()
    log(f"  {cfg.label()}  {cfg.steps} steps  world {WORLD} x micro {cfg.micro} x accum {accum}")

    for step in range(start, cfg.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(cfg, step)
        opt.zero_grad(set_to_none=True)
        tot = torch.zeros((), device=DEVICE)
        for m in range(accum):
            x, y = shards.batch(cfg.seed, step, RANK, m, cfg.micro, cfg.seq, WORLD, accum)
            # only sync gradients on the last micro-step of the accumulation
            ctx = model.no_sync() if (WORLD > 1 and m < accum - 1) else torch.enable_grad()
            with ctx:
                loss = model(x.to(DEVICE), y.to(DEVICE)) / accum
                loss.backward()
            # keep it on the GPU: float() here would sync every micro-step just to
            # build a number that is only read at log_every, and it warns because
            # the tensor still carries requires_grad
            tot += loss.detach()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if RANK == 0 and step % cfg.log_every == 0:
            logf.write(json.dumps({"step": step, "loss": float(tot), "lr": lr_at(cfg, step),
                                   "grad_norm": float(gn)}) + "\n"); logf.flush()
        if (step + 1) % cfg.val_every == 0 or step + 1 == cfg.steps:
            v = validate(model, shards, cfg)
            if RANK == 0:
                logf.write(json.dumps({"step": step + 1, "val_loss": v}) + "\n"); logf.flush()
                log(f"    step {step+1:>6}/{cfg.steps}  train {float(tot):.4f}  val {v:.4f}")
        if RANK == 0 and (time.time() - last_ckpt > cfg.ckpt_minutes * 60 or step + 1 == cfg.steps):
            # Swap the FILE, not the directory. os.replace on a regular file
            # overwrites atomically; on a directory it raises ENOTEMPTY as soon
            # as the destination has content, so renaming latest.tmp -> latest
            # succeeded for the first checkpoint of a run and failed for every
            # one after it. Writing model.pt.tmp beside model.pt and replacing
            # it keeps the guarantee that was wanted: at any instant the reader
            # sees either the previous checkpoint or the new one, never a
            # partial file, and there is no window with no checkpoint at all.
            live = d / "latest"; live.mkdir(parents=True, exist_ok=True)
            tmp = live / "model.pt.tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1}, tmp)
            tmp.replace(live / "model.pt")
            last_ckpt = time.time()
    if logf:
        logf.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="json list of Train configs")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="/opt/ml/checkpoints")
    a = ap.parse_args()

    if WORLD > 1:
        dist.init_process_group("nccl" if CUDA else "gloo")
        if CUDA:
            torch.cuda.set_device(RANK % torch.cuda.device_count())

    shards = TokenShards(a.data)
    cfgs = [Train.from_json(x) for x in json.loads(Path(a.bundle).read_text())]
    for i, cfg in enumerate(cfgs, 1):
        # the model pads the vocabulary to a multiple of 64 for GEMM alignment, so it
        # is larger than the tokenizer's: 50304 rows for gpt2's 50257 ids. The extra
        # rows are never fed in and learn near-zero logits. Must cover, not equal.
        assert cfg.shape.vocab >= shards.vocab, \
            f"model vocab {cfg.shape.vocab} < corpus {shards.vocab}: ids would be out of range"
        log(f"[{i}/{len(cfgs)}] {cfg.label()}")
        run_one(cfg, shards, Path(a.ckpt))
    if WORLD > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
