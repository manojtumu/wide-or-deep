#!/usr/bin/env python
"""Time one model. Throughput and MFU, nothing else.

    torchrun --nproc_per_node=1 bench.py --bundle bench.json --out bench.jsonl

One process per GPU under DDP, or one process alone. Each cell builds a fresh
model, warms up, times a fixed number of steps, and appends a JSON line.

Deliberately has no layout grid, no tensor or pipeline parallelism, no
collectives-by-difference. The study measures three models on one machine, so
the only thing worth timing is the model.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from metrics import PEAK_TFLOPS
from model import Transformer, set_init_seed
from runs import Bench

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"


def measure(run: Bench, world: int, rank: int, gpu: str) -> dict:
    torch.manual_seed(run.seed)
    set_init_seed(run.seed)
    model = Transformer(run.shape, run.seq, dtype=torch.bfloat16, device=DEVICE)
    if run.compile:
        model = torch.compile(model, dynamic=False)
    if world > 1:
        model = DDP(model, device_ids=[rank % torch.cuda.device_count()] if CUDA else None)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=CUDA)

    ids = torch.randint(0, run.shape.vocab, (run.batch, run.seq), device=DEVICE)
    tgt = torch.randint(0, run.shape.vocab, (run.batch, run.seq), device=DEVICE)

    def step():
        opt.zero_grad(set_to_none=True)
        model(ids, tgt).backward()
        opt.step()

    for _ in range(run.warmup):
        step()
    if CUDA:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(run.steps):
        step()
    if CUDA:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if world > 1:                                   # the slowest rank sets the step time
        t = torch.tensor([elapsed], device=DEVICE)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        elapsed = float(t.item())

    step_s = elapsed / run.steps
    tps = run.tokens_per_step_per_gpu * world / step_s
    peak = PEAK_TFLOPS[gpu] * 1e12
    return {"config": json.loads(run.to_json()), "run_id": run.run_id, "label": run.label(),
            "status": "ok", "world": world, "gpu": gpu,
            "step_s": step_s, "tokens_per_s": tps, "tokens_per_s_per_gpu": tps / world,
            "flops_per_token": run.shape.flops_per_token(run.seq),
            "params": run.shape.params, "blocks": run.shape.blocks,
            "mfu": tps * run.shape.flops_per_token(run.seq) / (world * peak),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if CUDA else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="json list of Bench configs")
    ap.add_argument("--out", default="bench.jsonl")
    ap.add_argument("--gpu", default="H100")
    a = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world > 1:
        dist.init_process_group("nccl" if CUDA else "gloo")
        if CUDA:
            torch.cuda.set_device(rank % torch.cuda.device_count())

    cells = [Bench.from_json(x) for x in json.loads(Path(a.bundle).read_text())]
    for i, run in enumerate(cells, 1):
        try:
            r = measure(run, world, rank, a.gpu)
        except torch.OutOfMemoryError:
            r = {"config": json.loads(run.to_json()), "run_id": run.run_id, "status": "oom"}
        if rank == 0:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
            done = f"{r['tokens_per_s']:>9,.0f} tok/s  MFU {r['mfu']*100:5.1f}%  " \
                   f"{r['peak_mem_gb']:5.1f} GB" if r["status"] == "ok" else "OOM"
            print(f"[{i}/{len(cells)}] {run.label():<28} {done}", flush=True)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
