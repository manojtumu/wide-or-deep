#!/usr/bin/env python
"""End-to-end on a synthetic corpus, small enough to run on a laptop CPU.

    python src/smoke.py

Proves the chain holds together: shapes -> runs -> bundle -> train -> log,
and bench on the same model. Catches the wiring mistakes that otherwise
surface an hour into a p5 job.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import write_tiny_corpus                     # noqa: E402
from runs import Bench, Train                          # noqa: E402
from shapes import Shape                               # noqa: E402

VOCAB, SEQ = 256, 64
TINY = {"deep": Shape(512, 4, 1024, VOCAB), "wide": Shape(1024, 1, 2048, VOCAB)}


def main():
    tmp = Path(tempfile.mkdtemp(prefix="smoke-"))
    data = write_tiny_corpus(tmp / "data", vocab=VOCAB, tokens=60_000)
    print(f"corpus {data}")
    for k, s in TINY.items():
        print(f"  {k:>5} {s.label():>8}  blocks {s.blocks/1e6:5.2f}M  total {s.params/1e6:5.2f}M")

    cfgs = [Train(shape=s, lr=1e-3, tokens=8 * SEQ * 12, seq=SEQ, micro=8, global_batch=8,
                  val_every=6, val_batches=2, log_every=4, compile=False, tag="smoke")
            for s in TINY.values()]
    (tmp / "block.json").write_text(json.dumps([c.to_json() for c in cfgs]))
    print(f"\ntrain: {len(cfgs)} runs x {cfgs[0].steps} steps")
    r = subprocess.run([sys.executable, str(HERE / "train.py"), "--bundle", str(tmp / "block.json"),
                        "--data", str(data), "--ckpt", str(tmp / "ckpt")],
                       text=True, capture_output=True)
    print(r.stdout.strip() or r.stderr[-1500:])
    assert r.returncode == 0, "train failed"

    for c in cfgs:
        rec = [json.loads(l) for l in (tmp / "ckpt" / c.run_id / "log.jsonl").read_text().splitlines()]
        vals = [x["val_loss"] for x in rec if "val_loss" in x]
        assert vals, f"no validation for {c.run_id}"
        print(f"  {c.shape.label():>8}  val {vals[0]:.3f} -> {vals[-1]:.3f}  "
              f"({len(rec)} log lines, resumable at {(tmp/'ckpt'/c.run_id/'latest').exists()})")

    bcfg = [Bench(shape=s, seq=SEQ, batch=4, steps=3, warmup=1, compile=False) for s in TINY.values()]
    (tmp / "bench.json").write_text(json.dumps([b.to_json() for b in bcfg]))
    print("\nbench:")
    r = subprocess.run([sys.executable, str(HERE / "bench.py"), "--bundle", str(tmp / "bench.json"),
                        "--out", str(tmp / "bench.jsonl")], text=True, capture_output=True)
    print(r.stdout.strip() or r.stderr[-1500:])
    assert r.returncode == 0, "bench failed"
    print("\nOK")


if __name__ == "__main__":
    main()
