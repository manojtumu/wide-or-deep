#!/usr/bin/env python
"""What to run. Prints a plan; --emit writes the bundle a job consumes.

Three plans, in the order they have to happen:

  sweep   3 shapes x N learning rates, short horizon, one seed.
          The grid is SHARED across shapes on purpose. Per-shape grids give you
          each shape's optimum and nothing else; a shared grid also gives the
          head-to-head at every rate, which is the comparison that does not
          depend on interpolating an optimum -- and is the protocol the original
          shape-independence result used.

  block   3 shapes x N seeds at the full horizon, at the rates the sweep picked.

  bench   3 shapes timed, compiled and eager. Eager is not waste: the gap
          between them is how much of any shape difference is per-layer software
          overhead rather than arithmetic, and that scales with layer count.

    python tools/plan.py sweep --rates 5.3e-4,6.7e-4,8.4e-4,1.06e-3,1.33e-3,1.68e-3,2.11e-3
    python tools/plan.py block --lr-file results/lr.json --seeds 3
    python tools/plan.py bench
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from runs import Bench, Train                      # noqa: E402
from shapes import shapes_for                      # noqa: E402


def sweep(rates: list[float], tokens: float, tag: str) -> list[Train]:
    sh = shapes_for()
    # interleave by rate, not by shape: a job killed part-way then has every
    # shape at the rates it did reach, instead of one shape and two gaps
    return [Train(shape=sh[k], lr=lr, tokens=tokens, seed=1, val_every=250,
                  warmup_frac=max(0.02, 150 / Train(shape=sh[k], lr=lr, tokens=tokens).steps),
                  tag=f"{tag}-sweep")
            for lr in rates for k in ("deep", "balanced", "wide")]


def block(lr: dict[str, float], seeds: int, tokens: float, tag: str,
          which: tuple[str, ...] = ("deep", "balanced", "wide")) -> list[Train]:
    sh = shapes_for()
    return [Train(shape=sh[k], lr=lr[k], tokens=tokens, seed=s, tag=f"{tag}-block")
            for s in range(1, seeds + 1) for k in which]


def bench(batch: int, tag: str) -> list[Bench]:
    sh = shapes_for()
    return [Bench(shape=sh[k], batch=batch, compile=c, seed=r, tag=f"{tag}-bench")
            for r in range(3) for c in (True, False) for k in ("deep", "balanced", "wide")]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep"); s.add_argument("--rates", default=None)
    s.add_argument("--only", default=None,
                   help="explicit shape:rate pairs, e.g. wide:9.4e-4,wide:1.26e-3,deep:1.26e-3 -- "
                        "for filling gaps when part of a grid is already measured")
    s.add_argument("--tokens", type=float, default=1e9)
    b = sub.add_parser("block"); b.add_argument("--lr-file", required=True)
    b.add_argument("--seeds", type=int, default=3); b.add_argument("--tokens", type=float, default=8e9)
    b.add_argument("--skip", default=None,
                   help="shape:seed pairs already done or in flight, e.g. deep:1 -- excluded from "
                        "the emitted set so a rebundle does not re-run them")
    b.add_argument("--shapes", default="deep,balanced,wide",
                   help="subset to emit -- a shape whose rate the sweep has already pinned down "
                        "can start while the rest of the sweep is still running")
    k = sub.add_parser("bench"); k.add_argument("--batch", type=int, default=16)
    for p in (s, b, k):
        p.add_argument("--tag", default="400M")
        p.add_argument("--emit", default=None, help="write the bundle here")
        p.add_argument("--split", type=int, default=1,
                       help="split into N bundles for N concurrent jobs, packed longest-first")
    a = ap.parse_args()

    if a.cmd == "sweep":
        if a.only:
            sh = shapes_for()
            pairs = [(k, float(v)) for k, v in (x.split(":") for x in a.only.split(","))]
            cfgs = [Train(shape=sh[k], lr=lr, tokens=a.tokens, seed=1, val_every=250,
                          warmup_frac=max(0.02, 150 / Train(shape=sh[k], lr=lr, tokens=a.tokens).steps),
                          tag=f"{a.tag}-sweep") for k, lr in pairs]
        else:
            cfgs = sweep([float(x) for x in a.rates.split(",")], a.tokens, a.tag)
    elif a.cmd == "block":
        cfgs = block(json.load(open(a.lr_file)), a.seeds, a.tokens, a.tag,
                     tuple(a.shapes.split(",")))
        if a.skip:
            drop = {(k, int(v)) for k, v in (x.split(":") for x in a.skip.split(","))}
            keep, sh = [], shapes_for()
            name = {sh[k].label(): k for k in ("deep", "balanced", "wide")}
            for c in cfgs:
                if (name[c.shape.label()], c.seed) not in drop:
                    keep.append(c)
            print(f"  skipping {len(cfgs) - len(keep)} run(s): {sorted(drop)}")
            cfgs = keep
    else:
        cfgs = bench(a.batch, a.tag)

    node_h = 0.0
    for c in cfgs:
        if isinstance(c, Train):
            # 2.00 node-hours per 8B-token run at 400M, measured on 8xH100
            node_h += 2.00 * (c.tokens / 8e9) * (c.shape.flops_per_token() / 2.96e9)
            print(f"  {c.shape.label():>9}  lr {c.lr:.2e}  seed {c.seed}  "
                  f"{c.tokens/1e9:>4.0f}B  {c.steps:>6} steps  {c.run_id}")
        else:
            print(f"  {c.shape.label():>9}  b{c.batch}  {'compiled' if c.compile else 'eager   '}  "
                  f"rep {c.seed}  {c.run_id}")
    print(f"\n  {len(cfgs)} runs", end="")
    if node_h:
        print(f", ~{node_h:.1f} node-hours, ~${node_h*24.05:,.0f} spot / ${node_h*63.30:,.0f} on-demand", end="")
    print()
    if not a.emit:
        return
    Path(a.emit).parent.mkdir(parents=True, exist_ok=True)
    if a.split <= 1:
        Path(a.emit).write_text(json.dumps([c.to_json() for c in cfgs]))
        print(f"  wrote {a.emit}")
        return
    # Longest-processing-time first: repeatedly give the biggest remaining run to
    # whichever bundle is currently shortest. Round-robin would stack the three
    # deep runs together and leave one machine running hours after the others.
    def cost(c):
        return (c.tokens * c.shape.flops_per_token()) if isinstance(c, Train) else 1.0
    bins = [[] for _ in range(a.split)]
    load = [0.0] * a.split
    for c in sorted(cfgs, key=cost, reverse=True):
        i = load.index(min(load))
        bins[i].append(c)
        load[i] += cost(c)
    stem = Path(a.emit)
    for i, (b, l) in enumerate(zip(bins, load)):
        f = stem.with_name(f"{stem.stem}-{i}{stem.suffix}")
        f.write_text(json.dumps([c.to_json() for c in b]))
        est = 2.00 * l / (8e9 * 2.96e9) if isinstance(cfgs[0], Train) else 0
        print(f"  wrote {f}  {len(b)} runs" + (f"  ~{est:.2f} node-h" if est else ""))
    if isinstance(cfgs[0], Train):
        print(f"  makespan ~{2.00*max(load)/(8e9*2.96e9):.2f} node-h "
              f"(vs {2.00*sum(load)/(8e9*2.96e9):.2f} sequential)")


if __name__ == "__main__":
    main()
