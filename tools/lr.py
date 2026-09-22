#!/usr/bin/env python
"""Read a learning-rate sweep and pick the rate for each shape.

    python tools/lr.py --ckpt results/sweep400M            # table + results/lr.json

Reports two things, and the second is the one that matters.

The per-shape optimum is what the block runs need, so it is what gets written.
It is also the weaker number: a parabola through three points has no residual
degrees of freedom, and the fit leans on whatever the neighbouring cells did.

The head-to-head is the stronger claim. Because the grid is shared across
shapes, every rate is a controlled comparison in which no shape had a tuning
advantage -- and that is the protocol the original shape-independence result
used. If one shape wins at every rate, the ordering owes nothing to tuning.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SHAPES = ("deep", "balanced", "wide")


def read(ckpt: Path) -> dict:
    """{(shape_label, lr): (final_val, frac_done)} from each run's env + log."""
    out = {}
    for d in sorted(p.parent for p in ckpt.rglob("env.json")):
        env = json.loads((d / "env.json").read_text())
        cfg = env["config"]
        # the current Train holds shape directly; runs recorded under the older
        # config nested it inside a Run, so read either
        shape = cfg.get("shape") or cfg["run"]["shape"]
        rec = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines() if l.strip()] \
            if (d / "log.jsonl").exists() else []
        vals = [r for r in rec if "val_loss" in r]
        if not vals:
            continue
        steps = env.get("steps") or max(r["step"] for r in rec)
        out[(f"{shape['d']}x{shape['n_layers']}", cfg["lr"])] = (
            vals[-1]["val_loss"], vals[-1]["step"] / steps)
    return out


def vertex(xs, ys, i):
    """Parabola through the three points around index i, in log-rate. None if it
    opens downward or the vertex falls outside the bracket."""
    if not 0 < i < len(xs) - 1:
        return None
    (x0, y0), (x1, y1), (x2, y2) = [(xs[k], ys[k]) for k in (i - 1, i, i + 1)]
    den = (x0 - x1) * (x0 - x2) * (x1 - x2)
    A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
    B = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den
    if A <= 0:
        return None
    xf = -B / (2 * A)
    return math.exp(xf) if x0 <= xf <= x2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="dir holding <run_id>/{env,log} for the sweep")
    ap.add_argument("--out", default="results/lr.json")
    a = ap.parse_args()

    rows = read(Path(a.ckpt))
    if not rows:
        raise SystemExit(f"no sweep journals under {a.ckpt}")
    labels = sorted({k[0] for k in rows}, key=lambda s: int(s.split("x")[0]))
    lrs = sorted({k[1] for k in rows})
    name = dict(zip(labels, SHAPES))

    print(f"{'shape':>10} " + "".join(f"{lr:>12.2e}" for lr in lrs) + f"{'best':>11}{'fitted':>11}")
    best = {}
    for lab in labels:
        cells, cand = [], []
        for lr in lrs:
            r = rows.get((lab, lr))
            if not r:
                cells.append(f"{'--':>12}"); continue
            done = r[1] > 0.999
            cells.append(f"{r[0]:>11.4f}" + ("" if done else "*"))
            if done:
                cand.append((lr, r[0]))
        note, grid = "", None
        if cand:
            cand.sort()
            xs = [math.log(lr) for lr, _ in cand]
            ys = [v for _, v in cand]
            i = min(range(len(ys)), key=ys.__getitem__)
            grid = cand[i][0]
            pick = grid
            if i in (0, len(cand) - 1):
                note = "  EDGE: extend the grid this way"
            else:
                v = vertex(xs, ys, i)
                if v:
                    pick = v
            best[name[lab]] = float(f"{pick:.3g}")
        gcol = f"{grid:>11.2e}" if grid else f"{'--':>11}"
        fcol = f"{best[name[lab]]:>11.2e}" if name[lab] in best else f"{'--':>11}"
        print(f"{lab:>10} " + "".join(cells) + gcol + fcol + note)
    print("\n  cells: final validation loss; * = run incomplete, excluded from the pick")

    print(f"\n  head to head, per rate (no shape has a tuning advantage at a shared rate):")
    wins = {n: 0 for n in SHAPES}
    for lr in lrs:
        got = {name[lab]: rows[(lab, lr)][0] for lab in labels
               if (lab, lr) in rows and rows[(lab, lr)][1] > 0.999}
        if len(got) < 2:
            continue
        w = min(got, key=got.get); wins[w] += 1
        spread = max(got.values()) - min(got.values())
        print(f"    lr {lr:.2e}  " + "  ".join(f"{k} {v:.4f}" for k, v in got.items())
              + f"   -> {w}" + ("  (within seed noise)" if spread < 0.006 else ""))
    print(f"    wins: " + ", ".join(f"{k} {v}" for k, v in wins.items() if v))

    if best:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(best, indent=1))
        print(f"\n  wrote {a.out}: {best}")


if __name__ == "__main__":
    main()
