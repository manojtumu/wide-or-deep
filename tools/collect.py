#!/usr/bin/env python
"""Pull every block run's journal out of S3 and say what is actually complete.

    python tools/collect.py                 # sync + report
    python tools/collect.py --no-sync       # report on what is already local

Runs are spread over however many jobs it took to get capacity -- a 4-node
sharded job, single-node jobs, resumes after a failure -- so this discovers the
prefixes instead of hardcoding one. Every run's directory is named by its
run_id, which is a hash of the full config, so two prefixes can never disagree
about what a directory means and a re-sync is idempotent.

Only env.json / log.jsonl are fetched. The model.pt files are 2.5GB each and
nothing downstream reads them.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REGION = os.environ.get("AWS_REGION", "us-west-2")


def bucket() -> str:
    """SHAPE_STUDY_BUCKET, or the SageMaker default bucket tools/launch.py writes to."""
    b = os.environ.get("SHAPE_STUDY_BUCKET")
    if b:
        return b
    import boto3
    acct = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
    return f"sagemaker-{REGION}-{acct}"


ROOT = "shape-study"
DEST = Path("results/block400M")
SHAPE = {"1024x32": "deep", "1536x12": "balanced", "2048x8": "wide"}


def prefixes() -> list[str]:
    out = subprocess.run(["aws", "s3", "ls", f"s3://{bucket()}/{ROOT}/", "--region", REGION],
                         capture_output=True, text=True).stdout
    names = [l.split()[-1].rstrip("/") for l in out.splitlines() if l.strip().startswith("PRE")]
    return [n for n in names if n.startswith("B-400M")]


def sync(names: list[str]) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for n in names:
        src = f"s3://{bucket()}/{ROOT}/{n}/ckpt/"
        r = subprocess.run(["aws", "s3", "sync", src, str(DEST), "--region", REGION,
                            "--exclude", "*", "--include", "*/env.json", "--include", "*/log.jsonl",
                            "--quiet"], capture_output=True, text=True)
        print(f"  synced {n:<28}" + ("" if r.returncode == 0 else f"  FAILED {r.stderr[:80]}"))


def report() -> None:
    rows, seen = [], {}
    for env_p in sorted(DEST.rglob("env.json")):
        d = env_p.parent
        env = json.loads(env_p.read_text())
        cfg = env["config"]
        sh = cfg.get("shape") or cfg["run"]["shape"]
        key = f"{sh['d']}x{sh['n_layers']}"
        if key not in SHAPE:                       # a run from the abandoned geometry
            print(f"  SKIP {d.name}: unexpected geometry {key}")
            continue
        log = d / "log.jsonl"
        vals = {}
        if log.exists():
            for line in log.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if "val_loss" in r:
                    vals[r["step"]] = r["val_loss"]   # dict collapses any resume overlap
        total = env.get("steps") or 30518
        last = max(vals) if vals else 0
        rows.append((SHAPE[key], cfg["seed"], cfg["lr"], last, total,
                     vals.get(last), d.name, env.get("job", "?")))
        seen.setdefault((SHAPE[key], cfg["seed"]), d.name)

    print(f"\n  {'shape':>9} {'seed':>4} {'lr':>9} {'progress':>14} {'final val':>10}  run_id")
    for s, sd, lr, last, total, v, rid, job in sorted(rows):
        pct = 100 * last / total
        print(f"  {s:>9} {sd:>4} {lr:>9.2e} {last:>6}/{total:<6} {pct:>3.0f}% "
              f"{(f'{v:.4f}' if v is not None else '-'):>10}  {rid}")

    done = {k for k, _ in seen.items()
            if any(r[0] == k[0] and r[1] == k[1] and r[3] >= r[4] for r in rows)}
    want = {(s, sd) for s in SHAPE.values() for sd in (1, 2, 3)}
    print(f"\n  complete {len(done)}/9")
    if want - done:
        print("  missing: " + ", ".join(f"{s} s{sd}" for s, sd in sorted(want - done)))
    per = {s: sum(1 for k in done if k[0] == s) for s in SHAPE.values()}
    print("  seeds complete per shape: " + ", ".join(f"{k} {v}" for k, v in per.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sync", action="store_true")
    a = ap.parse_args()
    if not a.no_sync:
        ps = prefixes()
        print(f"  prefixes: {ps}")
        sync(ps)
    report()
