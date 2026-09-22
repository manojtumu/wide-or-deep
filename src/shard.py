"""Run a bundle across the nodes of ONE job, one independent run per node.

The point is capacity, not scale. A 1-node job releases its instance when its
bundle ends, so the next runs need a fresh grant; and SageMaker's own Torchrun
distribution would instead put every node in a single process group and train
ONE config across all 32 GPUs -- which this study does not want, and which
fails accum() outright since global_batch 128 is not divisible by 32*16.

So: take no distribution wrapper at all, work out which node this is, take
this node's slice of the bundle, and spawn a LOCAL 8-GPU torchrun for it.
Four nodes therefore run four different seeds concurrently under one grant,
each with exactly the world=8 DDP geometry every other run in the study used.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def node_index(expect: int = 0) -> tuple[int, int]:
    """(index, count) for this container, from SageMaker's own resource config.

    Refuses to guess. If this node cannot establish WHICH node it is, every
    node would fall back to "node 0 of 1", run the whole bundle, and four
    containers would write the same run_id checkpoints on top of each other --
    4x the cost and corrupt results, discovered hours later. A loud failure is
    strictly better, so an unresolvable identity raises and --nodes is checked
    against what the cluster actually reports.
    """
    hosts = json.loads(os.environ.get("SM_HOSTS", "[]"))
    me = os.environ.get("SM_CURRENT_HOST", "")
    if not hosts:                                  # fall back to the file SageMaker always writes
        cfg = Path("/opt/ml/input/config/resourceconfig.json")
        if cfg.exists():
            rc = json.loads(cfg.read_text())
            hosts, me = rc.get("hosts", []), rc.get("current_host", me)
    if not hosts:
        if expect > 1:
            raise SystemExit(f"cannot determine node identity but --nodes={expect} was given: "
                             "refusing to run, every node would repeat the whole bundle")
        return 0, 1                                # genuinely single-node (or off SageMaker)
    if me not in hosts:
        raise SystemExit(f"current host {me!r} not in hosts {hosts}: refusing to guess")
    if expect and len(hosts) != expect:
        raise SystemExit(f"cluster reports {len(hosts)} nodes but --nodes={expect}: "
                         "the bundle would be split wrongly")
    return sorted(hosts).index(me), len(hosts)


def main():
    # SageMaker passes hyperparameters as "--key value", not "--key=value"
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="/opt/ml/input/data/bundle/bundle.json")
    ap.add_argument("--ckpt", default="/opt/ml/checkpoints")
    ap.add_argument("--data", default="/opt/ml/input/data/data")
    ap.add_argument("--nproc", default="8")
    ap.add_argument("--nodes", type=int, default=0,
                    help="how many nodes this job was submitted with; checked against the "
                         "cluster so a mis-sized split fails loudly instead of duplicating work")
    a, unknown = ap.parse_known_args()
    bundle, ckpt, data, nproc = a.bundle, a.ckpt, a.data, a.nproc
    if unknown:
        print(f"  ignoring extra args {unknown}", flush=True)

    i, n = node_index(a.nodes)
    runs = json.load(open(bundle))
    mine = runs[i::n]                      # interleaved: balances unequal run costs
    print(f"  node {i+1}/{n}: {len(mine)} of {len(runs)} runs", flush=True)
    if not mine:
        print("  nothing for this node", flush=True)
        return

    local = Path("/tmp/node_bundle.json")
    local.write_text(json.dumps(mine))
    for s in mine:
        c = json.loads(s)
        print(f"    {c['shape']['d']}x{c['shape']['n_layers']} "
              f"lr{c['lr']:.2e} seed{c['seed']}", flush=True)

    cmd = ["torchrun", "--nnodes=1", f"--nproc_per_node={nproc}",
           str(Path(__file__).parent / "train.py"),
           "--bundle", str(local), "--ckpt", ckpt, "--data", data]
    print(f"  exec {' '.join(cmd)}", flush=True)
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
