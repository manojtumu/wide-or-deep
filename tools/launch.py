#!/usr/bin/env python
"""Submit one SageMaker training job for this study.

    python tools/launch.py --entry prep  --hp tokens=10e9                                   # corpus, CPU box
    python tools/launch.py --entry bench --nproc 8 --bundle-file bundles/bench.json           # throughput table
    python tools/launch.py --entry train --name S-400M-sweep-0 --bundle-file bundles/sweep-0.json \
        --data-s3 s3://<bucket>/shape-study/<prep-job>/output/ --max-hours 6                  # LR sweep bundle
    python tools/launch.py --entry train --name B-400M-deep-0 --bundle-file bundles/block-deep-0.json \
        --data-s3 s3://<bucket>/shape-study/<prep-job>/output/ --max-hours 8                  # one block run
    python tools/launch.py --entry shard --instance-count 4 --bundle-file bundles/<bundle>.json ...   # N nodes, one job

Set SAGEMAKER_ROLE_ARN to an execution role that can read and write the
session's default bucket. The region defaults to us-west-2 (AWS_REGION
overrides it). p5 capacity is often spot-only, so spot is the default there:
SageMaker may reclaim the node with two minutes' notice, and the job resumes
from /opt/ml/checkpoints once capacity returns. Training jobs need a stable
--name because the checkpoint path is derived from it, and a resubmit with the
same name resumes instead of restarting.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import boto3

try:
    from sagemaker.core.training.configs import (CheckpointConfig, Compute, InputData, OutputDataConfig,
                                                 SourceCode, StoppingCondition)
except ImportError:                                  # older v3 layout
    from sagemaker.train.configs import (CheckpointConfig, Compute, InputData, OutputDataConfig,
                                         SourceCode, StoppingCondition)
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.distributed import Torchrun
from sagemaker.train.model_trainer import ModelTrainer

REGION = os.environ.get("AWS_REGION", "us-west-2")
ROLE = os.environ.get("SAGEMAKER_ROLE_ARN") or SystemExit(
    "set SAGEMAKER_ROLE_ARN to a SageMaker execution role that can read and write the session's default bucket")
if isinstance(ROLE, SystemExit):
    raise ROLE
IMAGE = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.8.0-gpu-py312"
PROJECT = "shape-study"
SRC = Path(__file__).resolve().parent.parent / "src"
ENTRY = {"train": "train.py", "bench": "bench.py", "prep": "prep_data.py", "smoke": "smoke.py",
         "shard": "shard.py"}
REQS = {"prep": "requirements_prep.txt"}          # everything else runs on the DLC's own torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entry", choices=ENTRY, required=True)
    p.add_argument("--instance-type", default=None, help="default: p5 for GPU entries, m5.12xlarge for prep")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--nproc", type=int, default=0, help="torchrun procs/node; 0 = plain python entry")
    p.add_argument("--spot", dest="spot", action="store_true", default=None)
    p.add_argument("--no-spot", dest="spot", action="store_false")
    p.add_argument("--max-hours", type=float, default=1.0)
    p.add_argument("--max-wait-hours", type=float, default=None, help="spot: how long to wait for capacity")
    p.add_argument("--nccl-net", default=None, help="Socket to force TCP over the EFA hardware")
    p.add_argument("--name", default=None)
    p.add_argument("--hp", action="append", default=[], help="key=value passed as --key value")
    p.add_argument("--data-s3", default=None, help="S3 prefix mounted at /opt/ml/input/data/data (train)")
    p.add_argument("--bundle-file", default=None, help="local JSON list of Train configs; uploaded and mounted as the bundle channel (sweep)")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if a.instance_type is None:
        a.instance_type = "ml.m5.12xlarge" if a.entry == "prep" else "ml.p5.48xlarge"
    if a.entry in ("train", "shard"):
        # The checkpoint S3 path is derived from --name. A resubmit after a
        # failure or a MaxWaitTimeExceeded only resumes if that path is stable,
        # so an auto-generated timestamped name is refused for training runs.
        if not a.name:
            raise SystemExit("train jobs need a stable --name (it is the checkpoint path); e.g. --name A-400M-a128-s1")
        # shard.py is a plain python entry: it works out which node it is and
        # spawns its OWN single-node torchrun, so SageMaker must NOT wrap it in
        # a multi-node process group (that would train one config on all 32
        # GPUs and fail accum()).
        if not a.nproc and a.entry == "train":
            a.nproc = 8
        if a.max_wait_hours is None:
            a.max_wait_hours = max(48.0, a.max_hours * 3)     # spot: outlast a bad capacity day
    spot = a.spot if a.spot is not None else a.instance_type.startswith("ml.p5")
    name = a.name or f"{PROJECT}-{a.entry}-{time.strftime('%m%d%H%M')}"
    sess = Session(boto_session=boto3.session.Session(region_name=REGION))
    bucket = sess.default_bucket()
    root = f"s3://{bucket}/{PROJECT}/{name}"
    hps = dict(kv.split("=", 1) for kv in a.hp)
    env = {"PYTHONUNBUFFERED": "1", **({"NCCL_NET": a.nccl_net} if a.nccl_net else {})}
    max_run = int(a.max_hours * 3600)
    max_wait = int((a.max_wait_hours or a.max_hours * 2) * 3600)
    # Wire the channel paths the entry scripts expect, so a caller passes a local
    # bundle file and nothing else. SageMaker mounts channel <n> at
    # /opt/ml/input/data/<n>, and anything under /opt/ml/output/data is uploaded.
    if a.bundle_file:
        hps.setdefault("bundle", "/opt/ml/input/data/bundle/bundle.json")
    if a.entry in ("train", "shard"):
        hps.setdefault("data", "/opt/ml/input/data/data")
        hps.setdefault("ckpt", "/opt/ml/checkpoints")
    if a.entry == "bench":
        hps.setdefault("out", "/opt/ml/output/data/bench.jsonl")

    print(f"  job       {name}\n  entry     {ENTRY[a.entry]}  nproc/node {a.nproc or 'plain'}\n"
          f"  compute   {a.instance_count} x {a.instance_type}  {'SPOT' if spot else 'on-demand'}\n"
          f"  limits    run {a.max_hours}h" + (f"  wait {max_wait/3600:.1f}h" if spot else "") + "\n"
          f"  output    {root}/output\n  env       {env}\n  hp        {hps}" + (f"\n  data      {a.data_s3}" if a.data_s3 else ""))
    if a.dry_run:
        return

    kw = dict(
        training_image=IMAGE, role=ROLE, sagemaker_session=sess, base_job_name=name,
        source_code=SourceCode(source_dir=str(SRC), entry_script=ENTRY[a.entry],
                               **({"requirements": REQS[a.entry]} if a.entry in REQS else {})),
        compute=Compute(instance_type=a.instance_type, instance_count=a.instance_count,
                        enable_managed_spot_training=spot),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=max_run,
                                             **({"max_wait_time_in_seconds": max(max_wait, max_run)} if spot else {})),
        output_data_config=OutputDataConfig(s3_output_path=f"{root}/output", compression_type="NONE"),
        environment=env, hyperparameters=hps,
    )
    # Checkpoints for every training job, not just spot: an on-demand job can
    # still hit its runtime cap, and resume-by-run_id is what makes a resubmit
    # continue instead of restart.
    if spot or a.entry in ("train", "shard"):
        kw["checkpoint_config"] = CheckpointConfig(s3_uri=f"{root}/ckpt", local_path="/opt/ml/checkpoints")
    if a.nproc:
        kw["distributed"] = Torchrun(process_count_per_node=a.nproc)
    channels = []
    if a.data_s3:
        channels.append(InputData(channel_name="data", data_source=a.data_s3))
    if a.bundle_file:
        key = f"{PROJECT}/{name}/bundle/bundle.json"
        boto3.client("s3", region_name=REGION).upload_file(a.bundle_file, bucket, key)
        channels.append(InputData(channel_name="bundle", data_source=f"s3://{bucket}/{PROJECT}/{name}/bundle/"))
        print(f"  bundle    s3://{bucket}/{key}")
    if channels:
        kw["input_data_config"] = channels

    trainer = ModelTrainer(**kw)
    trainer.train(wait=a.wait)
    job = getattr(getattr(trainer, "_latest_training_job", None), "training_job_name", None)
    print(f"\nsubmitted  {job or name}")
    if a.entry in ("train", "bench"):
        print(f"resume     re-run this exact command; checkpoints live at {root}/ckpt")
    print(f"console    https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs/{job or ''}")
    print(f"logs       aws logs tail /aws/sagemaker/TrainingJobs --follow --region {REGION} --log-stream-name-prefix {job or name}")


if __name__ == "__main__":
    main()
