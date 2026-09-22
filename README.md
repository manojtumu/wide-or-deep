# Wide or Deep?

Code and run records for *Wide or Deep? How Model Shape Sets Your Training Bill*
at https://manojtumu.github.io/.

Three decoder-only transformers with roughly matched block parameters, trained on
the same data for 8B tokens with three seeds each, then measured two ways: how much
each learns per token, and how many tokens per second one 8xH100 node delivers.
The deep shape learns the most per token. The wide shape is the cheapest to train
to the same loss.

| shape    | width x layers | ffn  | block params |
|----------|----------------|------|--------------|
| deep     | 1024 x 32      | 2688 | 398.5M       |
| balanced | 1536 x 12      | 5120 | 396.4M       |
| wide     | 2048 x 8       | 5376 | 398.5M       |

Multi-head attention (head dim 64), SwiGLU feed-forward, tied embeddings, 2,048-token
context, GPT-2 BPE (50,304 vocabulary), FineWeb-Edu. Everything trains in bf16 with
plain data parallelism under `torch.compile`.

## Layout

    src/        model, data, training, benchmark, sharding; runs on the SageMaker PyTorch 2.8 container
    tools/      plan.py builds job bundles, launch.py submits them, lr.py picks learning rates, collect.py pulls journals
    bundles/    the job specs that produced the runs in results/
    results/    run journals (env.json + log.jsonl per run), the throughput table, the picked rates, and the ledger

`results/block400M/` holds the nine 8B-token runs, `results/sweep400M/` the eighteen
1B-token learning-rate runs, `results/bench400M.jsonl` the throughput measurements,
`results/lr.json` the learning rate chosen per shape, and `results/LEDGER.md` the
day-by-day log of what was submitted, what ran, and what it cost. Checkpoints are not
included.

## Reproducing

Everything ran as SageMaker training jobs in us-west-2. You need an AWS account with
p5.48xlarge quota, the AWS CLI, and Python 3.12 with `boto3` and `sagemaker>=3`.
Export a SageMaker execution role before launching anything:

    export SAGEMAKER_ROLE_ARN=arn:aws:iam::<account>:role/<role>

1. Print the three shapes and confirm the match:

       python src/shapes.py

2. Build the corpus (about 10B GPT-2 tokens of FineWeb-Edu, as uint16 shards) on a CPU box:

       python tools/launch.py --entry prep --hp tokens=10e9

   Note the job's output prefix; it is the `--data-s3` argument below.

3. Throughput, 40 timed steps after 10 warmup, compiled and eager, three repeats per shape:

       python tools/plan.py bench --emit bundles/bench.json
       python tools/launch.py --entry bench --nproc 8 --bundle-file bundles/bench.json

4. Learning-rate sweep, six shared rates at 1B tokens (the bundles here are the ones that ran):

       python tools/launch.py --entry train --name S-400M-sweep-0 --bundle-file bundles/sweep-0.json \
           --data-s3 s3://<bucket>/shape-study/<prep-job>/output/ --max-hours 6
       python tools/lr.py --ckpt results/sweep400M          # writes results/lr.json

5. The block runs, three shapes by three seeds at 8B tokens:

       python tools/plan.py block --lr-file results/lr.json --seeds 3 --emit bundles/block.json
       python tools/launch.py --entry train --name B-400M-<shape>-<seed> --bundle-file bundles/<bundle>.json \
           --data-s3 s3://<bucket>/shape-study/<prep-job>/output/ --max-hours 8

   `bundles/big-0.json` and `bundles/big-1.json` are the four-run bundles that produced
   eight of the nine runs; `bundles/block-deep-0.json` produced the first.

6. Pull the journals and check completeness:

       python tools/collect.py

Data order depends only on (seed, step), so every shape sees identical batches. Runs
checkpoint to `/opt/ml/checkpoints` and resume by run id, so a resubmitted job with the
same `--name` continues rather than restarts. `python src/smoke.py` runs the whole path
on a synthetic corpus on CPU as a check before spending GPU time.

## Cost

The full study, including the sweep and the benchmark, was about $1,550 of SageMaker
on-demand time at the September 2026 rate of $63.30 per ml.p5.48xlarge hour. Each
8B-token run took 1.3 to 1.5 hours.

## License

MIT.
