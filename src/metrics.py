"""FLOP accounting, MFU, and money. One place, no torch.

Every number a run reports that is derived from a measured time comes through
here, so a change to a peak figure or a price changes every artifact
consistently rather than in whichever script somebody remembered to edit.
"""

from __future__ import annotations

import statistics

# Dense bf16 tensor-core peak, TFLOP/s per GPU. No sparsity.
PEAK_TFLOPS = {"H100": 989.4, "A100": 311.9, "A10G": 125.0}

# USD per NODE-hour, SageMaker training, us-west-2 -- on-demand from the AWS
# Pricing API (2026-09-12). Spot fraction MEASURED from three completed p5 spot
# jobs (2026-09-13): BillableTimeInSeconds / TrainingTimeInSeconds = 0.37-0.38.
# SageMaker bills billable_seconds x the on-demand rate, so effective p5 spot
# is ~$24/node-hour of training.
SPOT_FRACTION = 0.38
NODE_PRICE = {
    "ml.p5.48xlarge": {"ondemand": 63.296, "gpus": 8, "gpu": "H100"},
    "ml.p4d.24xlarge": {"ondemand": 25.251, "gpus": 8, "gpu": "A100"},
    "ml.m5.12xlarge": {"ondemand": 2.765, "gpus": 0, "gpu": None},
}
for _v in NODE_PRICE.values():
    _v["spot"] = round(_v["ondemand"] * SPOT_FRACTION, 2)

# Reference price for the Socket arm: an 8xH100 node on commodity networking,
# which is a different price list from a p5 with EFA. Checked 2026-09-16 against
# published on-demand rates, per GPU-hour:
#
#   getdeploying.com/gpus/nvidia-h100   median $3.38 across 38 providers
#                                       cheapest in stock: Lium $1.30, Vast.ai
#                                       $1.74, HyperAI $1.80, GPU.ai $2.48
#   RunPod                              $1.99 PCIe / $2.69 SXM
#   io.net                              $2.10-3.50
#   Lambda (8-GPU H100 SXM tier)        $3.99
#   AWS (hyperscaler, for contrast)     $3.90-6.88
#
# The arm models a cluster WITHOUT an RDMA fabric, which is the budget tier of
# that list; the providers that do offer InfiniBand sit at the upper end. So
# $2.50/GPU-hour = $20/node-hour is the budget tier, below the $3.38 median.
# Still a parameter rather than a measurement: the composed dollar figures scale
# with it linearly, and the cross-cluster comparison (not the shape ranking,
# which is unaffected) turns over at $17.68/node-hour.
COMMODITY_H100_NODE_HR = 20.0


def mfu(tokens_per_s: float, flops_per_token: float, n_gpus: int, peak_tflops: float) -> float:
    return tokens_per_s * flops_per_token / (n_gpus * peak_tflops * 1e12)


def cost_usd(seconds: float, n_nodes: int, node_price_hr: float) -> float:
    return seconds / 3600 * n_nodes * node_price_hr


def gpu_hours_per_1b_tokens(tokens_per_s: float, n_gpus: int) -> float:
    return 1e9 / tokens_per_s * n_gpus / 3600


def summarize(times: list[float]) -> dict:
    """Per-step timing summary. Spread across steps is PRECISION of this run,
    not accuracy of the estimate -- the label says so on purpose."""
    t = sorted(times)
    n = len(t)
    return {
        "median": t[n // 2],
        "p10": t[int(0.1 * n)],
        "p90": t[min(n - 1, int(0.9 * n))],
        "mean": statistics.fmean(t),
        "cv_precision": (statistics.pstdev(t) / statistics.fmean(t)) if n > 1 else 0.0,
    }
