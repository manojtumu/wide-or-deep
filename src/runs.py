"""What a run IS. Nothing here imports torch.

A frozen config that fully determines an experiment. Its hash is the run id and
the checkpoint path, so a resubmitted job resumes rather than restarts, and two
runs with the same id are the same experiment by construction.

Deliberately small. The previous version carried tp, pp, dp, dp_mode, schedule,
bucket_mb and reduce_dtype because the study swept parallelism layouts. It does
not any more: one node, replicated data parallel, and the only thing that varies
between runs is the shape, the learning rate and the seed.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

from shapes import SEQ, Shape


@dataclass(frozen=True)
class Train:
    """One loss-curve run."""
    shape: Shape
    lr: float
    tokens: float = 8e9
    seed: int = 1
    seq: int = SEQ
    micro: int = 16                  # sequences per GPU per forward
    global_batch: int = 128          # sequences per optimizer step, all GPUs
    min_lr_ratio: float = 0.1
    warmup_frac: float = 0.02
    weight_decay: float = 0.1
    beta2: float = 0.95
    grad_clip: float = 1.0
    val_every: int = 500
    val_batches: int = 16
    log_every: int = 20
    ckpt_minutes: float = 20.0
    compile: bool = True
    tag: str = ""

    @property
    def steps(self) -> int:
        return math.ceil(self.tokens / (self.global_batch * self.seq))

    def accum(self, world: int) -> int:
        per_step = world * self.micro
        assert self.global_batch % per_step == 0, \
            f"global_batch {self.global_batch} not divisible by world*micro {per_step}"
        return self.global_batch // per_step

    @property
    def run_id(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]

    def label(self) -> str:
        return f"{self.shape.label()} lr{self.lr:.2e} seed{self.seed} {self.tokens/1e9:.0f}B"

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "Train":
        d = json.loads(s)
        return Train(shape=Shape(**d.pop("shape")), **d)


@dataclass(frozen=True)
class Bench:
    """One throughput measurement. Same model, timed rather than trained."""
    shape: Shape
    seq: int = SEQ
    batch: int = 16                  # sequences per GPU per forward
    steps: int = 40
    warmup: int = 10
    compile: bool = True
    seed: int = 0
    tag: str = ""

    @property
    def tokens_per_step_per_gpu(self) -> int:
        return self.batch * self.seq

    @property
    def run_id(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]

    def label(self) -> str:
        return f"{self.shape.label()} b{self.batch}" + (" compiled" if self.compile else " eager")

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "Bench":
        d = json.loads(s)
        return Bench(shape=Shape(**d.pop("shape")), **d)
