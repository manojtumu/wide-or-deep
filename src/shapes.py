"""What the three models ARE. Nothing here imports torch.

One decision defines the study: hold a parameter budget fixed and spend it on a
tall narrow model or a short wide one. This file is that decision, written down.

Matched quantity is TRANSFORMER BLOCKS -- Kaplan's N, "the number of model
parameters, excluding all vocabulary and positional embeddings". Not blocks plus
the output head, which is what an earlier version matched and which let the
narrow shape carry 18% more of the quantity the scaling law is actually about.

Attention is multi-head, as in that literature. It matters more than it looks:
MHA attention costs 4*L*d^2, which depends only on L*d^2, so any two shapes on
the same L*d^2 curve get identical attention AND identical feed-forward. Deep
(32 x 1024^2) and wide (8 x 2048^2) are exactly such a pair. Grouped-query
attention breaks that -- its K and V cost 2*L*d*n_kv*head_dim and scale with
L*d instead, charging a deep shape twice over for the same key-value state.

Embeddings are tied, also as in that literature, so one d x vocab matrix serves
as both lookup and output projection. That makes the matmul count and the total
count the same number, which removes a whole class of "which count do you mean"
confusion.

What cannot be equalised: the table is d x vocab, so a wider model always
carries a proportionally bigger one. Kaplan's own fortyfold aspect sweep had
that too, at 3.4x against our 2x.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

HEAD_DIM = 64
D_STEP = 512         # d snaps here so n_heads = d/64 stays a round number
FFN_STEP = 128       # MLP width snaps here for GEMM alignment
VOCAB = 50304        # GPT-2 BPE (50257) padded to a multiple of 64
SEQ = 2048

ASPECT = {"deep": 32, "balanced": 128, "wide": 256}


@dataclass(frozen=True)
class Shape:
    d: int
    n_layers: int
    ffn: int
    vocab: int = VOCAB

    def __post_init__(self):
        assert self.d % HEAD_DIM == 0, f"d={self.d} not a multiple of head_dim {HEAD_DIM}"
        assert 2 * self.d <= self.ffn <= 6 * self.d, f"ffn={self.ffn} outside [2d, 6d] for d={self.d}"

    @property
    def n_heads(self) -> int:
        return self.d // HEAD_DIM

    @property
    def aspect(self) -> float:
        return self.d / self.n_layers

    def attn_params(self) -> int:
        """q, k, v, o -- each d x d under multi-head attention."""
        return 4 * self.d * self.d

    def mlp_params(self) -> int:
        return 3 * self.d * self.ffn                   # SwiGLU: gate, up, down

    @property
    def blocks(self) -> int:
        """Kaplan's N: everything except the vocabulary table."""
        return self.n_layers * (self.attn_params() + self.mlp_params())

    @property
    def table(self) -> int:
        """One d x vocab matrix, tied: lookup on the way in, projection on the way out."""
        return self.d * self.vocab

    @property
    def params(self) -> int:
        """Every parameter in the model. Tied, so this is also the matmul count."""
        return self.blocks + self.table

    def flops_per_token(self, seq: int = SEQ) -> int:
        """6 per parameter for fwd+bwd through every matmul, plus attention scores.

        The second term is 12*L*d*seq: QK^T and AV are each 2*seq*d per token
        forward over the full T x T matrix, tripled for backward. This is the
        PaLM / Chinchilla / Megatron convention that published MFU figures use;
        a causal flash-attention kernel executes roughly half of it. Leaving it out
        -- which the usual 6N approximation does -- understates a deep shape by
        far more than a wide one, because it scales with layers times width.
        """
        return 6 * self.params + 12 * self.n_layers * self.d * seq

    def label(self) -> str:
        return f"{self.d}x{self.n_layers}"

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def _snap(x: float, step: int) -> int:
    return max(step, int(round(x / step)) * step)


def solve(target_blocks: float, aspect: float, d: int | None = None,
          n_layers: int | None = None, vocab: int = VOCAB) -> Shape:
    """Find (d, L, ffn) whose blocks hit the target at roughly the given aspect.

    N ~ 12*L*d^2 with d = aspect*L gives L = (N/12a^2)^(1/3) as a seed. d snaps
    to D_STEP, which moves the realised aspect, so ffn absorbs the error exactly:
    subtract attention, divide by 3*L*d, snap. Pass d and n_layers to pin the
    geometry and solve ffn alone.
    """
    if d is not None and n_layers is not None:
        ffn = _snap((target_blocks / n_layers - 4 * d * d) / (3 * d), FFN_STEP)
        return Shape(d, n_layers, ffn, vocab)
    L0 = max(2, round((target_blocks / (12 * aspect ** 2)) ** (1 / 3)))
    best = None
    for L in range(max(2, L0 - 4), L0 + 5):
        dd = _snap(aspect * L, D_STEP)
        ffn = _snap((target_blocks / L - 4 * dd * dd) / (3 * dd), FFN_STEP)
        if not (2 * dd <= ffn <= 6 * dd):
            continue
        s = Shape(dd, L, ffn, vocab)
        err = abs(s.blocks - target_blocks) / target_blocks
        # budget first, then fidelity to the requested aspect, then residual error
        key = (err > 0.01, abs(math.log(s.aspect / aspect)), err)
        if best is None or key < best[0]:
            best = (key, s)
    if best is None:
        raise ValueError(f"no shape near {target_blocks:.3g} blocks at aspect {aspect}")
    return best[1]


def shapes_for(target_blocks: float = 4e8) -> dict[str, Shape]:
    """The three models.

    Two passes. Solve the geometry at the nominal target, then re-aim at what the
    exactly-matchable pair actually hits: under MHA, deep and wide sit on the same
    L*d^2 curve and land on an identical block count once ffn snaps, while a round
    400M leaves the middle shape over a percent high. The target is a measured
    property of the grid, not a number from the title.
    """
    geom = {k: solve(target_blocks, a) for k, a in ASPECT.items()}
    pair = [solve(target_blocks, ASPECT[k], d=geom[k].d, n_layers=geom[k].n_layers).blocks
            for k in ("deep", "wide")]
    tgt = sum(pair) / len(pair)
    return {k: solve(tgt, ASPECT[k], d=g.d, n_layers=g.n_layers) for k, g in geom.items()}


if __name__ == "__main__":
    sh = shapes_for()
    print(f"{'':>10} {'d x L':>9} {'ffn':>6} {'ffn/d':>7} {'attn':>9} {'mlp':>9} "
          f"{'BLOCKS':>9} {'table':>8} {'TOTAL':>9} {'flops/tok':>11}")
    for k, s in sh.items():
        print(f"{k:>10} {s.d:>5}x{s.n_layers:<3} {s.ffn:>6} {s.ffn/s.d:>6.3f}x "
              f"{s.n_layers*s.attn_params()/1e6:>8.1f}M {s.n_layers*s.mlp_params()/1e6:>8.1f}M "
              f"{s.blocks/1e6:>8.1f}M {s.table/1e6:>7.1f}M {s.params/1e6:>8.1f}M "
              f"{s.flops_per_token()/1e9:>10.3f}e9")
    b = [s.blocks for s in sh.values()]
    print(f"\n  block spread {(max(b)/min(b)-1)*100:.2f}%")
