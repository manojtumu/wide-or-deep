"""A plain decoder-only transformer. One GPU's worth, replicated by DDP.

No tensor parallelism, no pipeline stages, no parameter sharding. A 400M model
with Adam states is about 7 GB and these run on 80 GB cards, so there was never
a memory reason to shard -- and sharding is what produced the study's worst
remaining confound, since FSDP wraps per block and a 32-layer model then issues
four times as many collectives as an 8-layer one for identical gradient volume.

Multi-head attention and tied embeddings, matching the scaling-law setup. See
shapes.py for why both choices matter to the comparison.

Initialisation is seeded per weight by name, so a given seed gives bit-identical
weights regardless of how many GPUs run it.
"""
from __future__ import annotations

import math
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from shapes import HEAD_DIM, Shape

_KEY_PREFIX = ""


def set_init_seed(seed: int) -> None:
    global _KEY_PREFIX
    _KEY_PREFIX = f"s{seed}:" if seed else ""


def _init(key: str, out_f: int, in_f: int, std: float, dtype, device) -> nn.Parameter:
    # crc32, not hash(): Python's str hash is salted per process.
    g = torch.Generator().manual_seed(zlib.crc32((_KEY_PREFIX + key).encode()) & 0x7FFFFFFF)
    w = torch.empty(out_f, in_f).normal_(0.0, std, generator=g)
    return nn.Parameter(w.to(device=device, dtype=dtype))


class Linear(nn.Module):
    """nn.Linear without bias, with deterministic name-keyed init."""

    def __init__(self, key, in_f, out_f, dtype, device, std=0.02):
        super().__init__()
        self.weight = _init(key, out_f, in_f, std, dtype, device)

    def forward(self, x):
        return F.linear(x, self.weight)


class RMSNorm(nn.Module):
    def __init__(self, d, dtype, device, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, dtype=dtype, device=device))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def rope_cache(seq, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM))
    f = torch.outer(torch.arange(seq, device=device).float(), inv)      # [T, hd/2]
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    # slice the cache to the actual length: it is built once at the max sequence
    # length, and a shorter batch (validation, a smoke test) would otherwise
    # broadcast against the wrong dimension instead of erroring cleanly
    T, half = x.shape[1], HEAD_DIM // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos[None, :T, None, :].to(x.dtype), sin[None, :T, None, :].to(x.dtype)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Attention(nn.Module):
    def __init__(self, shape: Shape, layer: int, dtype, device):
        super().__init__()
        self.H = shape.n_heads
        self.qkv = Linear(f"L{layer}.qkv", shape.d, 3 * self.H * HEAD_DIM, dtype, device)
        self.o = Linear(f"L{layer}.o", self.H * HEAD_DIM, shape.d, dtype, device,
                        std=0.02 / math.sqrt(2 * shape.n_layers))

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(self.H * HEAD_DIM, dim=-1)
        q = apply_rope(q.view(B, T, self.H, HEAD_DIM), cos, sin)
        k = apply_rope(k.view(B, T, self.H, HEAD_DIM), cos, sin)
        v = v.view(B, T, self.H, HEAD_DIM)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))                # [B, H, T, hd]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, self.H * HEAD_DIM))


class MLP(nn.Module):
    def __init__(self, shape: Shape, layer: int, dtype, device):
        super().__init__()
        self.gate_up = Linear(f"L{layer}.gate_up", shape.d, 2 * shape.ffn, dtype, device)
        self.down = Linear(f"L{layer}.down", shape.ffn, shape.d, dtype, device,
                           std=0.02 / math.sqrt(2 * shape.n_layers))
        self.ffn = shape.ffn

    def forward(self, x):
        g, u = self.gate_up(x).split(self.ffn, dim=-1)
        return self.down(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, shape, layer, dtype, device):
        super().__init__()
        self.n1 = RMSNorm(shape.d, dtype, device)
        self.attn = Attention(shape, layer, dtype, device)
        self.n2 = RMSNorm(shape.d, dtype, device)
        self.mlp = MLP(shape, layer, dtype, device)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class Transformer(nn.Module):
    def __init__(self, shape: Shape, seq: int, dtype=torch.bfloat16, device="cuda",
                 rope_theta=10000.0, ce_chunks=4):
        super().__init__()
        self.shape, self.ce_chunks = shape, ce_chunks
        self.embed = nn.Embedding(shape.vocab, shape.d, dtype=dtype, device=device)
        self.embed.weight = _init("embed", shape.vocab, shape.d, 0.02, dtype, device)
        self.blocks = nn.ModuleList(Block(shape, i, dtype, device) for i in range(shape.n_layers))
        self.norm = RMSNorm(shape.d, dtype, device)
        cos, sin = rope_cache(seq, rope_theta, device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.grad_ckpt = False

    def _loss(self, h, targets):
        """Cross-entropy on row chunks under activation checkpointing, so the fp32
        [B*T, vocab] logits -- 6.6 GB at 16x2048x50k, held twice by the naive path --
        exist one chunk at a time and are recomputed in backward."""
        h = h.reshape(-1, h.shape[-1])
        t = targets.reshape(-1)
        rows = -(-h.shape[0] // self.ce_chunks)

        def piece(hc, tc):
            return F.cross_entropy(F.linear(hc, self.embed.weight).float(), tc, reduction="sum")

        total = h.new_zeros((), dtype=torch.float32)
        for hc, tc in zip(h.split(rows), t.split(rows)):
            total = total + torch.utils.checkpoint.checkpoint(piece, hc, tc, use_reentrant=False)
        return total / h.shape[0]

    def forward(self, ids, targets=None):
        x = self.embed(ids)
        for b in self.blocks:
            if self.grad_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(b, x, self.cos, self.sin, use_reentrant=False)
            else:
                x = b(x, self.cos, self.sin)
        x = self.norm(x)
        if targets is None:
            return F.linear(x, self.embed.weight)        # tied: one matrix, both roles
        return self._loss(x, targets)
