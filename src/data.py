"""Memmapped token shards, sampled statelessly.

The whole global batch for a step is a pure function of (seed, step): one RNG
seeded from those picks all its offsets into the concatenated token stream, and
each (rank, micro) takes a fixed slice, so the data is identical whatever the
GPU count or micro-batch. Nothing is iterated, so there is no
dataloader state to checkpoint -- resuming at step N reproduces exactly the
batches a run that never stopped would have seen at step N. Random-offset
sampling with replacement over ~10B tokens is the nanoGPT convention and the
repetition rate over an 8-10B-token run is negligible.

Shards are the flat uint16 files that prep_data.py writes; reading a window
is a memmap slice, so the input side never appears in the step time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class TokenShards:
    def __init__(self, root: str | Path):
        root = Path(root)
        self.meta = json.loads((root / "meta.json").read_text())
        self.vocab = self.meta["vocab"]
        files = sorted(root.glob("train_*.bin"))
        assert files, f"no train_*.bin under {root}"
        self.shards = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        self.sizes = np.array([len(m) for m in self.shards], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.total = int(self.sizes.sum())
        vpath = root / "val.bin"
        self.val = np.memmap(vpath, dtype=np.uint16, mode="r") if vpath.exists() else None

    def window(self, off: int, n: int) -> np.ndarray:
        """n tokens starting at global offset off, crossing shard edges if needed."""
        i = int(np.searchsorted(self.starts, off, side="right") - 1)
        out = []
        need = n
        while need > 0:
            local = off - int(self.starts[i])
            take = min(need, int(self.sizes[i]) - local)
            out.append(self.shards[i][local:local + take])
            need -= take
            off += take
            i += 1
        return np.concatenate(out) if len(out) > 1 else out[0]

    def batch(self, seed: int, step: int, rank: int, micro: int, B: int, seq: int,
              world: int = 1, accum: int = 1):
        """The step's whole global batch (world x accum x B sequences) is drawn
        from rng(seed, step) and this call takes slice (rank*accum + micro).
        Data order therefore depends only on seed, step and global batch --
        not on GPU count, micro-batch or accumulation -- so every shape, every
        hardware, and every resume sees the same sequences at the same step."""
        rng = np.random.default_rng([seed, step])
        offs = rng.integers(0, self.total - seq - 1, size=world * accum * B)
        piece = rank * accum + micro
        mine = offs[piece * B:(piece + 1) * B]
        x = np.stack([self.window(int(o), seq + 1) for o in mine]).astype(np.int64)
        x = torch.from_numpy(x)
        return x[:, :-1], x[:, 1:]

    def val_batch(self, i: int, B: int, seq: int):
        """Fixed contiguous windows, so every evaluation scores the same tokens."""
        assert self.val is not None, "no val.bin"
        span = B * (seq + 1)
        start = (i * span) % max(1, len(self.val) - span)
        x = np.asarray(self.val[start:start + span]).astype(np.int64).reshape(B, seq + 1)
        x = torch.from_numpy(x)
        return x[:, :-1], x[:, 1:]


def write_tiny_corpus(root: str | Path, vocab: int = 50304, tokens: int = 200_000, seed: int = 0):
    """A synthetic corpus in the exact on-disk format, for local smoke tests."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for k in range(2):
        rng.integers(0, vocab, size=tokens // 2, dtype=np.uint16).tofile(root / f"train_{k:05d}.bin")
    rng.integers(0, vocab, size=tokens // 4, dtype=np.uint16).tofile(root / "val.bin")
    (root / "meta.json").write_text(json.dumps({"vocab": vocab, "tokenizer": "synthetic",
                                                "total_tokens": tokens, "dtype": "uint16"}))
    return root
