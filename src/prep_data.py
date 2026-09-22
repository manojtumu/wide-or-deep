"""Tokenize a pretraining corpus into flat uint16 shards.

Every measurement in this project so far has fed the model `torch.randint`
tokens, which is fine for throughput and useless for loss. This produces the
other half: real text, tokenized once, laid out so a training loop can memmap
it and read contiguous windows without a dataloader in the hot path.

Layout is deliberately dumb -- a directory of `train_00000.bin` files, each a
flat array of uint16 token ids, plus one `val.bin`. No index, no metadata
beyond `meta.json`. A shard is `np.memmap(path, np.uint16, mode="r")` and a
batch is a slice. That keeps the input pipeline off the critical path, which
matters when the thing being measured is how fast the GPU goes.

uint16 holds ids below 65536, so the tokenizer vocab must fit. GPT-2 BPE
(50257) does; the model pads that to a multiple of 64 for GEMM alignment.

    python src/prep_data.py --out data/fineweb --tokens 10e9
    python tools/launch.py --entry prep --hp tokens=10e9     (SageMaker; writes SM_OUTPUT_DATA_DIR)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

EOT = 50256          # GPT-2 <|endoftext|>, the document separator
VOCAB = 50257

_enc = None


def _encode(text: str) -> np.ndarray:
    """Tokenize one document, prefixed with EOT so documents stay separable.

    The encoder is built lazily per worker: tiktoken's Encoding does not
    survive being pickled into a pool, so each process makes its own.
    """
    global _enc
    if _enc is None:
        import tiktoken

        _enc = tiktoken.get_encoding("gpt2")
    ids = [EOT] + _enc.encode_ordinary(text)
    arr = np.array(ids, dtype=np.uint16)
    assert (arr < VOCAB).all(), "token id overflowed uint16"
    return arr


def _stream(dataset: str, name: str | None):
    from datasets import load_dataset

    ds = load_dataset(dataset, name=name, split="train", streaming=True)
    for row in ds:
        text = row.get("text")
        if text:
            yield text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--name", default="sample-10BT")
    p.add_argument("--out", default=os.environ.get("SM_OUTPUT_DATA_DIR", "data/fineweb"))
    p.add_argument("--tokens", type=float, default=10e9, help="total token budget")
    p.add_argument("--shard-tokens", type=float, default=100e6)
    p.add_argument("--val-tokens", type=float, default=10e6)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget, shard_cap, val_cap = int(args.tokens), int(args.shard_tokens), int(args.val_tokens)

    # One preallocated buffer, refilled per shard. Growing a list of arrays and
    # concatenating at the end would hold the whole corpus in RAM.
    buf = np.empty(shard_cap, dtype=np.uint16)
    fill = 0
    shard_idx = 0
    total = 0
    val_done = False
    t0 = time.time()

    def flush(path: Path, n: int) -> None:
        buf[:n].tofile(path)
        mb = n * 2 / 1e6
        el = time.time() - t0
        print(f"  wrote {path.name:<18} {n/1e6:>7.1f}M tokens  {mb:>7.0f} MB  "
              f"[{total/1e9:.2f}B total, {total/1e6/el:.1f}M tok/s]", flush=True)

    with mp.Pool(args.workers) as pool:
        # imap keeps the pool fed without materializing the whole stream.
        for arr in pool.imap(_encode, _stream(args.dataset, args.name), chunksize=16):
            pos = 0
            while pos < len(arr):
                cap = val_cap if not val_done else shard_cap
                take = min(cap - fill, len(arr) - pos)
                buf[fill:fill + take] = arr[pos:pos + take]
                fill += take
                pos += take
                total += take

                if fill == cap:
                    if not val_done:
                        flush(out / "val.bin", fill)
                        val_done = True
                    else:
                        flush(out / f"train_{shard_idx:05d}.bin", fill)
                        shard_idx += 1
                    fill = 0

            if total >= budget:
                break

    if fill:
        flush(out / f"train_{shard_idx:05d}.bin", fill)
        shard_idx += 1

    meta = {
        "dataset": args.dataset,
        "name": args.name,
        "tokenizer": "gpt2",
        "vocab": VOCAB,
        "eot": EOT,
        "dtype": "uint16",
        "total_tokens": total,
        "train_shards": shard_idx,
        "val_tokens": val_cap,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n{total/1e9:.2f}B tokens across {shard_idx} shards in {time.time()-t0:.0f}s")
    print(f"meta: {out/'meta.json'}")


if __name__ == "__main__":
    main()
