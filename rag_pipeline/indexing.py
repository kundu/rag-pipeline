"""Retrieval index construction. Two modes, both pure stdlib and deterministic:

- keyword:   BM25 statistics (term/document frequencies, lengths)
- embedding: hashed TF-IDF vectors (md5 token hashing into fixed buckets,
             cosine similarity) — reproducible with no model downloads
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from .state import sha256_file, utc_now

TOKEN_RE = re.compile(r"\w+")
EMBED_DIM = 512
BM25_K1 = 1.5
BM25_B = 0.75

MODES = ("keyword", "embedding")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _bucket(token: str) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % EMBED_DIM


def _embed(tf: Counter, df: Counter, n_docs: int) -> list[float]:
    vec = [0.0] * EMBED_DIM
    for token, freq in tf.items():
        idf = math.log((n_docs + 1) / (df.get(token, 0) + 1)) + 1.0
        vec[_bucket(token)] += freq * idf
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def build_index(chunks: list[dict], mode: str) -> dict:
    if mode not in MODES:
        raise ValueError(f"Unknown retrieval mode {mode!r}; expected one of {MODES}")

    tokens_by_chunk = {c["chunk_id"]: tokenize(c["text"]) for c in chunks}
    n_docs = len(chunks)
    df: Counter = Counter()
    for tokens in tokens_by_chunk.values():
        df.update(set(tokens))
    total_len = sum(len(t) for t in tokens_by_chunk.values())

    index = {
        "mode": mode,
        "n_docs": n_docs,
        "df": df,
        "avgdl": (total_len / n_docs) if n_docs else 0.0,
        "doc_len": {cid: len(t) for cid, t in tokens_by_chunk.items()},
        "tf": {cid: Counter(t) for cid, t in tokens_by_chunk.items()},
    }
    if mode == "embedding":
        index["vectors"] = {
            cid: _embed(index["tf"][cid], df, n_docs) for cid in tokens_by_chunk
        }
    return index


def write_index_metadata(
    index: dict, chunks_path: Path, documents: list[str], out_path: Path
) -> None:
    meta = {
        "mode": index["mode"],
        "num_chunks": index["n_docs"],
        "num_documents": len(documents),
        "vocabulary_size": len(index["df"]),
        "avg_chunk_tokens": round(index["avgdl"], 4),
        "parameters": {
            "tokenizer": r"lowercase \w+ regex",
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
            "embedding_dim": EMBED_DIM if index["mode"] == "embedding" else None,
            "embedding_hash": "md5-bucket tf-idf" if index["mode"] == "embedding" else None,
        },
        "chunks_sha256": sha256_file(chunks_path),
        "built_at": utc_now(),
    }
    out_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
