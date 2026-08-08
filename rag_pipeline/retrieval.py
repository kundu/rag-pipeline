"""Query retrieval over the built index. Deterministic: ties broken by
chunk_id so identical inputs always produce identical rankings."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .indexing import BM25_B, BM25_K1, _bucket, _embed, tokenize


def _bm25_score(query_tokens: list[str], chunk_id: str, index: dict) -> float:
    score = 0.0
    n = index["n_docs"]
    avgdl = index["avgdl"] or 1.0
    tf = index["tf"][chunk_id]
    dl = index["doc_len"][chunk_id]
    for token in query_tokens:
        f = tf.get(token, 0)
        if f == 0:
            continue
        df = index["df"].get(token, 0)
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        score += idf * (f * (BM25_K1 + 1)) / (
            f + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl)
        )
    return score


def _cosine_score(query_tokens: list[str], chunk_id: str, index: dict) -> float:
    qvec = _embed(Counter(query_tokens), index["df"], index["n_docs"])
    cvec = index["vectors"][chunk_id]
    return sum(a * b for a, b in zip(qvec, cvec))


def retrieve(
    queries: list[dict], chunks: list[dict], index: dict, top_k: int
) -> list[dict]:
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    scorer = _bm25_score if index["mode"] == "keyword" else _cosine_score

    results = []
    for query in queries:
        query_tokens = tokenize(query["question"])
        scored = [
            (scorer(query_tokens, cid, index), cid) for cid in chunk_by_id
        ]
        # Deterministic ordering: score desc, then chunk_id asc on ties.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        retrieved = [
            {
                "chunk_id": cid,
                "document_name": chunk_by_id[cid]["document_name"],
                "rank": rank,
                "retrieval_score": round(score, 6),
            }
            for rank, (score, cid) in enumerate(scored[:top_k], start=1)
        ]
        results.append(
            {
                "query_id": query["query_id"],
                "question": query["question"],
                "retrieved_chunks": retrieved,
            }
        )
    return results


def write_retrieval_results(results: list[dict], out_path: Path) -> None:
    out_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
