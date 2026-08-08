"""Deterministic retrieval metrics (hit@k, recall@k) computed in code from
optional `expected_evidence` annotations on queries. Gracefully skips when no
query carries annotations."""
from __future__ import annotations

import json
from pathlib import Path


def _evidence_found(evidence: dict, retrieved: list[dict], chunk_by_id: dict) -> bool:
    phrase = evidence["phrase"].lower()
    for item in retrieved:
        chunk = chunk_by_id[item["chunk_id"]]
        if (
            chunk["document_name"] == evidence["document_name"]
            and phrase in chunk["text"].lower()
        ):
            return True
    return False


def compute_metrics(
    queries: list[dict],
    retrieval_results: list[dict],
    chunk_by_id: dict,
    top_k: int,
    out_path: Path,
) -> dict:
    annotated = [q for q in queries if q.get("expected_evidence")]
    if not annotated:
        result = {
            "skipped": True,
            "reason": "no expected_evidence annotations present in queries.json",
        }
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result

    retrieved_by_qid = {
        r["query_id"]: r["retrieved_chunks"] for r in retrieval_results
    }
    per_query = []
    for query in annotated:
        qid = query["query_id"]
        retrieved = retrieved_by_qid.get(qid, [])
        found = [
            _evidence_found(ev, retrieved, chunk_by_id)
            for ev in query["expected_evidence"]
        ]
        per_query.append(
            {
                "query_id": qid,
                "num_expected_evidence": len(found),
                "num_found": sum(found),
                "hit_at_k": 1 if any(found) else 0,
                "recall_at_k": round(sum(found) / len(found), 4),
            }
        )

    result = {
        "skipped": False,
        "k": top_k,
        "num_annotated_queries": len(per_query),
        "per_query": per_query,
        "aggregate": {
            "hit_at_k": round(
                sum(p["hit_at_k"] for p in per_query) / len(per_query), 4
            ),
            "recall_at_k": round(
                sum(p["recall_at_k"] for p in per_query) / len(per_query), 4
            ),
        },
    }
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
