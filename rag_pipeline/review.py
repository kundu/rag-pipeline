"""Human review checkpoint: display retrieval + draft labels, accept overrides.

The reviewer may force a final context (list of chunk_ids) for one or more
queries. review_overrides.json records both the raw overrides and the
resolved `final_contexts` for every query — the single source of truth for
downstream audit and reporting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .state import utc_now

REVIEW_PROMPT = (
    "Do you want to override retrieved chunks for any query before audit?\n"
    "Enter query_id and comma-separated chunk_ids to force as final context, "
    "or press Enter to continue.\n"
)


def display_review_table(
    retrieval_results: list[dict], drafts: list[dict]
) -> None:
    label_by_qid = {d["query_id"]: d["label"] for d in drafts}
    print("\n" + "=" * 72)
    print("HUMAN REVIEW CHECKPOINT — retrieval results and draft labels")
    print("=" * 72)
    for result in retrieval_results:
        qid = result["query_id"]
        print(f"\n{qid}: {result['question']}")
        print(f"  draft label: {label_by_qid.get(qid, '<missing>')}")
        for chunk in result["retrieved_chunks"]:
            print(
                f"  rank {chunk['rank']}: {chunk['chunk_id']}  "
                f"(doc={chunk['document_name']}, score={chunk['retrieval_score']})"
            )
    print()


def _parse_override_line(line: str, valid_qids: set, valid_chunk_ids: set):
    """Returns (query_id, [chunk_ids]) or raises ValueError."""
    parts = line.split(None, 1)
    if len(parts) != 2:
        raise ValueError(
            "expected: <query_id> <chunk_id>[,<chunk_id>...] — e.g. "
            "Q2 security::c0000,billing::c0000"
        )
    qid, ids_part = parts[0], parts[1]
    if qid not in valid_qids:
        raise ValueError(f"unknown query_id {qid!r}")
    chunk_ids = [c.strip() for c in ids_part.split(",") if c.strip()]
    if not chunk_ids:
        raise ValueError("no chunk_ids given")
    unknown = [c for c in chunk_ids if c not in valid_chunk_ids]
    if unknown:
        raise ValueError(f"chunk_id(s) not present in chunks.json: {unknown}")
    return qid, chunk_ids


def run_review(
    retrieval_results: list[dict],
    drafts: list[dict],
    chunk_by_id: dict,
    out_path: Path,
    auto_approve: bool = False,
) -> dict:
    display_review_table(retrieval_results, drafts)

    valid_qids = {r["query_id"] for r in retrieval_results}
    valid_chunk_ids = set(chunk_by_id)
    overrides: dict[str, list[str]] = {}

    if not auto_approve:
        while True:
            print(REVIEW_PROMPT, end="")
            try:
                line = input().strip()
            except EOFError:
                break
            if not line:
                break
            try:
                qid, chunk_ids = _parse_override_line(
                    line, valid_qids, valid_chunk_ids
                )
            except ValueError as exc:
                print(f"  !! invalid override: {exc}", file=sys.stderr)
                continue
            overrides[qid] = chunk_ids
            print(f"  -> override recorded for {qid}: {chunk_ids}")

    final_contexts = {
        r["query_id"]: overrides.get(
            r["query_id"], [c["chunk_id"] for c in r["retrieved_chunks"]]
        )
        for r in retrieval_results
    }
    record = {
        "reviewed_at": utc_now(),
        "auto_approved": auto_approve,
        "overrides": [
            {"query_id": qid, "chunk_ids": ids}
            for qid, ids in sorted(overrides.items())
        ],
        "final_contexts": final_contexts,
    }
    out_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return record
