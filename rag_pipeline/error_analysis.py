"""Retrieval error analysis, driven by observable evidence only: retrieval
scores/ranks, chunk boundaries vs expected phrases, and audit outcomes.
Classifies failures as ranking | chunking | ambiguity | corpus_gap."""
from __future__ import annotations

import json
from pathlib import Path

AMBIGUITY_SCORE_THRESHOLD = 1.0  # low lexical-overlap signal for BM25 scores


def _phrase_in_doc(phrase: str, doc_name: str, documents_dir: Path) -> bool:
    path = documents_dir / doc_name
    if not path.exists():
        return False
    return phrase.lower() in path.read_text(encoding="utf-8").lower()


def _phrase_in_some_chunk(phrase: str, doc_name: str, chunks: list[dict]) -> list[str]:
    return [
        c["chunk_id"]
        for c in chunks
        if c["document_name"] == doc_name and phrase.lower() in c["text"].lower()
    ]


def analyse_failures(
    queries: list[dict],
    retrieval_results: list[dict],
    chunks: list[dict],
    audits: list[dict],
    drafts: list[dict],
    metrics: dict,
    documents_dir: Path,
    out_path: Path,
) -> list[dict]:
    retrieved_by_qid = {
        r["query_id"]: r["retrieved_chunks"] for r in retrieval_results
    }
    audit_by_qid = {a["query_id"]: a for a in audits}
    draft_by_qid = {d["query_id"]: d for d in drafts}
    metric_by_qid = (
        {}
        if metrics.get("skipped")
        else {p["query_id"]: p for p in metrics.get("per_query", [])}
    )

    entries = []
    for query in queries:
        qid = query["query_id"]
        audit = audit_by_qid.get(qid, {})
        draft = draft_by_qid.get(qid, {})
        metric = metric_by_qid.get(qid)

        # Failure signals — all observable from artifacts.
        signals = []
        if metric and metric["hit_at_k"] == 0:
            signals.append("retrieval miss (hit@k=0)")
        if audit.get("audit_label") == "fail":
            signals.append("audit_label=fail")
        if audit.get("hallucination_risk") == "high":
            signals.append("hallucination_risk=high")
        if draft.get("label") and draft["label"] != "supported":
            signals.append(f"draft label={draft['label']}")
        if not signals:
            continue

        retrieved = retrieved_by_qid.get(qid, [])
        retrieved_ids = [c["chunk_id"] for c in retrieved]
        max_score = max((c["retrieval_score"] for c in retrieved), default=0.0)

        failure_type = None
        description = None
        for evidence in query.get("expected_evidence", []):
            phrase, doc = evidence["phrase"], evidence["document_name"]
            if not _phrase_in_doc(phrase, doc, documents_dir):
                failure_type = "corpus_gap"
                description = (
                    f"Expected evidence {phrase!r} does not exist in {doc} — "
                    f"the corpus cannot answer this. Signals: {signals}"
                )
                break
            holding = _phrase_in_some_chunk(phrase, doc, chunks)
            if not holding:
                failure_type = "chunking"
                description = (
                    f"Evidence {phrase!r} exists in {doc} but no single chunk "
                    f"contains it — split across chunk boundaries. Signals: {signals}"
                )
                break
            if not any(cid in retrieved_ids for cid in holding):
                failure_type = "ranking"
                description = (
                    f"Chunk(s) {holding} contain the evidence {phrase!r} but were "
                    f"not retrieved in top-{len(retrieved_ids)} "
                    f"(retrieved: {retrieved_ids}). Signals: {signals}"
                )
                break

        if failure_type is None:
            if not query.get("expected_evidence"):
                if draft.get("label") == "not_in_corpus":
                    failure_type = "corpus_gap"
                    description = (
                        f"No evidence annotations; draft labelled not_in_corpus and "
                        f"max retrieval score was {max_score}. Signals: {signals}"
                    )
                else:
                    failure_type = "ambiguity"
                    description = (
                        f"No evidence annotations; failure signals present with max "
                        f"retrieval score {max_score}. Signals: {signals}"
                    )
            elif max_score < AMBIGUITY_SCORE_THRESHOLD:
                failure_type = "ambiguity"
                description = (
                    f"Evidence was retrievable but lexical overlap is weak "
                    f"(max score {max_score} < {AMBIGUITY_SCORE_THRESHOLD}). "
                    f"Signals: {signals}"
                )
            else:
                failure_type = "ambiguity"
                description = (
                    f"Evidence was retrieved (max score {max_score}) yet the answer "
                    f"still failed — question/answer mismatch rather than a "
                    f"retrieval defect. Signals: {signals}"
                )

        entries.append(
            {
                "query_id": qid,
                "failure_type": failure_type,
                "description": description,
            }
        )

    out_path.write_text(
        json.dumps(entries, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return entries
