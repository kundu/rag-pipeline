#!/usr/bin/env python3
"""Validation command for the mini RAG pipeline.

Standalone:   python3 validate.py        (expects a fully finalised run)
In-pipeline:  run_checks(root, context="pipeline")  (called by run_pipeline.py
              after report generation, before VALIDATION_COMPLETE is recorded)

Prints one PASS/FAIL line per check; exits non-zero on any failure.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rag_pipeline.report import REQUIRED_SECTIONS
from rag_pipeline.state import STAGES, sha256_file

REQUIRED_ARTIFACTS = [
    "chunks.json",
    "index_metadata.json",
    "retrieval_results.json",
    "draft_answers.json",
    "review_overrides.json",
    "answer_audit.json",
    "revised_answers.json",
    "retrieval_metrics.json",
    "retrieval_error_analysis.json",
    "final_report.md",
    "llm_calls.jsonl",
    "pipeline_state.json",
]

CHUNK_FIELDS = {"chunk_id", "document_name", "start_char", "end_char", "text"}
RETRIEVED_FIELDS = {"chunk_id", "document_name", "rank", "retrieval_score"}
DRAFT_FIELDS = {"query_id", "answer", "label", "citations", "reasoning_summary"}
AUDIT_FIELDS = {
    "query_id", "audit_label", "support_assessment", "citation_check",
    "hallucination_risk", "recommended_fix", "final_context_chunk_ids",
}
LLM_CALL_FIELDS = {
    "stage", "query_id", "timestamp", "provider", "model", "prompt_hash",
    "input_artifacts", "output_artifact",
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _load(root: Path):
    """Load every artifact once; returns dict or raises with a clear message."""
    data = {}
    data["queries"] = json.loads((root / "queries.json").read_text())["queries"]
    data["policy"] = json.loads((root / "policy.json").read_text())
    data["chunks"] = json.loads((root / "chunks.json").read_text())
    data["index_meta"] = json.loads((root / "index_metadata.json").read_text())
    data["retrieval"] = json.loads((root / "retrieval_results.json").read_text())
    data["drafts"] = json.loads((root / "draft_answers.json").read_text())
    data["review"] = json.loads((root / "review_overrides.json").read_text())
    data["audits"] = json.loads((root / "answer_audit.json").read_text())
    data["revised"] = json.loads((root / "revised_answers.json").read_text())
    data["metrics"] = json.loads((root / "retrieval_metrics.json").read_text())
    data["errors"] = json.loads((root / "retrieval_error_analysis.json").read_text())
    data["state"] = json.loads((root / "pipeline_state.json").read_text())
    data["report"] = (root / "final_report.md").read_text()
    data["llm_calls"] = [
        json.loads(line)
        for line in (root / "llm_calls.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return data


def run_checks(root: Path = ROOT, context: str = "standalone") -> list[CheckResult]:
    results: list[CheckResult] = []
    check = lambda name, ok, detail="": results.append(  # noqa: E731
        CheckResult(name, bool(ok), detail)
    )

    # 1. Artifacts exist + parse + schema fields -----------------------------
    missing = [a for a in REQUIRED_ARTIFACTS if not (root / a).exists()]
    check("01a artifacts exist", not missing, f"missing: {missing}")
    if missing:
        return results
    try:
        d = _load(root)
    except (json.JSONDecodeError, KeyError) as exc:
        check("01b artifacts parse", False, str(exc))
        return results
    check("01b artifacts parse", True)

    bad_schema = []
    for chunk in d["chunks"]:
        if not CHUNK_FIELDS <= set(chunk):
            bad_schema.append(f"chunk {chunk.get('chunk_id')}")
    for rec in d["retrieval"]:
        for item in rec.get("retrieved_chunks", []):
            if not RETRIEVED_FIELDS <= set(item):
                bad_schema.append(f"retrieved in {rec.get('query_id')}")
    for draft in d["drafts"]:
        if not DRAFT_FIELDS <= set(draft):
            bad_schema.append(f"draft {draft.get('query_id')}")
    for audit in d["audits"]:
        if not AUDIT_FIELDS <= set(audit):
            bad_schema.append(f"audit {audit.get('query_id')}")
    for call in d["llm_calls"]:
        if not LLM_CALL_FIELDS <= set(call):
            bad_schema.append(f"llm_call {call.get('stage')}/{call.get('query_id')}")
    check("01c record schema fields", not bad_schema, f"bad: {bad_schema[:5]}")

    # 2. Inputs read from disk ------------------------------------------------
    inputs = d["state"].get("inputs", {})
    docs_dir = root / "documents"
    input_ok = (
        bool(inputs.get("documents"))
        and inputs.get("queries", {}).get("sha256")
        and inputs.get("policy", {}).get("sha256")
    )
    hash_ok = input_ok and all(
        (docs_dir / name).exists() and sha256_file(docs_dir / name) == digest
        for name, digest in inputs["documents"].items()
    )
    doc_names = {p.name for p in docs_dir.glob("*.txt")}
    chunk_docs_ok = all(c["document_name"] in doc_names for c in d["chunks"])
    check(
        "02 inputs read from disk (paths+hashes recorded, chunks map to real docs)",
        input_ok and hash_ok and chunk_docs_ok,
        "state.inputs incomplete, hash mismatch, or chunk references unknown document",
    )

    # 3. Chunking before any LLM call ------------------------------------------
    stage_ts = {s["stage"]: s["timestamp"] for s in d["state"]["stages"]}
    llm_ts = [c["timestamp"] for c in d["llm_calls"]]
    check(
        "03 chunking happened before any LLM call",
        "DOCUMENTS_CHUNKED" in stage_ts
        and (not llm_ts or stage_ts["DOCUMENTS_CHUNKED"] < min(llm_ts)),
        "DOCUMENTS_CHUNKED missing or later than first LLM call",
    )

    # 4. Every query has retrieval results ---------------------------------------
    retrieved_by_qid = {r["query_id"]: r["retrieved_chunks"] for r in d["retrieval"]}
    if d["chunks"]:
        empty = [
            q["query_id"] for q in d["queries"]
            if not retrieved_by_qid.get(q["query_id"])
        ]
        check("04 every query has >=1 retrieved chunk", not empty, f"empty: {empty}")
    else:
        check("04 every query has >=1 retrieved chunk", True,
              "skipped: corpus is empty")

    # 5. Labels allowed; citations subset + capped --------------------------------
    allowed = set(d["policy"]["answer_policy"]["allowed_labels"])
    max_cit = d["policy"]["answer_policy"]["max_citations_per_answer"]
    final_ctx = d["review"]["final_contexts"]
    label_bad, citation_bad = [], []
    for draft in d["drafts"]:
        qid = draft["query_id"]
        if draft["label"] not in allowed:
            label_bad.append(f"{qid}:{draft['label']}")
        rids = {c["chunk_id"] for c in retrieved_by_qid.get(qid, [])}
        if not set(draft["citations"]) <= rids or len(draft["citations"]) > max_cit:
            citation_bad.append(qid)
    for rev in d["revised"]:
        qid = rev["query_id"]
        if rev["label"] not in allowed:
            label_bad.append(f"revised {qid}:{rev['label']}")
        if (not set(rev["citations"]) <= set(final_ctx.get(qid, []))
                or len(rev["citations"]) > max_cit):
            citation_bad.append(f"revised {qid}")
    check("05a draft/revised labels within allowed_labels", not label_bad,
          f"bad: {label_bad}")
    check("05b citations subset of retrieved/final context and capped",
          not citation_bad, f"bad: {citation_bad}")

    # 6. Per-query LLM records per stage ---------------------------------------
    qids = [q["query_id"] for q in d["queries"]]
    by_stage: dict[str, list] = {}
    for call in d["llm_calls"]:
        by_stage.setdefault(call["stage"], []).append(call["query_id"])
    draft_calls = by_stage.get("draft_answer", [])
    audit_calls = by_stage.get("audit", [])
    revised_calls = set(by_stage.get("revised_answer", []))
    revised_qids = {r["query_id"] for r in d["revised"]}
    check(
        "06a one draft_answer LLM record per query",
        all(draft_calls.count(q) >= 1 for q in qids)
        and set(draft_calls) <= set(qids),
        f"draft calls: {draft_calls}",
    )
    check(
        "06b one audit LLM record per query (not batched)",
        all(audit_calls.count(q) >= 1 for q in qids)
        and set(audit_calls) <= set(qids),
        f"audit calls: {audit_calls}",
    )
    check(
        "06c revised_answer LLM records match revised_answers.json",
        revised_qids <= revised_calls | set() if revised_qids else True,
        f"revised entries {revised_qids} vs calls {revised_calls}",
    )

    # 7. Audit after human review --------------------------------------------------
    audit_ts = [c["timestamp"] for c in d["llm_calls"] if c["stage"] == "audit"]
    check(
        "07 audit ran after human review",
        "HUMAN_REVIEW_COMPLETE" in stage_ts
        and audit_ts
        and stage_ts["HUMAN_REVIEW_COMPLETE"] < min(audit_ts),
        "HUMAN_REVIEW_COMPLETE missing or audit call precedes it",
    )

    # 8. Overrides saved and applied to audit inputs --------------------------------
    audit_by_qid = {a["query_id"]: a for a in d["audits"]}
    override_bad = []
    # Every audit record must have used exactly the reviewed final context —
    # overridden queries AND untouched ones (final_contexts covers all).
    for qid, expected_ids in final_ctx.items():
        audit_rec = audit_by_qid.get(qid)
        if not audit_rec or audit_rec["final_context_chunk_ids"] != expected_ids:
            override_bad.append(qid)
    audit_inputs_ok = all(
        "review_overrides.json" in c["input_artifacts"]
        for c in d["llm_calls"] if c["stage"] == "audit"
    )
    check(
        "08 overrides saved and applied to audit inputs",
        not override_bad and audit_inputs_ok,
        f"mismatched: {override_bad}; audit inputs list review_overrides.json: {audit_inputs_ok}",
    )

    # 9. Report reflects reviewed final context --------------------------------------
    report_bad = []
    for override in d["review"]["overrides"]:
        for cid in override["chunk_ids"]:
            if cid not in d["report"]:
                report_bad.append(f"{override['query_id']}:{cid}")
    check(
        "09 final report reflects reviewed final context",
        not report_bad,
        f"override chunk ids missing from report: {report_bad}",
    )

    # 10. Retrieval mode recorded and consistent; chunks hash matches ------------------
    mode_ok = d["index_meta"].get("mode") == d["state"].get("mode")
    hash_match = d["index_meta"].get("chunks_sha256") == sha256_file(root / "chunks.json")
    check(
        "10 retrieval mode recorded consistently and chunks hash matches index",
        mode_ok and hash_match,
        f"mode index={d['index_meta'].get('mode')} state={d['state'].get('mode')}, hash match={hash_match}",
    )

    # 11. Report contains all six required sections --------------------------------------
    missing_sections = [
        s for s in REQUIRED_SECTIONS if f"## {s}" not in d["report"]
    ]
    check("11 report contains six required sections", not missing_sections,
          f"missing: {missing_sections}")

    # 12. Stage sequence exact and monotonic ------------------------------------------------
    recorded = [s["stage"] for s in d["state"]["stages"]]
    timestamps = [s["timestamp"] for s in d["state"]["stages"]]
    expected_through = (
        STAGES.index("FINAL_REPORT_GENERATED") + 1
        if context == "pipeline"
        else len(STAGES)
    )
    sequence_ok = recorded == STAGES[:expected_through] or (
        context == "pipeline" and recorded == STAGES[: expected_through]
    )
    monotonic = all(a <= b for a, b in zip(timestamps, timestamps[1:]))
    check(
        "12 stage sequence matches spec order with monotonic timestamps",
        sequence_ok and monotonic,
        f"recorded: {recorded}",
    )

    return results


def main() -> int:
    checks = run_checks(ROOT, context="standalone")
    failed = [c for c in checks if not c.ok]
    for c in checks:
        print(("PASS " if c.ok else "FAIL ") + c.name + ("" if c.ok else f" — {c.detail}"))
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
