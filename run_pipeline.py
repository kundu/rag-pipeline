#!/usr/bin/env python3
"""Replayable mini RAG pipeline orchestrator.

Runs the full staged pipeline:
  INIT -> INPUTS_LOADED -> DOCUMENTS_CHUNKED -> INDEX_BUILT
       -> RETRIEVAL_COMPLETE -> DRAFT_ANSWERS_GENERATED
       -> HUMAN_REVIEW_COMPLETE -> ANSWERS_AUDITED
       -> FINAL_REPORT_GENERATED -> VALIDATION_COMPLETE -> RESULTS_FINALISED

Usage:
  python3 run_pipeline.py [--mode keyword|embedding] [--auto-approve]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rag_pipeline import audit, chunking, error_analysis, generate, indexing
from rag_pipeline import metrics as metrics_mod
from rag_pipeline import report, retrieval, review, revise
from rag_pipeline.llm import LLMClient, detect_provider
from rag_pipeline.state import StageMachine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("keyword", "embedding"),
        default=None,
        help="retrieval mode; overrides policy.json retrieval.mode",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="skip the interactive review prompt (records zero overrides)",
    )
    args = parser.parse_args()

    documents_dir = ROOT / "documents"
    queries_path = ROOT / "queries.json"
    policy_path = ROOT / "policy.json"
    for required in (documents_dir, queries_path, policy_path):
        if not required.exists():
            print(f"FATAL: required input missing: {required}", file=sys.stderr)
            return 2

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    queries = json.loads(queries_path.read_text(encoding="utf-8"))["queries"]
    retrieval_cfg = policy["retrieval"]
    mode = args.mode or retrieval_cfg.get("mode", "keyword")

    provider, model = detect_provider()
    print(f"LLM provider: {provider} (model: {model}) · retrieval mode: {mode}")

    # Fresh run: remove stale generated artifacts so state is unambiguous.
    for artifact in (
        "pipeline_state.json", "llm_calls.jsonl", "chunks.json",
        "index_metadata.json", "retrieval_results.json", "draft_answers.json",
        "review_overrides.json", "answer_audit.json", "revised_answers.json",
        "retrieval_metrics.json", "retrieval_error_analysis.json",
        "final_report.md",
    ):
        (ROOT / artifact).unlink(missing_ok=True)

    machine = StageMachine(ROOT, mode=mode, provider=provider)
    machine.set_provider(provider, model)
    machine.record("INIT")

    # -- Inputs ------------------------------------------------------------
    machine.record_inputs(documents_dir, queries_path, policy_path)
    machine.record("INPUTS_LOADED")

    # -- Deterministic chunking (before ANY LLM call) ----------------------
    chunks = chunking.chunk_documents(
        documents_dir,
        retrieval_cfg["chunk_size_chars"],
        retrieval_cfg["chunk_overlap_chars"],
    )
    chunking.write_chunks(chunks, ROOT / "chunks.json")
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    machine.record("DOCUMENTS_CHUNKED")
    print(f"Chunked {len(set(c['document_name'] for c in chunks))} documents "
          f"into {len(chunks)} chunks.")

    # -- Index --------------------------------------------------------------
    index = indexing.build_index(chunks, mode)
    indexing.write_index_metadata(
        index,
        ROOT / "chunks.json",
        sorted({c["document_name"] for c in chunks}),
        ROOT / "index_metadata.json",
    )
    machine.record("INDEX_BUILT")

    # -- Retrieval ------------------------------------------------------------
    retrieval_results = retrieval.retrieve(
        queries, chunks, index, retrieval_cfg["top_k"]
    )
    retrieval.write_retrieval_results(
        retrieval_results, ROOT / "retrieval_results.json"
    )
    machine.record("RETRIEVAL_COMPLETE")
    print(f"Retrieved top-{retrieval_cfg['top_k']} chunks for "
          f"{len(retrieval_results)} queries.")

    # -- Stage 1: draft answers -------------------------------------------------
    client = LLMClient(ROOT, provider, model)
    machine.require("DOCUMENTS_CHUNKED", "RETRIEVAL_COMPLETE")
    drafts = generate.generate_drafts(
        client, queries, retrieval_results, chunk_by_id, policy,
        ROOT / "draft_answers.json",
    )
    machine.record("DRAFT_ANSWERS_GENERATED")
    print(f"Generated {len(drafts)} draft answers.")

    # -- Human review checkpoint ---------------------------------------------------
    review_record = review.run_review(
        retrieval_results, drafts, chunk_by_id,
        ROOT / "review_overrides.json",
        auto_approve=args.auto_approve,
    )
    machine.record("HUMAN_REVIEW_COMPLETE")
    print(f"Review complete: {len(review_record['overrides'])} override(s).")

    # -- Stage 2: audit (uses final context after overrides) --------------------------
    machine.require("HUMAN_REVIEW_COMPLETE")
    audits = audit.audit_answers(
        client, queries, drafts, review_record["final_contexts"],
        chunk_by_id, policy, ROOT / "answer_audit.json",
    )
    machine.record("ANSWERS_AUDITED")
    print(f"Audited {len(audits)} answers "
          f"({sum(1 for a in audits if a['audit_label'] == 'pass')} pass).")

    # -- Post-audit deterministic analysis + revisions ---------------------------------
    metric_result = metrics_mod.compute_metrics(
        queries, retrieval_results, chunk_by_id, retrieval_cfg["top_k"],
        ROOT / "retrieval_metrics.json",
    )
    revised = revise.revise_answers(
        client, queries, drafts, audits, review_record["final_contexts"],
        chunk_by_id, policy, ROOT / "revised_answers.json",
    )
    if revised:
        print(f"Regenerated {len(revised)} answer(s) after audit failure.")
    failures = error_analysis.analyse_failures(
        queries, retrieval_results, chunks, audits, drafts, metric_result,
        documents_dir, ROOT / "retrieval_error_analysis.json",
    )

    # -- Final report (stage machine guarantees prerequisites) ----------------------------
    machine.require(
        "DOCUMENTS_CHUNKED", "RETRIEVAL_COMPLETE", "DRAFT_ANSWERS_GENERATED",
        "HUMAN_REVIEW_COMPLETE", "ANSWERS_AUDITED",
    )
    report.write_report(
        mode=mode, provider=provider, model=model,
        queries=queries, retrieval_results=retrieval_results,
        review_record=review_record, drafts=drafts, audits=audits,
        revised=revised, metrics=metric_result, error_analysis=failures,
        top_k=retrieval_cfg["top_k"], out_path=ROOT / "final_report.md",
    )
    machine.record("FINAL_REPORT_GENERATED")
    print("Final report written to final_report.md")

    # -- In-pipeline validation, then finalise -----------------------------------------
    import validate as validate_mod

    checks = validate_mod.run_checks(ROOT, context="pipeline")
    failed = [c for c in checks if not c.ok]
    for check in checks:
        print(("PASS " if check.ok else "FAIL ") + check.name
              + ("" if check.ok else f" — {check.detail}"))
    if failed:
        print(f"\nValidation failed ({len(failed)} check(s)); "
              "not finalising results.", file=sys.stderr)
        return 1
    machine.record("VALIDATION_COMPLETE")
    machine.record("RESULTS_FINALISED")
    print("\nPipeline complete: RESULTS_FINALISED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
