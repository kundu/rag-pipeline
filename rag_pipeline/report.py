"""Final evaluation report. Generated only after chunking, retrieval,
generation, human review, and audit have completed (enforced by the stage
machine in the orchestrator). Uses the reviewed FINAL context, not the raw
retrieval, and clearly separates grounded answers from weak/unsupported ones."""
from __future__ import annotations

from pathlib import Path

from .state import utc_now

REQUIRED_SECTIONS = [
    "Retrieval Summary",
    "Query-by-Query Results",
    "Reviewed Overrides",
    "Audit Findings",
    "Failure Modes Observed",
    "Recommended Improvements",
]


def _grounding(draft: dict, audit: dict) -> str:
    if (
        draft["label"] == "supported"
        and audit["audit_label"] == "pass"
        and audit["hallucination_risk"] == "low"
    ):
        return "GROUNDED"
    if audit["audit_label"] == "fail" or audit["hallucination_risk"] == "high":
        return "UNSUPPORTED / FAILED AUDIT"
    return "WEAKLY SUPPORTED"


def write_report(
    *,
    mode: str,
    provider: str,
    model: str,
    queries: list[dict],
    retrieval_results: list[dict],
    review_record: dict,
    drafts: list[dict],
    audits: list[dict],
    revised: list[dict],
    metrics: dict,
    error_analysis: list[dict],
    top_k: int,
    out_path: Path,
) -> None:
    draft_by_qid = {d["query_id"]: d for d in drafts}
    audit_by_qid = {a["query_id"]: a for a in audits}
    revised_by_qid = {r["query_id"]: r for r in revised}
    retrieved_by_qid = {
        r["query_id"]: r["retrieved_chunks"] for r in retrieval_results
    }
    final_contexts = review_record["final_contexts"]

    lines: list[str] = []
    add = lines.append

    add("# Final Evaluation Report — Mini RAG Pipeline")
    add("")
    add(f"Generated: {utc_now()}  ")
    add(f"Retrieval mode: `{mode}` · top_k: {top_k} · LLM provider: `{provider}` (model: `{model}`)")
    add("")

    # -- Retrieval Summary ---------------------------------------------------
    add("## Retrieval Summary")
    add("")
    add(f"- Queries processed: {len(queries)}")
    add(f"- Retrieval mode: `{mode}`, top_k={top_k}")
    if metrics.get("skipped"):
        add(f"- Deterministic metrics: skipped ({metrics['reason']})")
    else:
        agg = metrics["aggregate"]
        add(
            f"- Deterministic metrics over {metrics['num_annotated_queries']} "
            f"annotated queries: hit@{top_k}={agg['hit_at_k']}, "
            f"recall@{top_k}={agg['recall_at_k']}"
        )
    add("")
    add("| query | top-ranked chunk | max score | min score |")
    add("|---|---|---|---|")
    for result in retrieval_results:
        scores = [c["retrieval_score"] for c in result["retrieved_chunks"]]
        top_chunk = (
            result["retrieved_chunks"][0]["chunk_id"]
            if result["retrieved_chunks"]
            else "—"
        )
        add(
            f"| {result['query_id']} | `{top_chunk}` | "
            f"{max(scores) if scores else '—'} | {min(scores) if scores else '—'} |"
        )
    add("")

    # -- Query-by-Query Results ----------------------------------------------
    add("## Query-by-Query Results")
    add("")
    for query in queries:
        qid = query["query_id"]
        draft = draft_by_qid[qid]
        audit = audit_by_qid[qid]
        revision = revised_by_qid.get(qid)
        final_ids = final_contexts[qid]
        original_ids = [c["chunk_id"] for c in retrieved_by_qid[qid]]
        overridden = final_ids != original_ids

        add(f"### {qid} — {_grounding(draft, audit)}")
        add("")
        add(f"**Question:** {query['question']}")
        add("")
        add(
            f"- **Final context chunk IDs**"
            + (" (human-overridden)" if overridden else "")
            + ": "
            + ", ".join(f"`{cid}`" for cid in final_ids)
        )
        if overridden:
            add(
                "- Original retrieval was: "
                + ", ".join(f"`{cid}`" for cid in original_ids)
            )
        add(f"- **Draft label:** `{draft['label']}`")
        add(
            f"- **Audit label:** `{audit['audit_label']}` "
            f"(hallucination risk: `{audit['hallucination_risk']}`)"
        )
        add(f"- **Answer:** {draft['answer']}")
        add(f"- **Citations:** {', '.join(f'`{c}`' for c in draft['citations']) or 'none'}")
        fix = audit["recommended_fix"]
        if revision:
            add(
                f"- **Final recommendation:** revised answer issued "
                f"(label `{revision['label']}`): {revision['answer']}"
            )
        elif fix and fix.lower() not in ("none", "none."):
            add(f"- **Final recommendation:** {fix}")
        else:
            add("- **Final recommendation:** answer stands as drafted.")
        add("")

    # -- Reviewed Overrides ----------------------------------------------------
    add("## Reviewed Overrides")
    add("")
    if review_record["overrides"]:
        for override in review_record["overrides"]:
            qid = override["query_id"]
            original_ids = [c["chunk_id"] for c in retrieved_by_qid[qid]]
            add(
                f"- **{qid}**: reviewer forced final context "
                + ", ".join(f"`{cid}`" for cid in override["chunk_ids"])
                + " (original retrieval: "
                + ", ".join(f"`{cid}`" for cid in original_ids)
                + "). Downstream audit used the overridden context."
            )
    else:
        add("- No overrides were made; every query was audited against its original top-k retrieval.")
    add("")

    # -- Audit Findings ---------------------------------------------------------
    add("## Audit Findings")
    add("")
    add("| query | audit | risk | support assessment | citation check |")
    add("|---|---|---|---|---|")
    for audit in audits:
        add(
            f"| {audit['query_id']} | {audit['audit_label']} | "
            f"{audit['hallucination_risk']} | "
            f"{audit['support_assessment'].replace('|', '/')} | "
            f"{audit['citation_check'].replace('|', '/')} |"
        )
    add("")
    n_pass = sum(1 for a in audits if a["audit_label"] == "pass")
    add(f"{n_pass}/{len(audits)} answers passed audit. "
        f"{len(revised)} answer(s) were regenerated after audit failure/high risk.")
    add("")

    # -- Failure Modes Observed ---------------------------------------------------
    add("## Failure Modes Observed")
    add("")
    if error_analysis:
        for entry in error_analysis:
            add(
                f"- **{entry['query_id']}** — `{entry['failure_type']}`: "
                f"{entry['description']}"
            )
    else:
        add("- No failure signals observed: all queries retrieved their expected evidence and passed audit.")
    add("")

    # -- Recommended Improvements ---------------------------------------------------
    add("## Recommended Improvements")
    add("")
    recommendations = []
    if any(e["failure_type"] == "chunking" for e in error_analysis):
        recommendations.append(
            "Increase chunk overlap or switch to sentence-aware boundaries — "
            "evidence phrases were split across chunk boundaries."
        )
    if any(e["failure_type"] == "ranking" for e in error_analysis):
        recommendations.append(
            "Evidence-bearing chunks exist but ranked below top-k: consider "
            "hybrid keyword+embedding scoring or a higher top_k."
        )
    if any(e["failure_type"] == "ambiguity" for e in error_analysis):
        recommendations.append(
            "Low lexical overlap between queries and evidence: add query "
            "expansion or embedding-based retrieval (`--mode embedding`)."
        )
    if any(e["failure_type"] == "corpus_gap" for e in error_analysis):
        recommendations.append(
            "Some questions are unanswerable from the corpus: expand the "
            "document set or surface `not_in_corpus` answers to content owners."
        )
    if any(a["audit_label"] == "fail" for a in audits):
        recommendations.append(
            "Audit failures occurred: tighten the draft prompt's grounding "
            "instructions and keep the two-stage audit gate in place."
        )
    if not recommendations:
        recommendations.append(
            "Pipeline is healthy on this corpus. Next steps: grow the query "
            "set with harder multi-hop questions and track hit@k over time."
        )
    for rec in recommendations:
        add(f"- {rec}")
    add("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
