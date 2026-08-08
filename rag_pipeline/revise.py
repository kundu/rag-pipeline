"""Regenerated answers after audit failure: any query whose audit came back
audit_label=fail or hallucination_risk=high gets one revision call using the
audited final context, with instructions to be more conservative."""
from __future__ import annotations

import json
from pathlib import Path

from .generate import DRAFT_KEYS, _draft_schema, format_context, validate_draft
from .llm import LLMClient


def needs_revision(audit: dict) -> bool:
    return audit["audit_label"] == "fail" or audit["hallucination_risk"] == "high"


def build_revision_prompt(
    query: dict, draft: dict, audit: dict, final_ids: list[str],
    chunk_by_id: dict, policy: dict,
) -> str:
    ap = policy["answer_policy"]
    return f"""You are the revision stage of a retrieval-augmented QA pipeline. The draft answer below FAILED audit (or was flagged high hallucination risk). Write a corrected, MORE CONSERVATIVE answer.

QUESTION (query_id={query['query_id']}):
{query['question']}

FAILED DRAFT (JSON):
{json.dumps(draft, indent=2, ensure_ascii=False)}

AUDIT FINDINGS (JSON):
{json.dumps(audit, indent=2, ensure_ascii=False)}

FINAL CONTEXT (the ONLY permitted evidence):

{format_context(final_ids, chunk_by_id)}

ANSWER POLICY (JSON):
{json.dumps(ap, indent=2)}

REVISION INSTRUCTIONS:
- Fix the problems the audit identified. Be strictly conservative: when in doubt, prefer "insufficient_support" or "not_in_corpus" over "supported".
- Claim nothing the final context does not state. If evidence is weak or missing, the answer text must explicitly say so.
- "label" must be one of: {ap['allowed_labels']}.
- "citations" may only reference chunk_ids from the final context above, at most {ap['max_citations_per_answer']}.

Return ONLY a single JSON object with exactly these keys:
{{"query_id": "...", "answer": "...", "label": "...", "citations": ["chunk_id"], "reasoning_summary": "..."}}"""


def revise_answers(
    client: LLMClient,
    queries: list[dict],
    drafts: list[dict],
    audits: list[dict],
    final_contexts: dict[str, list[str]],
    chunk_by_id: dict,
    policy: dict,
    out_path: Path,
) -> list[dict]:
    query_by_qid = {q["query_id"]: q for q in queries}
    draft_by_qid = {d["query_id"]: d for d in drafts}
    revised = []
    for audit in audits:
        if not needs_revision(audit):
            continue
        qid = audit["query_id"]
        final_ids = final_contexts[qid]
        prompt = build_revision_prompt(
            query_by_qid[qid], draft_by_qid[qid], audit, final_ids,
            chunk_by_id, policy,
        )
        raw = client.call_json(
            stage="revised_answer",
            query_id=qid,
            prompt=prompt,
            required_keys=DRAFT_KEYS,
            schema=_draft_schema(policy["answer_policy"]["allowed_labels"]),
            input_artifacts=[
                "queries.json",
                "policy.json",
                "chunks.json",
                "draft_answers.json",
                "answer_audit.json",
                "review_overrides.json",
            ],
            output_artifact="revised_answers.json",
        )
        entry = validate_draft(raw, qid, final_ids, policy)
        entry["revision_reason"] = (
            f"audit_label={audit['audit_label']}, "
            f"hallucination_risk={audit['hallucination_risk']}"
        )
        revised.append(entry)
    out_path.write_text(
        json.dumps(revised, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return revised
