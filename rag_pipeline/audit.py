"""Stage 2: answer audit. One LLM call per query — never batched — using the
FINAL context after any human overrides. Each saved record stores the
final_context_chunk_ids actually sent, so validation can prove overrides
flowed into the audit inputs."""
from __future__ import annotations

import json
from pathlib import Path

from .generate import format_context
from .llm import LLMClient, LLMError

AUDIT_KEYS = [
    "query_id",
    "audit_label",
    "support_assessment",
    "citation_check",
    "hallucination_risk",
    "recommended_fix",
]

AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "query_id": {"type": "string"},
        "audit_label": {"type": "string", "enum": ["pass", "fail"]},
        "support_assessment": {"type": "string"},
        "citation_check": {"type": "string"},
        "hallucination_risk": {"type": "string", "enum": ["low", "medium", "high"]},
        "recommended_fix": {"type": "string"},
    },
    "required": AUDIT_KEYS,
    "additionalProperties": False,
}


def build_audit_prompt(
    query: dict, draft: dict, final_ids: list[str], chunk_by_id: dict, policy: dict
) -> str:
    ap = policy["answer_policy"]
    return f"""You are the independent audit stage of a retrieval-augmented QA pipeline. Audit the draft answer below strictly against the FINAL CONTEXT (which may have been changed by a human reviewer after retrieval).

ORIGINAL QUESTION (query_id={query['query_id']}):
{query['question']}

DRAFT ANSWER (JSON):
{json.dumps(draft, indent=2, ensure_ascii=False)}

CITED CHUNK IDS: {draft['citations']}

FINAL CONTEXT (post-review; the only evidence that counts):

{format_context(final_ids, chunk_by_id)}

ANSWER POLICY (JSON):
{json.dumps(ap, indent=2)}

FORBIDDEN BEHAVIOURS: {ap['forbidden_behaviours']}

AUDIT INSTRUCTIONS:
- support_assessment: is the answer actually supported by the FINAL context? Quote or reference the specific evidence, or state what is missing.
- citation_check: are the cited chunk_ids appropriate — do they exist in the final context and contain the claimed evidence? Note citations pointing at chunks absent from the final context.
- audit_label: "pass" only if the answer is supported by the final context, correctly labeled, properly cited, and does not overclaim beyond the corpus; otherwise "fail".
- hallucination_risk: "low" | "medium" | "high" — risk that the answer asserts something the final context does not state (including any forbidden behaviour).
- recommended_fix: one concrete fix, or "none" if the answer is sound.

Return ONLY a single JSON object with exactly these keys:
{{"query_id": "...", "audit_label": "pass|fail", "support_assessment": "...", "citation_check": "...", "hallucination_risk": "low|medium|high", "recommended_fix": "..."}}"""


def audit_answers(
    client: LLMClient,
    queries: list[dict],
    drafts: list[dict],
    final_contexts: dict[str, list[str]],
    chunk_by_id: dict,
    policy: dict,
    out_path: Path,
) -> list[dict]:
    draft_by_qid = {d["query_id"]: d for d in drafts}
    audits = []
    for query in queries:
        qid = query["query_id"]
        draft = draft_by_qid[qid]
        final_ids = final_contexts[qid]
        prompt = build_audit_prompt(query, draft, final_ids, chunk_by_id, policy)
        raw = client.call_json(
            stage="audit",
            query_id=qid,
            prompt=prompt,
            required_keys=AUDIT_KEYS,
            schema=AUDIT_SCHEMA,
            input_artifacts=[
                "queries.json",
                "policy.json",
                "chunks.json",
                "draft_answers.json",
                "review_overrides.json",
            ],
            output_artifact="answer_audit.json",
        )
        if raw["audit_label"] not in ("pass", "fail"):
            raise LLMError(f"{qid}: invalid audit_label {raw['audit_label']!r}")
        if raw["hallucination_risk"] not in ("low", "medium", "high"):
            raise LLMError(
                f"{qid}: invalid hallucination_risk {raw['hallucination_risk']!r}"
            )
        audits.append(
            {
                "query_id": qid,
                "audit_label": raw["audit_label"],
                "support_assessment": str(raw["support_assessment"]),
                "citation_check": str(raw["citation_check"]),
                "hallucination_risk": raw["hallucination_risk"],
                "recommended_fix": str(raw["recommended_fix"]),
                "final_context_chunk_ids": final_ids,
            }
        )
    out_path.write_text(
        json.dumps(audits, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return audits
