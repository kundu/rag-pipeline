"""Stage 1: draft answer generation. One LLM call per query, grounded only in
that query's retrieved chunks. Labels and citations are re-validated in code
after each call — the model is never trusted to enforce the policy alone."""
from __future__ import annotations

import json
from pathlib import Path

from .llm import LLMClient, LLMError

DRAFT_KEYS = ["query_id", "answer", "label", "citations", "reasoning_summary"]


def _draft_schema(allowed_labels: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "query_id": {"type": "string"},
            "answer": {"type": "string"},
            "label": {"type": "string", "enum": allowed_labels},
            "citations": {"type": "array", "items": {"type": "string"}},
            "reasoning_summary": {"type": "string"},
        },
        "required": DRAFT_KEYS,
        "additionalProperties": False,
    }


def format_context(chunk_ids: list[str], chunk_by_id: dict) -> str:
    parts = []
    for cid in chunk_ids:
        chunk = chunk_by_id[cid]
        parts.append(
            f"[chunk_id={cid}] (document: {chunk['document_name']})\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def build_draft_prompt(
    query: dict, retrieved_ids: list[str], chunk_by_id: dict, policy: dict
) -> str:
    ap = policy["answer_policy"]
    return f"""You are the answer-generation stage of a retrieval-augmented QA pipeline.

QUESTION (query_id={query['query_id']}):
{query['question']}

RETRIEVED CONTEXT — this is the ONLY permitted evidence. Do not use outside knowledge as if it came from these documents.

{format_context(retrieved_ids, chunk_by_id)}

ANSWER POLICY (JSON):
{json.dumps(ap, indent=2)}

INSTRUCTIONS:
- Answer the question using ONLY the retrieved context above.
- "label" must be exactly one of: {ap['allowed_labels']}.
  - "supported": the context directly supports your answer.
  - "insufficient_support": the context is only weakly or partially relevant — your answer text must explicitly say the evidence is weak or incomplete.
  - "not_in_corpus": the context contains no relevant evidence — your answer text must explicitly say the corpus does not answer this.
- "citations": chunk_id values (from the context above only) that support the answer, at most {ap['max_citations_per_answer']}. Cite the specific chunks you relied on.
- Never fabricate product capabilities, never claim compliance not stated in the context, and treat negative statements in the context (e.g. "X is not offered") as valid supporting evidence for a negative answer.

Return ONLY a single JSON object with exactly these keys:
{{"query_id": "...", "answer": "...", "label": "...", "citations": ["chunk_id", "..."], "reasoning_summary": "..."}}"""


def validate_draft(
    draft: dict, query_id: str, retrieved_ids: list[str], policy: dict
) -> dict:
    ap = policy["answer_policy"]
    if draft["label"] not in ap["allowed_labels"]:
        raise LLMError(
            f"{query_id}: label {draft['label']!r} not in allowed_labels"
        )
    citations = [c for c in draft.get("citations", []) if c in retrieved_ids]
    citations = citations[: ap["max_citations_per_answer"]]
    return {
        "query_id": query_id,
        "answer": str(draft["answer"]),
        "label": draft["label"],
        "citations": citations,
        "reasoning_summary": str(draft["reasoning_summary"]),
    }


def generate_drafts(
    client: LLMClient,
    queries: list[dict],
    retrieval_results: list[dict],
    chunk_by_id: dict,
    policy: dict,
    out_path: Path,
) -> list[dict]:
    retrieved_by_qid = {
        r["query_id"]: [c["chunk_id"] for c in r["retrieved_chunks"]]
        for r in retrieval_results
    }
    drafts = []
    for query in queries:
        qid = query["query_id"]
        retrieved_ids = retrieved_by_qid[qid]
        prompt = build_draft_prompt(query, retrieved_ids, chunk_by_id, policy)
        raw = client.call_json(
            stage="draft_answer",
            query_id=qid,
            prompt=prompt,
            required_keys=DRAFT_KEYS,
            schema=_draft_schema(policy["answer_policy"]["allowed_labels"]),
            input_artifacts=[
                "queries.json",
                "policy.json",
                "chunks.json",
                "retrieval_results.json",
            ],
            output_artifact="draft_answers.json",
        )
        drafts.append(validate_draft(raw, qid, retrieved_ids, policy))
    out_path.write_text(
        json.dumps(drafts, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return drafts
