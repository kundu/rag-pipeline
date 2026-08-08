# Replayable Mini RAG Pipeline — Implementation Plan

## Context

Build a replayable, auditable RAG pipeline in `/home/sauvik/Documents/di` (currently empty). Business outcome: an evaluator must be able to clone it fresh, swap the corpus/queries with equivalent fixtures, run the staged pipeline end-to-end, and verify that (a) chunking/indexing are deterministic code, (b) retrieval is observable and reproducible, (c) answer generation is grounded only in retrieved context, (d) audit is a separate second LLM stage, (e) a human review checkpoint can override retrieved context and that override genuinely changes downstream audit inputs and the final report, and (f) unsupported/weakly-supported answers are clearly visible. Static precomputed outputs fail; the pipeline must actually run and regenerate every artifact.

**User decisions (confirmed):**
- **LLM provider:** auto-detect multi-provider, in order: `ANTHROPIC_API_KEY` + anthropic SDK → Anthropic (`claude-opus-5`); `claude` CLI on PATH → headless `claude -p` (works on this machine today via subscription auth — verified); `OPENAI_API_KEY` → OpenAI. Override with env `RAG_LLM_PROVIDER=anthropic|claude_cli|openai`. Selected provider/model logged per call in `llm_calls.jsonl`.
- **Scope:** everything — MUST (1–6) + SHOULD (7–8) + STRETCH (9–10).

**Environment facts (probed):** Python 3.12 stdlib available; `anthropic`/`openai` SDKs NOT installed; `claude` CLI 2.1.224 installed and authenticated (`claude -p` verified working); `OPENAI_API_KEY` set; no `ANTHROPIC_API_KEY`, no `ant` profile. → Pipeline core must be **pure stdlib** (no numpy/sklearn/rank_bm25 deps) so a clean checkout runs with zero pip installs when using the claude CLI or OpenAI-via-urllib paths. `requirements.txt` lists `anthropic` as optional (only needed for the Anthropic path, per claude-api skill: SDK, never raw HTTP, model `claude-opus-5`).

## Repository layout (all new files)

```
/home/sauvik/Documents/di/
├── documents/                  # 3 sample .txt fixtures verbatim from spec
├── queries.json                # 4 sample queries + optional expected_evidence annotations (for metrics)
├── policy.json                 # verbatim from spec + "retrieval.mode": "keyword" (config for stretch #10)
├── rag_pipeline/
│   ├── __init__.py
│   ├── state.py                # Stage machine: ordered stage enum, pipeline_state.json writer/enforcer
│   ├── chunking.py             # deterministic char-window chunker → chunks.json
│   ├── indexing.py             # index build (both modes) → index_metadata.json
│   ├── retrieval.py            # BM25 keyword + hashed-TFIDF-cosine embedding → retrieval_results.json
│   ├── llm.py                  # provider abstraction + llm_calls.jsonl logger
│   ├── generate.py             # Stage 1: one draft-answer call per query → draft_answers.json
│   ├── review.py               # interactive human override checkpoint → review_overrides.json
│   ├── audit.py                # Stage 2: one audit call per query → answer_audit.json
│   ├── revise.py               # revised answers for fail/high-risk → revised_answers.json
│   ├── metrics.py              # recall@k / hit@k from annotations → retrieval_metrics.json
│   ├── error_analysis.py       # evidence-based failure classification → retrieval_error_analysis.json
│   └── report.py               # final_report.md generator
├── run_pipeline.py             # CLI entry: --mode keyword|embedding, --auto-approve
├── validate.py                 # standalone validation command (exit 0/1, itemized checks)
├── Makefile                    # make run / make validate / make clean
├── requirements.txt            # optional anthropic (commented per-provider notes)
└── README.md                   # run instructions, provider config, override syntax
```

## Design

### Stage machine (`state.py`)
Ordered stages exactly as spec: `INIT → INPUTS_LOADED → DOCUMENTS_CHUNKED → INDEX_BUILT → RETRIEVAL_COMPLETE → DRAFT_ANSWERS_GENERATED → HUMAN_REVIEW_COMPLETE → ANSWERS_AUDITED → FINAL_REPORT_GENERATED → VALIDATION_COMPLETE → RESULTS_FINALISED`. Each transition appends `{stage, timestamp}` to `pipeline_state.json`; advancing asserts the previous stage completed in-process (code-enforced ordering, not just convention). `INPUTS_LOADED` also records input file paths + sha256 hashes (proves disk reads). Report stage asserts chunking, retrieval, generation, review, and audit stages are all complete before writing.

### 1. Chunking (`chunking.py`) — no LLM involved
Read `documents/*.txt` sorted by filename. Sliding char window using `policy.retrieval.chunk_size_chars` / `chunk_overlap_chars`. `chunk_id = "<doc-stem>::c<i:04d>"`. Records: chunk_id, document_name, start_char, end_char, text. Pure function of (file bytes, policy) → byte-identical `chunks.json` on rerun (sorted keys, no timestamps inside).

### 2. Index + retrieval (`indexing.py`, `retrieval.py`) — stretch #10 included
Two modes, selectable via `policy.retrieval.mode` or `--mode` flag (flag wins):
- **keyword**: hand-rolled BM25 (k1=1.5, b=0.75), regex `\w+` lowercase tokenizer. Pure stdlib.
- **embedding**: deterministic hashed TF-IDF — md5-hash tokens into 512 fixed buckets, tf-idf weights, cosine similarity. Pure stdlib, fully reproducible (no model download, no numpy).

`index_metadata.json`: mode, chunk/doc counts, vocab size, BM25/hash params, sha256 of chunks.json, built_at. Retrieval: top_k per policy; every query gets ≥1 chunk (rank by score, keep top_k even if scores are 0). `retrieval_results.json` per spec schema with rank + retrieval_score.

### LLM layer (`llm.py`)
- `detect_provider()` in the confirmed order; env override `RAG_LLM_PROVIDER`.
- **anthropic**: `anthropic` SDK, `client.messages.create(model="claude-opus-5", max_tokens=16000, output_config={"format": {"type": "json_schema", "schema": ...}})` — structured outputs guarantee parseable JSON (per claude-api skill; no raw HTTP, no temperature, thinking left at default).
- **claude_cli**: `subprocess.run(["claude", "-p", prompt, "--output-format", "json"])`, prompt demands JSON-only output; extract `result` field, strip code fences, `json.loads`.
- **openai**: stdlib `urllib.request` POST to chat completions with `response_format: {"type": "json_object"}`; model from `OPENAI_MODEL` env, default `gpt-4o-mini`. (openai SDK not installed; urllib keeps zero-install.)
- Every call: validate parsed JSON against the expected keys; one retry with a "return only valid JSON matching the schema" repair prompt on failure; then append record to `llm_calls.jsonl`: `{stage, query_id, timestamp (ISO-8601), provider, model, prompt_hash (sha256 of full prompt), input_artifacts, output_artifact}`.

### 3. Stage 1 draft answers (`generate.py`)
One call per query. Prompt includes: question, the retrieved chunks (id + text — only these, nothing else, grounding constraint stated), the full answer_policy JSON, allowed labels, citation requirement + max_citations. Output schema: `{query_id, answer, label, citations, reasoning_summary}`. Post-validate in code: label ∈ allowed_labels (else fail loudly), citations filtered to that query's retrieved chunk_ids, capped at max_citations. → `draft_answers.json`.

### 4. Human review checkpoint (`review.py`)
Prints a per-query table: query_id, question, retrieved chunk_ids with ranks/scores, draft label. Then loops on the exact spec prompt:
```
Do you want to override retrieved chunks for any query before audit?
Enter query_id and comma-separated chunk_ids to force as final context, or press Enter to continue.
```
Input format `Q2 billing::c0001,security::c0000`. Validates query_id exists and every chunk_id exists in `chunks.json` (re-prompts on error). Multiple overrides allowed; empty line / EOF ends review. `--auto-approve` flag skips interaction (records empty overrides) for CI; piped stdin also works naturally. → `review_overrides.json`: `{reviewed_at, overrides: [{query_id, chunk_ids}], final_contexts: {query_id: [chunk_ids]}}` where final_contexts = override if present else original top-k — this is the single source of truth for downstream context.

### 5. Stage 2 audit (`audit.py`)
One call per query, never batched. Prompt: question, draft answer + label + citations, **final context after overrides** (chunk ids + texts from `review_overrides.json.final_contexts`), answer policy + forbidden_behaviours. Output: `{query_id, audit_label (pass|fail), support_assessment, citation_check, hallucination_risk (low|medium|high), recommended_fix}`. Each saved record also stores `final_context_chunk_ids` actually sent — the machine-checkable proof that overrides flowed into audit. `input_artifacts` for these calls includes `review_overrides.json`. → `answer_audit.json`.

### 8. Revised answers (`revise.py`)
For any query with `audit_label == "fail"` or `hallucination_risk == "high"`: one more LLM call (stage `revised_answer`) using audited final context, instructed to be more conservative, keep citation discipline, prefer `insufficient_support`/`not_in_corpus` over overclaiming. Same code-side label/citation validation. → `revised_answers.json` (written even if empty: `{revised: []}`).

### 7. Retrieval metrics (`metrics.py`) — deterministic, no LLM
Sample `queries.json` entries carry optional `expected_evidence: [{document_name, phrase}]`. If ANY query has annotations: compute per-query hit@k (any retrieved chunk from expected doc containing phrase) and recall@k (fraction of expected evidence items found), plus aggregates. If absent: gracefully write `{"skipped": true, "reason": "no expected_evidence annotations"}`. → `retrieval_metrics.json`.

### 9. Retrieval error analysis (`error_analysis.py`) — evidence-based, no LLM
For queries that failed (metrics miss, audit fail, or label ≠ supported), classify from observable signals:
- `chunking`: expected phrase exists in the source doc but is split across chunk boundaries (check raw doc text vs chunk texts).
- `ranking`: expected doc's chunks exist and contain the phrase but ranked below top_k.
- `corpus_gap`: draft/audit says not_in_corpus / insufficient and no doc contains matching evidence.
- `ambiguity`: low max retrieval score + evidence exists (weak lexical overlap).
Description field cites the concrete evidence (scores, ranks, phrase locations). → `retrieval_error_analysis.json`.

### 6. Final report (`report.py`)
`final_report.md` with the six required sections: Retrieval Summary (mode, top_k, per-query score stats), Query-by-Query Results (question, **final** context chunk ids, draft label, audit label, final recommendation from recommended_fix/revised answer), Reviewed Overrides (each override: original top-k vs forced context), Audit Findings, Failure Modes Observed (from error analysis), Recommended Improvements. Grounded answers marked distinctly (e.g. `GROUNDED` vs `⚠ WEAK/UNSUPPORTED`) — driven by (draft label, audit_label, hallucination_risk). Generated only after audit stage completes (stage machine enforces).

### Final stages: VALIDATION_COMPLETE → RESULTS_FINALISED
The spec's stage sequence ends `... FINAL_REPORT_GENERATED → VALIDATION_COMPLETE → RESULTS_FINALISED`, so validation is an in-pipeline stage, not only an external command. `validate.py` exposes `run_checks() -> list[CheckResult]`; `run_pipeline.py` imports and runs it after report generation, records `VALIDATION_COMPLETE` (only if all checks pass), then writes final status and records `RESULTS_FINALISED`. `python3 validate.py` / `make validate` runs the same checks standalone for the evaluator.

### Validation (`validate.py`, also `make validate`)
Itemized pass/fail lines, non-zero exit on any failure. Checks (mapping spec list):
1. All required artifacts exist; JSON/JSONL parse; records carry required schema fields (retrieved_chunks: chunk_id/document_name/rank/retrieval_score; drafts: query_id/answer/label/citations/reasoning_summary; audits: query_id/audit_label/support_assessment/citation_check/hallucination_risk/recommended_fix; llm_calls: stage/query_id/timestamp/provider/model/prompt_hash/input_artifacts/output_artifact).
2. Inputs read from disk: `pipeline_state.json` records document/queries/policy paths + hashes; every chunk's document_name exists in `documents/`.
3. Chunking before any LLM call: `DOCUMENTS_CHUNKED` timestamp < min timestamp in `llm_calls.jsonl`.
4. Every query in `queries.json` has ≥1 retrieved chunk (exception per spec: empty corpus → check skipped with a warning).
5. Every draft label ∈ `policy.answer_policy.allowed_labels`; citations ⊆ that query's retrieved chunk_ids; count ≤ max_citations.
6. `llm_calls.jsonl` has exactly one `draft_answer` record and one `audit` record per query (and `revised_answer` records matching revised_answers.json).
7. Audit after review: `HUMAN_REVIEW_COMPLETE` timestamp < min audit-call timestamp.
8. Overrides saved and applied: for each override, `answer_audit.json` record's `final_context_chunk_ids` equals the override; audit calls list `review_overrides.json` in `input_artifacts`.
9. Report reflects reviewed context: for each overridden query, its section in `final_report.md` contains the override chunk_ids (and they differ from original retrieval when they should).
10. `index_metadata.json` mode matches the mode recorded in `pipeline_state.json`; chunks.json sha256 matches index metadata (stretch #10 validation).
11. `final_report.md` contains all six required section headings (Retrieval Summary, Query-by-Query Results, Reviewed Overrides, Audit Findings, Failure Modes Observed, Recommended Improvements).
12. `pipeline_state.json` stage sequence matches the spec order exactly, ending `RESULTS_FINALISED`, with monotonically increasing timestamps.

### Makefile
`make run` (interactive), `make run-auto` (`--auto-approve`), `make run-embedding` (`--mode embedding`), `make validate`, `make clean` (delete generated artifacts, keep `documents/`, `queries.json`, `policy.json`).

## Implementation order
1. Fixtures (`documents/`, `queries.json` + annotations, `policy.json` + mode key)
2. `state.py`, `chunking.py`, `indexing.py`, `retrieval.py` (deterministic core, testable without LLM)
3. `llm.py` (all three providers; claude_cli tested live)
4. `generate.py`, `review.py`, `audit.py`, `revise.py`
5. `metrics.py`, `error_analysis.py`, `report.py`
6. `run_pipeline.py` orchestrator, `validate.py`, `Makefile`, `README.md`, `requirements.txt`

## Verification (end-to-end)
1. **Determinism:** run chunk+index twice, `diff chunks.json` runs — must be byte-identical; same for `retrieval_results.json`.
2. **Full run (no overrides):** `printf '\n' | python3 run_pipeline.py` — all artifacts produced; `python3 validate.py` exits 0.
3. **Override path:** `make clean`, then `printf 'Q4 billing::c0000\n\n' | python3 run_pipeline.py` — confirm `review_overrides.json` has the override, `answer_audit.json` Q4 `final_context_chunk_ids == ["billing::c0000"]`, audit output changes vs run 2, report Q4 section shows overridden context. `validate.py` exits 0.
4. **Embedding mode:** `python3 run_pipeline.py --mode embedding --auto-approve` — `index_metadata.json.mode == "embedding"`, validate passes.
5. **Graceful metric skip:** temporarily strip `expected_evidence` from queries.json, rerun retrieval+metrics — `retrieval_metrics.json` shows skipped, nothing crashes.
6. **Negative validation test:** hand-corrupt one draft label, confirm `validate.py` exits 1 with the specific check named, restore.
7. LLM calls run live via claude CLI (already verified reachable); inspect `llm_calls.jsonl` for per-stage records.
