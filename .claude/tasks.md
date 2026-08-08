# Task List — Replayable Mini RAG Pipeline

Derived from [plan.md](plan.md). Execute in order; each phase depends on the previous.

## Phase 0 — Fixtures & Scaffolding
- [x] T0.1 Create `documents/` with the 3 sample corpus files verbatim from spec (`product_overview.txt`, `billing.txt`, `security.txt`)
- [x] T0.2 Create `queries.json` with the 4 sample queries + `expected_evidence` annotations (`document_name` + `phrase`) per query for metrics
- [x] T0.3 Create `policy.json` verbatim from spec + add `"retrieval": {"mode": "keyword"}` key for configurable retrieval mode
- [x] T0.4 Create `rag_pipeline/__init__.py` package skeleton

## Phase 1 — Deterministic Core (no LLM)
- [x] T1.1 `rag_pipeline/state.py` — stage machine: ordered stage enum (INIT → … → RESULTS_FINALISED), `pipeline_state.json` writer, in-process enforcement that a stage cannot start before its predecessor completes; `INPUTS_LOADED` records input paths + sha256 hashes
- [x] T1.2 `rag_pipeline/chunking.py` — sliding char-window chunker from `policy.retrieval.chunk_size_chars` / `chunk_overlap_chars`; files sorted by name; `chunk_id = "<doc-stem>::c<i:04d>"`; writes `chunks.json` (sorted keys, no timestamps → byte-identical on rerun)
- [x] T1.3 `rag_pipeline/indexing.py` — build index for both modes; write `index_metadata.json` (mode, counts, params, sha256 of chunks.json, built_at)
- [x] T1.4 `rag_pipeline/retrieval.py` — keyword mode: hand-rolled BM25 (k1=1.5, b=0.75, regex `\w+` lowercase tokens); embedding mode: deterministic hashed TF-IDF (md5 → 512 buckets, cosine); top_k from policy; every query ≥1 chunk; writes `retrieval_results.json` per spec schema
- [x] T1.5 Determinism check: run chunk+index+retrieve twice, diff outputs byte-identical

## Phase 2 — LLM Layer
- [x] T2.1 `rag_pipeline/llm.py` — `detect_provider()` order: `ANTHROPIC_API_KEY`+SDK → `claude` CLI → `OPENAI_API_KEY`; env override `RAG_LLM_PROVIDER`
- [x] T2.2 Anthropic adapter: `anthropic` SDK, model `claude-opus-5`, structured outputs via `output_config.format` json_schema
- [x] T2.3 Claude CLI adapter: `subprocess.run(["claude", "-p", prompt, "--output-format", "json"])`, JSON extraction (strip fences), parse `result` field
- [x] T2.4 OpenAI adapter: stdlib `urllib.request` POST, `response_format: json_object`, model from `OPENAI_MODEL` (default `gpt-4o-mini`)
- [x] T2.5 JSON validation + one repair-retry on parse failure; `llm_calls.jsonl` logger: `{stage, query_id, timestamp, provider, model, prompt_hash, input_artifacts, output_artifact}`

## Phase 3 — Generation, Review, Audit, Revision
- [x] T3.1 `rag_pipeline/generate.py` — Stage 1: one call per query; prompt = question + retrieved chunks (only) + answer policy + allowed labels + citation rules; code-side validation (label ∈ allowed, citations ⊆ retrieved ids, ≤ max_citations); writes `draft_answers.json`
- [x] T3.2 `rag_pipeline/review.py` — print per-query table (retrieval + draft label); loop on exact spec prompt text; parse `QID id1,id2` overrides; validate chunk ids against `chunks.json`; `--auto-approve` for CI; EOF/empty = continue; writes `review_overrides.json` with `overrides` + `final_contexts` (single source of truth downstream)
- [x] T3.3 `rag_pipeline/audit.py` — Stage 2: one call per query, never batched; prompt = question + draft answer + citations + **final context after overrides** + policy + forbidden_behaviours; record stores `final_context_chunk_ids`; `input_artifacts` includes `review_overrides.json`; writes `answer_audit.json`
- [x] T3.4 `rag_pipeline/revise.py` — for `audit_label == "fail"` or `hallucination_risk == "high"`: one conservative revision call per affected query using final context; writes `revised_answers.json` (even if empty)

## Phase 4 — Metrics, Analysis, Report
- [x] T4.1 `rag_pipeline/metrics.py` — hit@k / recall@k from `expected_evidence`; graceful `{"skipped": true}` when annotations absent; writes `retrieval_metrics.json`
- [x] T4.2 `rag_pipeline/error_analysis.py` — evidence-based classification (`chunking` / `ranking` / `corpus_gap` / `ambiguity`) with concrete evidence in descriptions; writes `retrieval_error_analysis.json`
- [x] T4.3 `rag_pipeline/report.py` — `final_report.md` with 6 required sections; per-query rows (question, final context ids, draft label, audit label, final recommendation); GROUNDED vs ⚠ WEAK/UNSUPPORTED marking; stage machine asserts audit complete first

## Phase 5 — Orchestration, Validation, Docs
- [x] T5.1 `run_pipeline.py` — CLI orchestrator: `--mode keyword|embedding`, `--auto-approve`; runs all stages in order **including final stages**: after report, imports `validate.run_checks()` → records `VALIDATION_COMPLETE` (only if all pass) → records `RESULTS_FINALISED`
- [x] T5.2 `validate.py` — 12 itemized checks per plan §Validation (incl. schema-field checks, 6 report section headings, full stage sequence ending `RESULTS_FINALISED`); callable as `run_checks()` for T5.1 reuse; exit 0/1 standalone
- [x] T5.3 `Makefile` — `run`, `run-auto`, `run-embedding`, `validate`, `clean`
- [x] T5.4 `requirements.txt` (optional `anthropic`, per-provider comments) + `README.md` (run instructions, provider config, override syntax example)

## Phase 6 — End-to-End Verification
- [x] T6.1 Execute all acceptance test cases in [test-cases.md](test-cases.md) (TC-01 … TC-10)
- [x] T6.2 Fix any failures, re-run `make validate` until exit 0 on all paths
