# Acceptance Test Cases — Replayable Mini RAG Pipeline

Derived from [plan.md](plan.md) §Verification and the spec's VALIDATION REQUIREMENTS. Each case has explicit steps and pass criteria. All commands run from repo root.

---

## TC-01 — Deterministic chunking & indexing
**Covers:** spec "chunking and indexing are deterministic", MUST #1
**Steps:**
1. `python3 run_pipeline.py --auto-approve` (or run chunk/index stages only)
2. Save copies: `cp chunks.json /tmp/c1.json && cp retrieval_results.json /tmp/r1.json`
3. Re-run the same command
4. `diff chunks.json /tmp/c1.json && diff retrieval_results.json /tmp/r1.json`
**Pass:** both diffs empty (byte-identical). `index_metadata.json` sha256 of chunks.json identical across runs.

## TC-02 — Full pipeline run, no overrides
**Covers:** MUST #1–#6, all required artifacts
**Steps:**
1. `make clean`
2. `printf '\n' | python3 run_pipeline.py`
3. `python3 validate.py`
**Pass:** exit 0 on both. All artifacts exist and parse: `chunks.json`, `index_metadata.json`, `retrieval_results.json`, `draft_answers.json`, `review_overrides.json`, `answer_audit.json`, `final_report.md`, `retrieval_metrics.json`, `revised_answers.json`, `retrieval_error_analysis.json`, `llm_calls.jsonl`, `pipeline_state.json`. Every query has ≥1 retrieved chunk. Stage order in `pipeline_state.json` matches the spec sequence exactly and ends `RESULTS_FINALISED` (validation ran in-pipeline). `final_report.md` contains all six required section headings.

## TC-03 — Human override genuinely affects downstream audit
**Covers:** MUST #4, spec "human overrides genuinely affect downstream outputs"
**Steps:**
1. `make clean`
2. `printf 'Q4 billing::c0000\nQ2 security::c0000\n\n' | python3 run_pipeline.py` (two overrides — spec requires "one or more")
3. Inspect `review_overrides.json`, `answer_audit.json`, `final_report.md`
**Pass:**
- `review_overrides.json` contains both overrides; `final_contexts["Q4"] == ["billing::c0000"]` and `final_contexts["Q2"] == ["security::c0000"]`; checks below apply to each overridden query
- `answer_audit.json` Q4 record has `final_context_chunk_ids == ["billing::c0000"]` (≠ original retrieval)
- Q4 audit LLM call in `llm_calls.jsonl` lists `review_overrides.json` in `input_artifacts`
- `final_report.md` Q4 row shows `billing::c0000` as final context and lists the override in Reviewed Overrides section
- `python3 validate.py` exits 0

## TC-04 — Invalid override rejected
**Covers:** rule "override chunk IDs must exist in chunks.json"
**Steps:**
1. Run pipeline interactively, enter `Q1 nonexistent::c9999` at review prompt
**Pass:** input rejected with error message naming the invalid chunk id; re-prompt shown; entering a valid override or empty line then proceeds normally. Bad id never written to `review_overrides.json`.

## TC-05 — Draft answer policy compliance
**Covers:** MUST #3 requirements
**Steps:**
1. After TC-02 run, inspect `draft_answers.json` against `policy.json`
**Pass:** every `label` ∈ `allowed_labels`; every citation ∈ that query's retrieved chunk_ids; citation count ≤ `max_citations_per_answer`; every record has non-empty `answer` and `reasoning_summary`. Q3 (refunds) and Q4 (HIPAA) answers do not overclaim — labels/text reflect what corpus actually states (negative evidence handled: "not described as HIPAA compliant").

## TC-06 — Per-query audit calls, staged after review
**Covers:** MUST #5, "queries must not be batched in audit", "audit was run after human review"
**Steps:**
1. After TC-02 run, inspect `llm_calls.jsonl` and `pipeline_state.json`
**Pass:** exactly one `stage: "audit"` record per query_id (4 records for sample fixture); exactly one `stage: "draft_answer"` record per query_id; `HUMAN_REVIEW_COMPLETE` timestamp < min audit-call timestamp; `DOCUMENTS_CHUNKED` timestamp < min of ALL LLM-call timestamps; every audit record has valid `audit_label` (pass|fail), `hallucination_risk` (low|medium|high), `support_assessment`, `citation_check`, `recommended_fix`.

## TC-07 — Revised answers on audit failure
**Covers:** SHOULD #8
**Steps:**
1. After a full run, check `answer_audit.json` for any `audit_label == "fail"` or `hallucination_risk == "high"`
2. If none occur naturally, force one: temporarily edit a draft answer to a fabricated claim (e.g. "HIPAA compliant: yes"), re-run audit+revise stages
**Pass:** for each failing query there is a `revised_answers.json` entry and a `stage: "revised_answer"` record in `llm_calls.jsonl`; revised answer is more conservative (label downgraded or hedged text) and citations still ⊆ final context chunk ids. With zero failures, `revised_answers.json` exists with empty list.

## TC-08 — Retrieval metrics + graceful skip
**Covers:** SHOULD #7
**Steps:**
1. After TC-02 run with annotated `queries.json`: inspect `retrieval_metrics.json`
2. Then: `cp queries.json /tmp/q.json`, strip `expected_evidence` fields, re-run retrieval+metrics stages, inspect again; restore queries.json
**Pass:** step 1 — per-query hit@k and recall@k present, values in [0,1], aggregates computed, deterministic across reruns. Step 2 — file contains `{"skipped": true, ...}`, no crash, pipeline continues.

## TC-09 — Embedding retrieval mode
**Covers:** STRETCH #10
**Steps:**
1. `make clean && python3 run_pipeline.py --mode embedding --auto-approve`
2. `python3 validate.py`
**Pass:** `index_metadata.json` has `"mode": "embedding"`; mode recorded in `pipeline_state.json` matches; retrieval results reproducible (re-run → identical); validate exits 0. Same check with `--mode keyword` yields `"mode": "keyword"`.

## TC-10 — Validation catches corruption (negative test)
**Covers:** VALIDATION REQUIREMENTS robustness
**Steps:**
1. After a passing run: `cp draft_answers.json /tmp/d.json`
2. Edit one draft `label` to `"totally_supported"` (not in allowed_labels)
3. `python3 validate.py`; restore file
4. Repeat with: delete one audit record from `llm_calls.jsonl`; swap `final_context_chunk_ids` in an overridden audit record
**Pass:** each corruption makes `validate.py` exit 1 and print the specific failed check by name; after restore, exit 0 again.

---

## TC-11 — Error analysis grounded in evidence (STRETCH #9)
**Steps:**
1. After a run with at least one failure signal (metrics miss / audit fail / label ≠ supported), inspect `retrieval_error_analysis.json`
**Pass:** every entry has `failure_type` ∈ {ranking, chunking, ambiguity, corpus_gap} and a `description` citing concrete observable evidence (scores, ranks, phrase location, audit result) — not freeform opinion alone. Queries with no failure signal are absent.

## TC-12 — Clean-checkout replayability
**Covers:** EXECUTION REQUIREMENTS
**Steps:**
1. Copy only committed inputs + code to a fresh temp dir (no generated artifacts): `documents/`, `queries.json`, `policy.json`, `rag_pipeline/`, `run_pipeline.py`, `validate.py`, `Makefile`, `README.md`, `requirements.txt`
2. In temp dir: `printf '\n' | python3 run_pipeline.py && python3 validate.py`
3. Swap in an equivalent fixture (different .txt files + matching queries), re-run
**Pass:** both runs regenerate all artifacts from scratch and validate exits 0; nothing depends on exact filenames, wording, or precomputed outputs.
