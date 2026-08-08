# Replayable Mini RAG Pipeline

A staged, auditable retrieval-augmented QA pipeline: deterministic chunking and
retrieval, grounded draft answers with citations, an interactive human review
checkpoint that can override retrieved context, a second-stage per-query answer
audit, and a final evaluation report — with every LLM call logged and a
validation command that verifies the whole chain from disk artifacts.

## Requirements

- Python 3.10+ (stdlib only — no pip installs needed for the default setup)
- One LLM provider (auto-detected in this order):
  1. **Anthropic API** — `ANTHROPIC_API_KEY` set and `pip install anthropic` (model `claude-opus-5`)
  2. **Claude CLI** — `claude` installed and authenticated (headless `claude -p`)
  3. **OpenAI API** — `OPENAI_API_KEY` set (stdlib HTTP; model via `OPENAI_MODEL`, default `gpt-4o-mini`)

Force a provider with `RAG_LLM_PROVIDER=anthropic|claude_cli|openai`.
Pin the claude CLI model with `RAG_CLAUDE_CLI_MODEL=<model>` (e.g. `sonnet`).

## Run

```bash
make run              # interactive human review checkpoint
make run-auto         # non-interactive (records zero overrides)
make run-embedding    # embedding retrieval mode, non-interactive
make validate         # standalone validation of the last run
make clean            # delete generated artifacts (inputs are kept)
```

Or directly: `python3 run_pipeline.py [--mode keyword|embedding] [--auto-approve]`

### Human review checkpoint

After draft answers are generated, the pipeline prints each query's retrieval
results and draft label, then pauses:

```
Do you want to override retrieved chunks for any query before audit?
Enter query_id and comma-separated chunk_ids to force as final context, or press Enter to continue.
```

Enter one override per line, e.g. `Q4 billing::c0000,security::c0000`
(chunk ids must exist in `chunks.json`; invalid input is rejected and
re-prompted). Press Enter (or EOF) to continue. Overridden context becomes the
final context for the audit stage and the final report.

Scripted example: `printf 'Q4 billing::c0000\n\n' | python3 run_pipeline.py`

## Inputs

- `documents/` — corpus of `.txt` files
- `queries.json` — query set; optional `expected_evidence` annotations
  (`document_name` + `phrase`) enable deterministic retrieval metrics
- `policy.json` — retrieval config (`mode`, `top_k`, chunk sizing) and answer
  policy (allowed labels, citation rules, forbidden behaviours)

Swap any of these with equivalent fixtures; nothing depends on exact filenames
or wording.

## Generated artifacts

| Artifact | Contents |
|---|---|
| `chunks.json` | deterministic chunks (id, document, char offsets, text) |
| `index_metadata.json` | retrieval mode, stats, params, chunks sha256 |
| `retrieval_results.json` | per-query top-k chunks with ranks and scores |
| `draft_answers.json` | Stage 1 answers (label, citations, reasoning summary) |
| `review_overrides.json` | reviewer overrides + resolved final context per query |
| `answer_audit.json` | Stage 2 per-query audit incl. final context actually used |
| `revised_answers.json` | conservative regenerations for failed/high-risk answers |
| `retrieval_metrics.json` | hit@k / recall@k (or skipped when unannotated) |
| `retrieval_error_analysis.json` | evidence-based failure classification |
| `final_report.md` | six-section evaluation report, grounded vs weak marked |
| `llm_calls.jsonl` | one record per LLM call (stage, query, provider, prompt hash) |
| `pipeline_state.json` | stage transitions with timestamps + input hashes |

## Pipeline stages

```
INIT -> INPUTS_LOADED -> DOCUMENTS_CHUNKED -> INDEX_BUILT
     -> RETRIEVAL_COMPLETE -> DRAFT_ANSWERS_GENERATED
     -> HUMAN_REVIEW_COMPLETE -> ANSWERS_AUDITED
     -> FINAL_REPORT_GENERATED -> VALIDATION_COMPLETE -> RESULTS_FINALISED
```

Ordering is enforced in code (`rag_pipeline/state.py`); the final report cannot
be generated before chunking, retrieval, generation, review, and audit have
completed. Validation runs in-pipeline before results are finalised, and can be
re-run standalone at any time with `make validate` (12 checks: artifact
presence/schemas, disk-input hashes, chunking-before-LLM ordering, label and
citation policy compliance, per-query un-batched audit calls, audit-after-review
ordering, override propagation into audit inputs and the report, mode
consistency, report sections, and exact stage sequencing).
