# Replayable Mini RAG Pipeline — Project Instructions

Staged, auditable RAG pipeline: deterministic chunking/retrieval → grounded draft
answers → interactive human review (context overrides) → per-query audit →
report, with every LLM call logged and a 12-check validator. Full spec history:
[.claude/plan.md](.claude/plan.md) · [.claude/tasks.md](.claude/tasks.md) ·
[.claude/test-cases.md](.claude/test-cases.md).

For ANY development on this project, invoke the `rag-pipeline` skill
(`.claude/skills/rag-pipeline/SKILL.md`) — it is the owning engineer for this
codebase: change-to-seam routing, invariants it enforces, the mandatory
regression gate, and the doc-sync rules.

## Commands

```bash
make run              # interactive review checkpoint
make run-auto         # non-interactive (zero overrides)
make run-embedding    # embedding retrieval mode
make validate         # standalone 12-check validation (17 PASS lines)
make clean            # delete generated artifacts, keep inputs
# scripted override run:
printf 'Q4 billing::c0000\n\n' | python3 run_pipeline.py
```

## Hard constraints — do not break

1. **Pure stdlib core.** No pip dependencies in `rag_pipeline/` (only the
   optional `anthropic` SDK import inside the anthropic provider branch).
2. **Determinism.** `chunks.json` and `retrieval_results.json` must be
   byte-identical across runs for identical inputs: keep `sort_keys=True`, no
   timestamps in those files, retrieval ties broken by `chunk_id`, md5-based
   embedding buckets. Never introduce randomness, dict-order dependence, or
   wall-clock values into them.
3. **Stage order is code-enforced** (`rag_pipeline/state.py::STAGES`). New work
   must slot between existing stages, not bypass `StageMachine.record()`.
4. **No LLM before `DOCUMENTS_CHUNKED`**; audit calls only after
   `HUMAN_REVIEW_COMPLETE`; one call per query, never batched. Validation
   checks these from timestamps — they are not conventions.
5. **`review_overrides.json.final_contexts` is the single source of truth** for
   downstream context. Audit records must store the `final_context_chunk_ids`
   actually sent; validate check 08 compares every query, not just overridden ones.
6. **Never trust model output**: labels/citations are re-validated in code after
   every call (`generate.validate_draft`); JSON parse failures get exactly one
   repair retry, then the pipeline fails loudly.
7. **`claude` CLI subprocess keeps `stdin=subprocess.DEVNULL`** (llm.py). The
   pipeline's stdin carries the human-review override lines; removing DEVNULL
   makes the CLI eat them (real bug, found by TC-03).

## Testing gate

Any change to pipeline code requires: `make clean` → the affected test cases
from [.claude/test-cases.md](.claude/test-cases.md) → `make validate` (17/17).
Minimum regression set for core changes: TC-01 (determinism), TC-02 (full run),
TC-03 (override chain), TC-10 (negative validation). LLM-touching changes: use
`RAG_CLAUDE_CLI_MODEL=sonnet` to keep runs fast.

## Environment knobs

| Var | Effect |
| --- | --- |
| `RAG_LLM_PROVIDER` | force `anthropic` / `claude_cli` / `openai` (default: auto-detect in that order) |
| `RAG_CLAUDE_CLI_MODEL` | model for claude CLI calls (e.g. `sonnet`, `haiku`) |
| `RAG_ANTHROPIC_MODEL` | Anthropic SDK model (default `claude-opus-5`) |
| `OPENAI_MODEL` | OpenAI model (default `gpt-4o-mini`) |
