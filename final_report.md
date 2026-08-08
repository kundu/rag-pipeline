# Final Evaluation Report — Mini RAG Pipeline

Generated: 2026-08-08T16:34:27.530270+00:00  
Retrieval mode: `keyword` · top_k: 3 · LLM provider: `claude_cli` (model: `sonnet`)

## Retrieval Summary

- Queries processed: 4
- Retrieval mode: `keyword`, top_k=3
- Deterministic metrics over 4 annotated queries: hit@3=1.0, recall@3=1.0

| query | top-ranked chunk | max score | min score |
|---|---|---|---|
| Q1 | `product_overview::c0000` | 5.530857 | 0.588743 |
| Q2 | `product_overview::c0001` | 3.202551 | 0.147425 |
| Q3 | `billing::c0000` | 3.611295 | 1.270307 |
| Q4 | `security::c0000` | 3.884811 | 0.49393 |

## Query-by-Query Results

### Q1 — GROUNDED

**Question:** How long is event data retained on the standard plan?

- **Final context chunk IDs**: `product_overview::c0000`, `security::c0000`, `product_overview::c0001`
- **Draft label:** `supported`
- **Audit label:** `pass` (hallucination risk: `low`)
- **Answer:** Standard plan retains event data 13 months.
- **Citations:** `product_overview::c0000`
- **Final recommendation:** answer stands as drafted.

### Q2 — GROUNDED

**Question:** Does the product support SCIM provisioning?

- **Final context chunk IDs**: `product_overview::c0001`, `product_overview::c0000`, `billing::c0000`
- **Draft label:** `supported`
- **Audit label:** `pass` (hallucination risk: `low`)
- **Answer:** Yes. Product supports SCIM provisioning, available on enterprise plans, alongside SSO via SAML.
- **Citations:** `product_overview::c0001`
- **Final recommendation:** answer stands as drafted.

### Q3 — GROUNDED

**Question:** Can customers get refunds for unused days in a month?

- **Final context chunk IDs**: `billing::c0000`, `security::c0000`, `product_overview::c0000`
- **Draft label:** `supported`
- **Audit label:** `pass` (hallucination risk: `low`)
- **Answer:** No. Refunds for partial/unused months are not offered, except where required by law.
- **Citations:** `billing::c0000`
- **Final recommendation:** answer stands as drafted.

### Q4 — GROUNDED

**Question:** Is the service HIPAA compliant?

- **Final context chunk IDs**: `security::c0000`, `product_overview::c0000`, `billing::c0000`
- **Draft label:** `supported`
- **Audit label:** `pass` (hallucination risk: `low`)
- **Answer:** No. Service not described as HIPAA compliant in current public documentation.
- **Citations:** `security::c0000`
- **Final recommendation:** answer stands as drafted.

## Reviewed Overrides

- No overrides were made; every query was audited against its original top-k retrieval.

## Audit Findings

| query | audit | risk | support assessment | citation check |
|---|---|---|---|---|
| Q1 | pass | low | Chunk product_overview::c0000 say straight up: 'retains event data for 13 months on standard plan'. Match draft exact. | Cite id exist in final context, hold claimed fact. One cite, under max 3. Good. |
| Q2 | pass | low | c0001 say: 'Authentication supports email-password, SSO via SAML, and SCIM provisioning on enterprise plans.' Match answer claim exact — SCIM yes, enterprise plan gate, SSO SAML mention. No overclaim. | c0001 exist in final context, contain claimed evidence. Single citation enough, within max 3. |
| Q3 | pass | low | billing::c0000 states verbatim: 'Refunds are not offered for partial months, except where required by law.' Draft answer matches this exactly, no overclaim. | Citation billing::c0000 exists in final context and contains the exact claimed evidence. Single citation, within max of 3. |
| Q4 | pass | low | Chunk security::c0000 say direct: 'service not described as HIPAA compliant in current public documentation.' Draft answer match this exact, no overclaim. | Citation security::c0000 exist in final context, contain exact claimed evidence. Only 1 citation, under max 3. Good. |

4/4 answers passed audit. 0 answer(s) were regenerated after audit failure/high risk.

## Failure Modes Observed

- No failure signals observed: all queries retrieved their expected evidence and passed audit.

## Recommended Improvements

- Pipeline is healthy on this corpus. Next steps: grow the query set with harder multi-hop questions and track hit@k over time.
