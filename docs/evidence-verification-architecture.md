# Evidence Verification Architecture

This is a future architecture only. No retrieval service, web access, or
verification endpoint is implemented by this change.

```text
User input
  → input type detection
  → claim extraction
  → claim decomposition
  → search-query generation
  → sparse retrieval
  → dense retrieval
  → evidence deduplication
  → evidence reranking
  → source/date validation
  → claim-evidence entailment
  → evidence aggregation
  → uncertain verdict
  → citation-backed explanation
```

## Pipeline stages

1. **Input detection** identifies text, URL, image, screenshot, video, or
   mixed input. It records language and parsing failures without silently
   discarding content.
2. **Claim extraction** identifies factual, checkable statements and keeps the
   original message. It may return multiple claims or no atomic claim.
3. **Claim decomposition** separates subjects, predicates, quantities, places,
   dates, and conditions so evidence can address each part.
4. **Query generation** creates reproducible search queries from the claim. It
   must not invent entities, dates, or URLs.
5. **Sparse retrieval** uses lexical methods such as BM25 or TF-IDF to provide
   transparent initial candidates.
6. **Dense retrieval** optionally adds multilingual embedding retrieval. It is
   a recall mechanism, not a truth decision.
7. **Deduplication** groups repeated pages, syndicated copies, redirects, and
   near-identical passages before scoring sources.
8. **Reranking** scores passage-level relevance and retains the query, model
   version, and candidate set.
9. **Source/date validation** records source category, publisher, publication
   time, retrieval time, event time, redirects, and availability. A source
   category is a quality factor, not a permanent truth score.
10. **Entailment** determines whether a passage supports, contradicts, or is
    unrelated to the claim. It must preserve `neutral` and uncertainty.
11. **Aggregation** combines independent evidence, detects disagreement, and
    applies temporal rules. It must abstain when evidence is absent, weak,
    stale, or contradictory.
12. **Explanation** cites only retrieved evidence and includes excerpts,
    timestamps, scores, and limitations.

## Evidence record

Every stored evidence item should preserve:

- claim text and claim identifier;
- evidence text/excerpt;
- source title, URL, publisher, and source category;
- publication date, event date when known, and retrieval timestamp;
- retrieval method and query;
- relevance and entailment scores;
- support/contradiction/neutral relation;
- deduplication group and reranker version;
- human-readable justification generated from the recorded fields.

The future typed contract is in `backend/app/verification_contract.py`. It is
not mounted into the current FastAPI routes.

## Source taxonomy

Use the categories `official_government`, `official_institutional`,
`primary_research`, `court_or_legal`, `reputable_news`,
`professional_fact_check`, `expert_technical`, `user_generated`, and
`unknown_low_context`. Display the category and quality factors. Do not encode
an arbitrary permanent domain truth score or use a hardcoded good/bad-domain
list.

## Modality branches

- **Image-only:** assess manipulation or reuse signals only.
- **OCR:** retain OCR text and OCR confidence, then send the extracted text
  through claim extraction and verification.
- **Video:** preserve frame/time ranges, audio/transcript provenance, and
  temporal uncertainty before verification.
- **URL:** record URL structure and retrieved article text separately; never
  treat domain identity alone as proof.
- **Multimodal:** run text, image, OCR, and evidence baselines independently
  before fusion. Report each modality's contribution and missing inputs.
