# AVeriTeC Internal BM25 Retrieval Results

**INTERNAL DEVELOPMENT RESULT**  
**NOT OFFICIAL AVeriTeC PERFORMANCE**  
**NOT LIVE-WEB PERFORMANCE**

## Protocol

This benchmark uses only `internal_train.jsonl` to construct the searchable
corpus and `internal_dev.jsonl` for queries. It does not use the official
AVeriTeC development split for scoring, the blind test, live retrieval, APIs,
LLMs, or the large knowledge store.

Claims, URLs, evidence text, and evidence-document identities are canonicalized
before the benchmark. Evaluation gold evidence is checked against the corpus
before scoring. Any overlap stops evaluation.

## Query and BM25 variants

The runner records the exact query for every item and compares original,
normalized, entity-focused, location-focused, date-focused, question-only, and
claim-plus-question queries. It compares evidence-only and title-plus-evidence
fields, original/normalized tokenization, stopword settings, and a documented
lightweight suffix-stemming ablation. Results are reported at k = 1, 3, 5, and
10.

## Interpretation limitation

The clean split deliberately prevents development gold evidence from appearing
in the internal training corpus. Consequently, the current local corpus has no
gold-evidence coverage for internal-dev scoring. Recall@k, MRR, and hit rate are
therefore expected to be zero and must not be interpreted as a realistic measure
of web evidence retrieval quality. This is a corpus-coverage limitation, not a
claim that the claims are false.

The generated JSON, CSV, failure report, summary, and charts under
`backend/data/reports/verification/` are local ignored artifacts. The summary
identifies the best configuration only within this coverage-limited internal
comparison. An independent knowledge store is still required before meaningful
retrieval benchmarking or real-world claim verification.

## Recorded run

The completed run used `2,454` internal-train records, `613` internal-dev
records, and `5,222` unique train evidence documents. It evaluated `68,656`
query/configuration items in `153.642992` seconds. Gold-evidence coverage was
`0.0`, so Recall@1, Recall@3, Recall@5, Recall@10, MRR, and evidence hit rate
were all `0.0` across the compared variants. All `68,656` failures were
classified as `missing_evidence`, because the clean split intentionally keeps
development gold evidence out of the searchable training corpus.

The fastest tied configuration was evidence-only input with normalized
tokenization and stopwords enabled, using the original claim query. This is a
runtime tie-break only; it is not evidence that the query or tokenization is
more accurate. Titles and fact-check-question queries cannot be judged from
this run because the evaluation gold evidence has zero corpus coverage.

The results include breakdowns by label, claim type, location, language,
question count, evidence count, claim length, and query type. Charts are
generated locally and remain ignored by Git.

Run locally:

```bash
make verify-averitec-clean-benchmark
make run-averitec-internal-retrieval
```
