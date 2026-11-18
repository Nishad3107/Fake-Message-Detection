# AVeriTeC Evaluation Protocol

AVeriTeC is maintained as a separate claim-verification benchmark. It is not
merged with PHEME rumor classification or HinFakeNews article classification.

## Evidence modes

- `fixture_only`: unit tests only; never research metrics.
- `oracle_evidence`: validates nested question/answer/citation schemas. It is
  not retrieval performance because the gold evidence is supplied.
- `train_evidence_corpus`: a local index made only from training answers. It is
  useful for development checks but is not an independent web-verification
  result.
- `external_reference_corpus`: future independently supplied evidence.
- `live_web`: disabled and unavailable in this project phase.

## Split and leakage rules

Development claims and labels remain untouched during corpus construction. The
train evidence corpus must not contain development claims, URLs, or evidence
content. Duplicate and near-duplicate checks happen before any benchmark score.
Evidence publication dates must be available before making a temporal claim;
otherwise temporal leakage cannot be assessed and research readiness remains
blocked.

## Metrics

Future retrieval experiments will report Recall@1, Recall@3, Recall@5, MRR,
evidence hit rate, citation completeness, and per-language/source-category
results. Entailment and final verification will separately report macro F1,
per-class recall, calibration, abstention quality, and temporal correctness.
Accuracy alone is insufficient.

`INSUFFICIENT_EVIDENCE` must remain an explicit outcome. Conflicting sources
must remain `CONFLICTING_EVIDENCE`; they must not be collapsed into a forced
true/false label. Oracle evidence results must be named oracle results and never
reported as end-to-end retrieval performance.

## Current readiness

The local train/dev files support schema and oracle tests. A train-derived
corpus can support development retrieval after leakage checks. A knowledge
store or independently supplied evidence corpus is required before honest
research retrieval metrics. The blind test set is also required for final
evaluation. No model training occurs during preparation.
