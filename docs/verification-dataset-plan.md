# Verification Dataset Plan

Datasets remain isolated by task, language, modality, annotation process, and
license. No dataset is imported or trained in this planning phase.

| Task | Dataset type | Candidate data | Required annotation |
|---|---|---|---|
| PHEME rumor veracity | Social-media rumor threads | PHEME | `true`/`false`/`unverified` |
| Indian fake-news classification | Hindi/Indian article text | HinFakeNews, pending license and leakage decisions | Article label |
| Evidence retrieval | Claim-evidence pairs | FEVER or comparable corpus | Relevant documents/passages |
| Claim entailment | Claim plus passages | FEVER-style claim/evidence data | Support/refute/not enough information |
| Real-world web verification | Claims with web evidence | AVeriTeC | Verdict plus web evidence/answers |
| Image misinformation | Image plus claim/veracity labels | Future approved image dataset | Image/claim labels |
| OCR verification | Screenshot/image text plus labels | Future approved OCR corpus | OCR text and verification labels |
| URL analysis | URL/source/article metadata | Future approved source dataset | Provenance and task-specific labels |

HinFakeNews must not be used as an evidence-grounded benchmark unless the
release contains reliable evidence annotations. PHEME and HinFakeNews results
must be reported separately. Cross-dataset testing is a later experiment, not
an implicit combined score.

## Import gates

Before any dataset is copied into a training workflow, record its exact
release, license, source, label semantics, collection period, duplicate policy,
and grouping unit. Run leakage checks before creating splits. Preserve raw
files locally when redistribution is not explicitly permitted.
