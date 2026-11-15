# HinFakeNews Data Decision Record

Date: 2026-11-15

## Decision

Prepare a separate local Indian benchmark using `CONTENT` as the first text
field, while preserving `TITLE`, `URL`, `domain`, and source-row metadata. Do
not combine it with PHEME and do not train a model in this phase.

The supplied catalog evidence defines `BOOL=0` as `FAKE` and `BOOL=1` as
`REAL`. The normalized labels are therefore `fake` and `real`, while the
original numeric value is retained as `label_original`.

## Evidence

- Workbook audit: one sheet, `Real&Fake Combined`, 68,754 records, four fields.
- Catalog description: Hindi news corpus; real articles from mainstream/RNI-
  registered outlets; fake articles from IFCN-approved fact-checkers.
- Catalog metadata screenshot: source organization PES University, web scraping,
  visibility Open, and license field `NA`.
- The description warns that fake rows may include a justification beside the
  article. This is a potential label-leakage feature.

## Alternatives considered

- `TITLE` only: safer for an initial short-text baseline, but it discards the
  article body.
- `TITLE + CONTENT`: useful later, but should be a separate experiment.
- URL/domain features: excluded from the first text model because they may
  encode source or label shortcuts; evaluated later as an ablation.
- PHEME merge: rejected because PHEME is social-media rumor veracity and this
  is Hindi news-article classification.

## Preparation result

- Raw rows: 68,754.
- Exact duplicate rows: 1,167.
- Duplicate normalized contents removed: 1,577 rows.
- Conflicting-label duplicate groups: 0.
- Prepared local rows: 67,177.
- Empty content rows: 0.
- No dates were available in the workbook schema.
- Seven URLs were invalid or scraping artifacts and were assigned no domain.
- Training: not performed.

The local prepared CSV and split manifest are ignored by Git. The first split is
deterministic and domain-grouped with seed 42; no domain or exact-content
overlap exists across train, validation, and test. Near-duplicate overlap was
not calculated and remains a limitation.

## Risks and approval

The license remains unresolved because the catalog displays `NA`. The raw
workbook and processed files must remain local until the dataset owner confirms
research, redistribution, and deployment permissions. Before training, fake
justifications must be separated from article text or the task must be changed
to an explicitly evidence-grounded task.
