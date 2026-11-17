# HinFakeNews-V1 Dataset Card

## Status

This card records an audit of the locally supplied workbook. No Indian model
was trained and no normalized training CSV was generated. The workbook remains
ignored by Git and must not be redistributed until its terms are confirmed.

## Dataset identity and provenance

- **Name supplied:** `HinFakeNews-V1.xlsx`
- **Repository copy:** `backend/data/raw/indian/hinfakenews/HinFakeNews-V1.xlsx`
- **SHA-256:** `cffa4f4877a824a43284b623602eeedb7bab530146e49816142dc222f361e5f3`
- **Workbook sheet:** `Real&Fake Combined`
- **Workbook records:** 68,754 (68,755 rows including the header)
- **Workbook metadata:** creator and last editor are recorded in the ignored source metadata report; this is provenance metadata, not a license grant.
- **Catalog source organization:** PES University.
- **Catalog collection method:** Web scraping.
- **Catalog visibility:** Open.
- **Catalog license field:** `NA`.
- **Catalog authors/uploader:** Authors and uploader are listed in the catalog screenshot; they are not a substitute for a license.
- **External source/license:** The workbook does not contain a license. The catalog displays `NA`, not an explicit license grant. Manual confirmation is required before redistribution, GitHub publication, or deployment. License status is therefore **Unknown**.

The audit found an article-style schema: `URL`, `TITLE`, `CONTENT`, and `BOOL`.
It is not PHEME-style social-media thread data. The supplied dataset description
states that the corpus covers Hindi political, social, and communal news;
real news is sourced from mainstream and Registrar of Newspapers for India
(RNI)-registered outlets, while fake news is sourced from publicly available
fact-checking platforms approved by the International Fact-Checking Network
(IFCN). The stated purpose includes fake-news detection and explainable or
evidence-grounded research.

## Observed schema and labels

| Field | Observed content | Missing values |
|---|---|---:|
| `URL` | Article/source URLs | 0 |
| `TITLE` | Hindi/Latin article headlines | 0 |
| `CONTENT` | Hindi/Latin article body text | 0 |
| `BOOL` | Numeric-looking values stored/read as `0` and `1` | 0 |

Observed label counts are `0: 22,057` and `1: 46,697`. The supplied dataset
description confirms the mapping `0=FAKE` and `1=REAL`. The preparation script
preserves the original values in `label_original` and may create a documented
normalized label only with this explicit mapping.

The accompanying catalog screenshot reports `URL`, `TITLE`, and `CONTENT` as
`VARCHAR`, `BOOL` as `NUMERIC`, and zero nulls. It reports 64,519 distinct URLs,
65,187 distinct titles, 67,202 distinct content values, and two distinct BOOL
values. The displayed BOOL mean is approximately `0.6791895744` with standard
deviation `0.47`; these are descriptive dataset statistics, not model results.

Both `TITLE` and `CONTENT` are plausible text fields. The audit therefore
requires an explicit choice before preparation. A first text-only experiment
should predeclare whether it uses title, content, or a documented combination,
and should exclude `URL` from the feature text until source leakage is audited.

The dataset description also states that fake-article content includes a
justification alongside the news item. This is a high-severity leakage risk:
fact-checking language may directly reveal the label or provide post-hoc
evidence. An article-classification experiment must separate the article from
the justification, or explicitly define a separate evidence-grounded task.
The justification must not be treated as ordinary article text without a
documented protocol.

## Language and text audit

The workbook is Hindi-oriented according to the supplied description, although
the catalog metadata displays language as `N.A.`. It contains substantial
Latin-script text and mixed rows. Exact observed statistics are in:

- `backend/data/reports/indian/hinfakenews_workbook_audit.json`
- `backend/data/reports/indian/hinfakenews_text_audit.json`
- `backend/data/reports/indian/hinfakenews_leakage_audit.json`

For `TITLE`: 68,721 rows contain Devanagari, 28,613 contain Latin characters,
and 28,582 contain both. Mean length is 94.8497 characters and 15.7868
whitespace-delimited words. There are 3,567 repeated title values and 1,167
exact duplicate rows.

For `CONTENT`: 68,399 rows contain Devanagari, 62,533 contain Latin
characters, and 62,178 contain both. Mean length is 2,059.4434 characters and
396.0993 whitespace-delimited words. There are 1,552 repeated content values,
1,559 rows containing URLs, 3,950 containing hashtags, 5,802 containing
mentions, 1,680 containing emoji, and 223 containing HTML-like markup.

These are descriptive observations, not evidence that the labels are correct.
Raw text was not changed.

## Leakage and quality risks

The audit identified the following risks:

- repeated titles/content and exact duplicate rows can cross a random split;
- URL domains, URL paths, and fact-checker branding may correlate with `BOOL`;
- article templates, scraping artifacts, or label-related wording may be
  shortcuts rather than generalizable language signals;
- source, outlet, topic, and collection time may be confounded with labels;
- the label semantics and annotation process are not documented in the workbook;
- fake-row justifications may directly encode the fake label or fact-checking
  outcome;
- no near-duplicate detector was applied to the workbook in this first audit;
- no image or image-path field was found, so this file does not support an
  image-only or multimodal experiment by itself.

The first split should be source/domain- and time-aware where metadata permits,
with exact and near-duplicate groups kept within one partition. A random row
split is not academically defensible until these risks are addressed.

## Suitability and separation from PHEME

HinFakeNews is potentially suitable as a separate Indian Hindi news-article
classification benchmark. It is not a replacement for the PHEME social-media
rumor-veracity benchmark. Results must be reported separately; they must not be
combined into a single accuracy or F1 score, and a model trained here must not
be described as validated on PHEME.

It is not ready for training in this repository until the input text-column
policy, justification-separation policy, release provenance, and license terms
are recorded. The label mapping is now documented, but `TITLE` versus `CONTENT`
(or a documented combination) must still be selected before normalization.

The first local preparation decision now selects `CONTENT`, preserves `TITLE`
and URL/domain metadata, and excludes URL/domain values from model text. The
deduplicated local file contains 67,177 rows after removing 1,577 repeated
normalized contents; no conflicting-label duplicate groups were observed. A
deterministic domain-grouped split manifest was generated with 48,545 training,
9,892 validation, and 8,740 test rows. These files are ignored and local only.
The content audit found 13,815 records with fact-check markers, 12,253 with
debunking language, 21,535 with label-related words, and 32,924 records with
at least one structural leakage flag. These are screening signals, not proof
that every flagged segment is an annotation. The MinHash/LSH near-duplicate
screen found 283 near-duplicate pairs affecting 551 records, including 12
cross-domain and 12 cross-split pairs; no cross-label pairs were found. The
current domain split is therefore unsafe for final evaluation until the pairs
are reviewed and the split policy is regenerated if required. Fake-row
justifications and the article/annotation boundary still require manual review
before training.

## Ethical and legal considerations

News content may contain personal data, copyrighted text, and sensitive claims.
Keep the workbook local and ignored unless redistribution is explicitly
permitted. Do not send raw rows to external APIs. Document the annotation
process, source concentration, language coverage, and known errors before
making model claims. A classifier output is a learned statistical signal, not
proof that an article is true or fake.

## Required manual approvals

Before normalization, the project owner must confirm:

1. the official source and license/usage terms;
2. whether the supplied RNI/IFCN provenance description applies to this exact
   workbook release;
3. how fake-article justifications will be excluded from the first
   article-classification experiment, or whether a separate evidence task is
   intended;
4. whether URL/source fields may be retained as metadata but excluded from the
   first text-only model;
5. the source/time/domain-aware split policy and near-duplicate policy.

The raw workbook must remain outside Git unless the license explicitly permits
redistribution. The preparation script refuses to proceed without explicit
sheet and column arguments, and training is intentionally not included in the
audit command.
