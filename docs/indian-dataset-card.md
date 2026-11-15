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
- **External source/license:** not stated in the workbook. The catalog or release page and any accompanying paper must be checked manually before research redistribution or deployment. License status is therefore **Unknown**.

The audit found an article-style schema: `URL`, `TITLE`, `CONTENT`, and `BOOL`.
It is not PHEME-style social-media thread data. The workbook alone does not
establish the source organization, collection period, annotation protocol, or
the semantic mapping of `BOOL` values.

## Observed schema and labels

| Field | Observed content | Missing values |
|---|---|---:|
| `URL` | Article/source URLs | 0 |
| `TITLE` | Hindi/Latin article headlines | 0 |
| `CONTENT` | Hindi/Latin article body text | 0 |
| `BOOL` | Numeric-looking values stored/read as `0` and `1` | 0 |

Observed label counts are `0: 22,057` and `1: 46,697`. The workbook does not
say whether `0` means fake or real, so no semantic relabeling is performed.
The label is currently called `label_original`; a normalized label may only be
created after the official documentation or dataset owner confirms the mapping.

Both `TITLE` and `CONTENT` are plausible text fields. The audit therefore
requires an explicit choice before preparation. A first text-only experiment
should predeclare whether it uses title, content, or a documented combination,
and should exclude `URL` from the feature text until source leakage is audited.

## Language and text audit

The workbook is Hindi-oriented but contains substantial Latin-script text and
mixed rows. Exact observed statistics are in:

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

It is **not ready for training** in this repository because the label meaning,
text-column choice, provenance, and license terms still require confirmation.

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
2. whether `BOOL=0` and `BOOL=1` map to fake/real, and the exact terminology;
3. whether `TITLE`, `CONTENT`, or a documented combination is the input;
4. whether URL/source fields may be retained as metadata but excluded from the
   first text-only model;
5. the source/time/group-aware split policy.

The raw workbook must remain outside Git unless the license explicitly permits
redistribution. The preparation script refuses to proceed without explicit
sheet and column arguments, and training is intentionally not included in the
audit command.
