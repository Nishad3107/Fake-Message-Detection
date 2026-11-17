# HinFakeNews Content Leakage Policy

## Current finding

The prepared corpus contains 67,177 deduplicated records. The content audit
flagged:

- 13,815 records with fact-check markers;
- 6,302 with verdict/conclusion-style markers;
- 22,110 with claim markers;
- 12,253 with debunking language;
- 21,535 with label-related words;
- 3,885 with source or metadata markers;
- 221 with HTML-like content;
- 2,032 with repeated-prefix templates.

These signals are strongly concentrated in fake rows for several categories.
They may be legitimate article language, appended fact-check explanations, or
scraping artifacts. The audit does not assume which without manual review.

## Why this matters

The dataset description says fake articles may include a justification beside
the news item. If that justification contains words such as “fake”, “false”,
“fact check”, or a verdict, a classifier can learn the annotation rather than
the article claim. This would inflate validation results and would not measure
general misinformation detection.

URLs, domains, publisher names, and templates can create a similar shortcut.
The first content benchmark therefore keeps URL/domain values as metadata and
does not include them in model text.

## Candidate views

- `content_raw`: preserved for audit only; not approved for training.
- `content_clean`: Unicode and whitespace normalization only; not approved until
  the justification boundary is known.
- `content_article_only_candidate`: currently identical to normalized content,
  because no unambiguous boundary was found automatically; manual review is
  required.
- `title_only`: a possible lower-leakage baseline, but it is a separate task
  and was not selected as the current primary view.
- `title_content_candidate`: future experiment only.

## Near-duplicate findings

The prepared data has no exact normalized-content duplicates after preparation.
The reproducible approximate screen used eight-hash MinHash LSH over word
trigrams, with four bands of two hashes, followed by Jaccard verification at
`0.85`. It found 283 candidate near-duplicate pairs affecting 551 records,
including 12 cross-domain pairs and 12 pairs whose domains map to different
existing splits. No cross-label near-duplicate pairs were found by this method.

Because the existing domain split separates some near-duplicate pairs, it is
not safe to call the current split final for model evaluation. Near-duplicate
groups must be assigned to one partition, or the affected pairs must be
manually reviewed under a predeclared policy. No rows were deleted in this
audit.

## Recommendation

**Manual content-boundary review required. Near-duplicate cleanup and split
regeneration are required before final evaluation. License confirmation is also
required before research training or public use.**

The next approved experiment should use a documented article-only text view,
exclude URL/domain/source metadata, group exact and near duplicates within one
partition, and report macro F1, per-class recall, calibration, and source/
domain-held-out performance separately.

## Required decisions

1. Identify the boundary between article text and appended fact-check
   justification, or explicitly define an evidence-grounded task.
2. Approve the Jaccard threshold and whether all near-duplicate groups are
   removed or grouped.
3. Regenerate the domain/duplicate-aware split after that decision.
4. Confirm the dataset license and whether local research, redistribution, and
   deployment are permitted.
