# Exploratory Data Analysis

## Scope

The reproducible EDA command is:

```bash
make analyze-dataset
```

It selects `backend/data/processed/pheme_veracity.csv` when prepared PHEME data exists. Otherwise it analyzes only `backend/data/sample_dataset.csv` and marks the output as **synthetic development data**. The files are never combined, and EDA does not train a model.

The JSON report is written to `backend/data/reports/eda_report.json`. Charts are written to `backend/data/reports/eda/`; generated charts and reports are intentionally ignored by Git.

## Dataset analyzed

At the time of this analysis, PHEME is not locally available because the official archive download was inaccessible from the development environment. Therefore, the observed results are based only on the 20-row synthetic sample:

- `likely_true`: 10 records
- `misinformation`: 10 records
- Missing cells: 0
- Exact duplicate texts: 0
- Character length: minimum 70, maximum 97, mean 81.65

These values describe the development fixture only and are not representative of social-media misinformation. When PHEME is prepared, rerun the command and use the generated JSON report for the actual statistics.

## What the workflow measures

The analyzer records schema and dtypes, missing values, duplicate rows, normalized duplicate texts, label counts and percentages, character and word-length statistics, empty/very short messages, common words, unigrams, bigrams, URLs, hashtags, mentions, numbers, emojis, and punctuation.

It also creates label, length, vocabulary, missing-value, and duplicate-summary charts. Every chart title identifies the source as either PHEME real data or synthetic development data.

## Leakage and splitting risks

Exact duplicate texts and repeated texts with conflicting labels are reported. A practical token-set near-duplicate check is also included. For PHEME, source-event and thread relationships can place highly related records in both training and test sets if rows are split randomly. Event-level or thread-level grouping should therefore be used when defining academic splits. `source_event`, `source_tweet_id`, annotation metadata, reactions, and future conversation context should not become text features unless they are part of a separately defined experiment.

The sample dataset has no event metadata, so its split risk cannot be generalized. It is too small for meaningful research evaluation.

## Class balance and modeling implications

The report flags severe imbalance when a single label represents at least 80% of records. The sample is balanced by construction, but that does not establish balance in PHEME. The observed vocabulary and length distributions should guide later preprocessing and model comparison, while keeping vectorizer fitting inside the training partition to prevent leakage.

Short texts, URLs, hashtags, mentions, punctuation, and repeated expressions may affect TF-IDF features. The existing explainable baseline should be compared with other linear text models only after the real dataset, split policy, and label mapping are finalized.

## Limitations

This EDA does not establish factual correctness, causal relationships, generalization, or model accuracy. It does not perform language identification, semantic fact verification, bot detection, or comprehensive near-duplicate detection. PHEME-related conclusions must account for event selection, Twitter availability, annotation judgments, deleted content, historical context, and possible platform and demographic bias.

No final accuracy claim is made here. The current results are only from synthetic development data until PHEME is supplied and prepared.
