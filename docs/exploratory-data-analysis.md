# Exploratory Data Analysis

## Scope

Run:

```bash
make analyze-dataset
```

The workflow selects `backend/data/processed/pheme_veracity.csv` when it exists. Otherwise it analyzes only `backend/data/sample_dataset.csv` as synthetic development data. The datasets are never combined, and EDA does not train a model.

The JSON report is written to `backend/data/reports/eda_report.json`. Charts are written to `backend/data/reports/eda/` and are ignored by Git.

## Dataset analyzed

This analysis uses real prepared PHEME data:

- 2,271 processed source-tweet records
- 9 events and 2,271 unique thread IDs
- `true`: 1,064 records (46.85%)
- `false`: 635 records (27.96%)
- `unverified`: 572 records (25.19%)
- Missing values: 0
- Exact duplicate texts: 0 after preparation
- Character length: minimum 22, maximum 148, mean 118.48, median 126, standard deviation 21.77
- Word length: minimum 3, maximum 29, mean 18.82, median 19, standard deviation 3.92

The 20-row synthetic sample remains available as a fallback but was not included in this analysis.

## Linguistic patterns

URLs appear in 1,447 records, hashtags in 1,420, mentions in 343, numbers in 1,585, and emojis in 9. Common surface tokens include URL-shortener components (`http`, `t`, `co`), `police`, `breaking`, event names, and hashtags. The most frequent bigrams include `t co`, `http t`, `war memorial`, `massey hall`, `parliament hill`, and `sydney cafe`.

These patterns reflect the event and Twitter context as well as message content. URLs, event names, and hashtags must be reviewed carefully before modeling because they may act as shortcuts.

## Duplicate and quality analysis

The validator found 6,425 threads, 6,425 valid source tweets, and 6,425 valid annotation files. It found no malformed JSON, missing source tweets, empty source text, or duplicate tweet IDs. Preparation excluded 4,023 non-rumor threads, 124 unsupported/unlabelled rumor annotations, and 7 normalized duplicate texts.

The processed CSV contains no exact duplicate rows or texts. The EDA near-duplicate check found 17 high-similarity pairs. These pairs should be inspected and kept within a single evaluation partition where appropriate.

## Leakage and splitting risks

PHEME contains event and thread structure. A random row split can place related posts from the same event or thread in both training and testing, allowing event-specific vocabulary or conversation context to inflate results. Keep every `thread_id` in one partition and use event-level grouping for the academic split. A grouped 70/15/15 split is recommended if all labels remain represented; otherwise use leave-one-event-out evaluation.

The fields `event`, `thread_id`, and `source_tweet_id` are audit metadata, not text features. Reactions, annotation metadata, and future conversation content must not enter an early-post model unless a separate experiment explicitly defines them.

## Class balance and modeling implications

The analyzer flags severe imbalance when one label is at least 80% of records. PHEME is not severely imbalanced under that rule, although `true` is the largest class at 46.85%. Later modeling should report macro and per-class metrics rather than relying on accuracy alone.

The observed URLs, hashtags, mentions, punctuation, short event-specific phrases, and near-duplicates should inform preprocessing and leakage controls. TF-IDF must be fit only on the training partition. Explainable linear baselines remain appropriate for the first comparison, with event-grouped evaluation.

## Charts

Generated charts include label distribution, text length distribution, text length by label, top words by label, top bigrams, missing values, and duplicate summary. Each chart title identifies the source as PHEME real data.

## Limitations

This EDA does not establish factual correctness, causality, generalization, or model accuracy. It does not perform language identification, semantic fact verification, bot detection, or comprehensive semantic near-duplicate detection. Conclusions must account for event selection, platform availability, annotation judgments, deleted content, historical context, and demographic or platform bias.

No final accuracy claim is made here.
