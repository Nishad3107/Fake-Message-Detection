# Dataset Card: PHEME Rumour Detection and Veracity Classification

Status: **Imported and prepared locally**
Audit date: 2026-11-14
Project role: real academic dataset for the documented three-class rumor-veracity task. The 20-row synthetic dataset remains a development fallback only.

## 1. Dataset name and official source

**PHEME dataset for Rumour Detection and Veracity Classification**.

Official record: <https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078>
DOI: <https://doi.org/10.6084/m9.figshare.6392078>

The release contains Twitter rumors and non-rumors from breaking-news events, organized by event and tweet/thread ID. The imported release contains 9 event folders, `rumours` and `non-rumours` branches, `source-tweets/*.json`, reactions, `annotation.json`, and `structure.json`.

## 2. Access and usage terms

The official record states that annotations are provided under CC BY 4.0, while Twitter retains ownership and rights to tweet content. The project preserves the original downloaded archive outside the repository and copies the raw event tree without modification. Raw files and generated datasets remain ignored by Git.

## 3. Purpose and task mapping

PHEME is a stronger academic fit than the synthetic sample because it contains real social-media posts, event context, and veracity annotations. The current task is **three-class veracity classification of annotated rumor source posts**:

- `true`
- `false`
- `unverified`

Non-rumor branches are validated and counted, but are not silently mapped to `true`. A separate rumor-vs-non-rumor experiment would require a separate label policy.

## 4. Raw and normalized format

```text
backend/data/raw/pheme/
└── <event>/
    ├── rumours/
    │   └── <thread_id>/
    │       ├── source-tweets/*.json
    │       ├── reactions/*.json
    │       ├── annotation.json
    │       └── structure.json
    └── non-rumours/
```

The processed file `backend/data/processed/pheme_veracity.csv` contains:

| Column | Meaning |
|---|---|
| `text` | Source tweet text with whitespace normalized |
| `label` | `true`, `false`, or `unverified` |
| `event` | PHEME event directory |
| `thread_id` | Original tweet/thread directory identifier |
| `source_tweet_id` | Source tweet identifier; identical to `thread_id` here |

## 5. Observed records and labels

| Field | Observed value |
|---|---:|
| Events | 9 |
| Total raw threads | 6,425 |
| Rumor threads | 2,402 |
| Non-rumor threads | 4,023 |
| Valid source tweets | 6,425 |
| Valid annotations | 6,425 |
| Processed records | 2,271 |
| `true` | 1,064 (46.85%) |
| `false` | 635 (27.96%) |
| `unverified` | 572 (25.19%) |
| Removed records | 4,154 |

Removed records comprise 4,023 non-rumor threads, 124 unsupported/unlabelled rumor annotations, and 7 normalized duplicate texts. The validation report found no missing source tweets, malformed annotations, empty source text, or duplicate tweet IDs.

## 6. Label mapping

The original PHEME flags are preserved in the raw files. For the processed veracity CSV:

- `true=1` maps to `true`.
- `misinformation=1` maps to `false`.
- Both flags equal to `0` maps to `unverified`.
- Records without a supported mapping are excluded and counted.

No binary transformation has been applied.

## 7. Missing data, duplicates, and quality reports

The validation report is written to `backend/data/reports/pheme_validation.json`. Preparation statistics are written to `backend/data/reports/pheme_quality.json`, and EDA statistics are written to `backend/data/reports/eda_report.json`.

The prepared dataset has no missing values, duplicate rows, or exact duplicate texts. Preparation detected 7 duplicate texts before removal. EDA found 17 high-similarity near-duplicate pairs that require review before model splitting.

## 8. Bias, leakage, and limitations

- The 9-event selection is not representative of all social-media conversations.
- Breaking-news rumors may differ from ordinary misinformation or political claims.
- Labels reflect historical annotation judgments, not an objective universal truth function.
- Twitter users, events, and available posts are not a representative sample of the public.
- Deleted, protected, or unavailable tweets can create survivorship bias.
- Event names, thread IDs, annotation metadata, reactions, and future replies can leak information.
- Near-duplicate messages can inflate evaluation results.
- The dataset is English-oriented; language detection is not performed by preparation.

## 9. Recommended split and metrics

Do not use a naive random row split. The processed data contains 2,271 unique threads across 9 events. Keep every thread in one partition and use an event-grouped approximately 70/15/15 train/validation/test split if class coverage permits. Otherwise use leave-one-event-out evaluation and report event-wise results. Record the seed, event assignments, and processed-file hash.

For future evaluation, report accuracy, macro and weighted precision/recall/F1, per-class metrics, confusion matrix, and—if probabilities are used—calibration or Brier-style diagnostics. No model results are claimed by this dataset card.

## 10. Preprocessing decisions

Preparation reads source tweet text without editing raw files, normalizes repeated whitespace, removes records with missing text or unsupported labels, deduplicates normalized text, and preserves event/thread identifiers. The later model stage may lowercase text, remove URLs and punctuation, preserve useful hashtag terms, and fit TF-IDF only on the training partition.

## 11. Ethical considerations

The labels describe annotated rumors, not absolute facts. The system must not be used to censor users, identify individuals, make high-stakes decisions, or label protected groups. Tweet content may contain personal data and must be handled under platform terms, research ethics, and privacy requirements. Predictions require uncertainty and human review.

## 12. Reproducible workflow

```bash
make validate-pheme
make prepare-dataset
make analyze-dataset
```

This workflow does not train a model or overwrite the synthetic development dataset.
