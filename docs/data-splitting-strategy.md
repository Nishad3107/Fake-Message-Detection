# Leakage-Safe PHEME Data Splitting Strategy

## Why random row splitting is unsafe

PHEME is organized around breaking-news events and discussion threads. A random row split can place messages from the same event in training and testing, allowing event-specific names, hashtags, URLs, and vocabulary to act as shortcuts. If future versions include multiple records per thread, a random split could also place related posts from one conversation in different partitions.

The current processed file has 2,271 records and 2,271 unique threads, so there are no repeated thread rows today. The thread constraint is still enforced because the raw dataset is conversation-oriented and the processed schema preserves `thread_id`.

## Selected strategy

The repository uses a deterministic exhaustive event-level allocation:

- 5 events for training
- 2 events for validation
- 2 events for testing
- all three labels required in every subset
- no random row sampling
- recorded seed: `42` for experiment reproducibility, although the current selection is exhaustive and deterministic

The script evaluates every 5/2/2 event assignment that contains `true`, `false`, and `unverified` in every subset. It selects the assignment closest to target record proportions of approximately 65%/17.5%/17.5%, while also considering label-distribution similarity.

## Exact event allocation

| Split | Events | Records | `true` | `false` | `unverified` |
|---|---|---:|---:|---:|---:|
| Train | ebola-essien, ferguson, ottawashooting, prince-toronto, sydneysiege | 1,507 | 720 | 399 | 388 |
| Validation | charliehebdo, putinmissing | 467 | 193 | 125 | 149 |
| Test | germanwings-crash, gurlitt | 297 | 151 | 111 | 35 |

The full event names and counts are stored in `backend/data/splits/pheme_split_manifest.json`.

## Verification results

- Event overlap: none
- Thread overlap: none
- Every processed record assigned exactly once: yes
- All labels present in every split: yes
- Records per thread: exactly one in the current processed dataset

The grouping report is stored at `backend/data/reports/grouping_report.json`.

## Limitations

The 9 events are highly uneven in size. Consequently, validation is 20.57% of records and test is 13.08%, rather than exact 15–20% partitions. Three events contain only one label, so they cannot provide balanced within-event evidence. No events were removed or relabelled to improve the split.

This is one held-out event allocation, not a substitute for grouped cross-validation. The test result will describe generalization to the selected historical events, not all future social-media misinformation. Grouped cross-validation or leave-one-event-out evaluation should be considered for the final academic comparison, especially because there are only nine events.

## Effect on evaluation

Accuracy alone may hide poor performance on `false` or `unverified`. Future model evaluation should report macro and per-class precision, recall, F1, confusion matrices, and probability calibration where applicable. All preprocessing and TF-IDF fitting must happen inside each training partition, and event/thread metadata must not be used as text features.

During a viva, explain that event-level separation reduces topic and conversation leakage, but the small number of events and historical platform bias limit generalization. The split is leakage-safe by construction, not evidence that the model is factually verifying claims.

## Reproduce the split

```bash
make create-splits
```

This command reads only `backend/data/processed/pheme_veracity.csv`, writes the grouping report and split manifest, and does not train a model or create duplicate dataset files.
