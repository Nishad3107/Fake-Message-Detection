# Dataset Card: PHEME Rumour Detection and Veracity Classification

Status: **Not locally obtained yet**  
Audit date: 2026-11-14  
Project role: candidate academic dataset; the existing synthetic dataset remains the development/demo dataset.

## 1. Dataset name and official source

**PHEME dataset for Rumour Detection and Veracity Classification**.

Official record: <https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078>  
DOI: <https://doi.org/10.6084/m9.figshare.6392078>

The official record describes Twitter rumors and non-rumors collected during breaking news, organized by event and tweet ID. Rumor directories include source-tweet data, reactions, `annotation.json`, and `structure.json`.

## 2. Access and usage terms

The official record states that the annotations are provided under CC BY 4.0, while Twitter retains ownership and rights to the tweet content. The project must preserve the raw download, follow the source/platform terms, cite the dataset, and avoid redistributing tweet text unless permitted.

The automated download check reached the official Figshare page, but the direct 44.38 MB download endpoint returned HTTP 403 in the current environment. No PHEME files were downloaded, copied, or fabricated.

## 3. Purpose and mapping to this project

PHEME is a better academic fit than the current 20-row synthetic file because it contains real social-media conversation data and veracity annotations associated with breaking-news rumors. It supports the stated topic “Fake/Misinformation Message Detection using NLP” while making the task definition explicit.

The recommended first task is **rumor veracity classification on annotated rumor source posts** with the three labels:

- `true`
- `false`
- `unverified`

The raw non-rumor branches are not silently mapped into `true`. The preparation script counts and skips them in its default veracity mode. A separate rumor-vs-non-rumor experiment may be added later with a separately documented label policy.

## 4. Dataset format

### Expected raw tree

After downloading and extracting the official archive, place it under:

```text
backend/data/raw/pheme/
└── <event>/
    ├── rumours/
    │   └── <tweet_id>/
    │       ├── source-tweet/*.json
    │       ├── reactions/*.json
    │       ├── annotation.json
    │       └── structure.json
    └── non-rumours/
        └── <tweet_id>/...
```

The preparation script expects the documented `annotation.json` files and source tweet JSON. It accepts direct `veracity`/`label` values when present and also understands the common PHEME `true`/`misinformation` annotation flags.

### Normalized schema

`backend/data/processed/pheme_veracity.csv` will contain:

| Column | Meaning |
|---|---|
| `text` | Source tweet text, whitespace-normalized but not factually edited |
| `label` | `true`, `false`, or `unverified` |
| `source_event` | PHEME event directory |
| `source_tweet_id` | Original tweet directory identifier |

The raw data is never modified. The existing model training script currently expects only `text` and `label`; it must be explicitly adapted to the three-class PHEME task before any final training.

## 5. Records and labels

| Field | Observed value in this repository |
|---|---|
| Raw PHEME records | **Unknown** — PHEME is not downloaded locally |
| Processed PHEME records | **Unknown** — preparation has not run |
| Label distribution | **Unknown** — must be reported by the preparation script |
| Dataset collection period | **Unknown from the local files**; the release describes breaking-news events |
| Language | English is expected; not yet measured locally |
| Social platform | Twitter/X content and conversation structure |

No PHEME record count, label count, duplicate count, or text statistic is claimed until the official files are supplied and processed. The preparation script writes these values to `backend/data/reports/pheme_quality.json`.

## 6. Missing values, duplicates, and quality checks

The preparation script will report:

- Raw annotation-file count.
- Empty or unreadable annotation/source tweet records.
- Missing text/label values.
- Case-insensitive duplicate source text count.
- Processed record count and removed-record count.
- Label distribution.
- Average, minimum, and maximum text length.

Current PHEME values are **Unknown** because the raw release is not present locally.

## 7. Known biases and leakage risks

- Event selection is not representative of all social-media conversations.
- Breaking-news rumors may differ from ordinary misinformation or political claims.
- Labels reflect annotation judgments at a historical point in time and may not remain correct as later evidence appears.
- Twitter/X users and events are not a representative sample of the public.
- Deleted, protected, or unavailable tweets can create missingness and survivorship bias.
- Conversation structure and event names can leak the label if they are included as features.
- Replies posted after a clarification or debunking may leak future knowledge into an early-detection task.
- Duplicate or near-duplicate tweets across reactions and events can inflate scores.
- Splitting individual tweets randomly can leak event-specific vocabulary across train and test sets.

## 8. Recommended split

Do not use a naive random row split for the final academic result. First inspect the number of events and class coverage. If the event count supports it, use a **grouped event-level split**, approximately 70% train, 15% validation, and 15% test, with no event appearing in more than one split. If the event count is too small, use leave-one-event-out evaluation and report the event-wise results.

The split policy must be recorded alongside the experiment seed and the exact processed-file hash.

## 9. Preprocessing decisions

The preparation stage will:

1. Read source tweet text without editing the raw files.
2. Normalize line breaks and repeated whitespace.
3. Remove records with missing text or an unavailable veracity label.
4. Deduplicate normalized source text and report the number removed.
5. Preserve event and tweet identifiers for grouping and auditability.

The later model stage may lowercase text, remove URLs and punctuation, preserve hashtag terms, and fit TF-IDF only on the training split. It must not use event identifiers, annotation metadata, reactions, or future conversation content as text features unless that is a separately defined experiment.

## 10. Ethical considerations

The labels describe annotated rumors, not an objective universal truth function. The system should not be used to censor users, identify individuals, make high-stakes decisions, or label protected groups. Tweet content may contain personal data and must be handled under applicable platform terms, research ethics, and privacy requirements. Results should be presented as model outputs with uncertainty and human review, not final fact-checks.

## 11. Why PHEME is better than the current synthetic dataset

The current `backend/data/sample_dataset.csv` has only 20 authored examples, no external provenance, no annotation protocol, and no social-media event context. PHEME offers real platform-formatted posts, event-grouped conversation structure, and veracity annotations. It therefore supports a defensible academic dataset card and leakage-aware evaluation. It also introduces important access, privacy, historical-platform, and event-bias limitations that must be reported rather than hidden.

## 12. Preparation command

After the official archive has been manually downloaded and extracted into `backend/data/raw/pheme/`, run:

```bash
make prepare-dataset
```

The command writes:

```text
backend/data/processed/pheme_veracity.csv
backend/data/reports/pheme_quality.json
```

This step does not train a model or overwrite the synthetic development dataset.
