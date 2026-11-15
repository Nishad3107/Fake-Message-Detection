# Improvement Experiment Matrix

Planning matrix only. No experiment in this table has been run during this
audit. “Expected runtime” is a planning category; actual elapsed time must be
recorded by the runner and must not be invented.

| ID | Dataset | Modality | Model / features | Split strategy | Primary metric | Secondary metrics | Expected runtime | Hardware | Status | Result path |
|---|---|---|---|---|---|---|---|---|---|---|
| P0 | PHEME | Text | Frozen tuned MNB + TF-IDF `(1,2)` | Existing event split | Macro F1 | Per-class recall, balanced accuracy, ECE, log loss | Existing 62.545390 s workflow | CPU | Frozen control | Existing `backend/data/reports/model_comparison/` |
| P1 | PHEME | Text | Logistic Regression + word TF-IDF | Event split; grouped CV | Macro F1 | Per-class metrics, calibration | CPU pilot; measure | CPU | Planned control | `backend/data/experiments/<run>/` |
| P2 | PHEME | Text | LinearSVC + word TF-IDF + calibration | Event split; grouped CV | Macro F1 | Balanced accuracy, calibration, margins | CPU pilot; measure | CPU | Planned control | `backend/data/experiments/<run>/` |
| P3 | PHEME | Text | MultinomialNB + word TF-IDF | Event split; grouped CV | Macro F1 | Log loss, calibration, per-class recall | CPU pilot; measure | CPU | Planned control | `backend/data/experiments/<run>/` |
| P4 | PHEME | Text | Character n-gram Logistic Regression | Event split; grouped CV | Macro F1 | Noise robustness, balanced accuracy | CPU pilot; measure | CPU, higher RAM | Planned | `backend/data/experiments/<run>/` |
| P5 | PHEME | Text | Word + character TF-IDF fusion | Event split; grouped CV | Macro F1 | Per-class recall, event macro F1 | CPU pilot; measure | CPU, higher RAM | Planned | `backend/data/experiments/<run>/` |
| P6 | PHEME | Text | Calibrated selected classical model | Event split; validation calibration only | ECE | Brier, log loss, coverage/abstention | CPU pilot; measure | CPU | Planned | `backend/data/experiments/<run>/` |
| P7 | PHEME | Text | Noise/paraphrase robustness suite | Held-out transformed test only | Macro F1 delta | Per-class recall delta, abstention | CPU inference; measure | CPU | Planned | `backend/data/experiments/<run>/robustness/` |
| I0 | HinFakeNews | Hindi text | TF-IDF + Logistic Regression | Article/source/time grouping after audit | Macro F1 | Balanced accuracy, class recall, ECE | CPU pilot after import | CPU | Awaiting approval/download | `backend/data/experiments/<run>/` |
| I1 | HinFakeNews | Hindi text | Character + word fusion | Source/time grouped split | Macro F1 | Cross-source generalization, robustness | CPU pilot; measure | CPU, higher RAM | Planned | `backend/data/experiments/<run>/` |
| I2 | IFND | Indian text | Classical models | Article/source/topic grouping | Macro F1 | Duplication/augmentation audit, balanced accuracy | CPU pilot after audit | CPU | Awaiting provenance review | `backend/data/experiments/<run>/` |
| I3 | FactDrill | Indian social text | Classical text baseline | Claim/source/time grouping | Macro F1 | Language-wise recall, cross-language gap | CPU pilot after access review | CPU | Awaiting access/schema review | `backend/data/experiments/<run>/` |
| I4 | FakeNewsIndia | Incident/social links | Incident/source analysis, not immediate classifier | Incident/source/time grouping | Task-specific; predeclare | Link coverage, source concentration | CPU analysis; measure | CPU | Awaiting release inspection | `backend/data/experiments/<run>/` |
| T0 | Approved Indian corpus | Multilingual text | MuRIL fine-tuning | Source/time/event grouping | Macro F1 | Per-language recall, ECE, cross-domain F1 | GPU pilot; measure | GPU recommended | Planned investigation | `backend/data/experiments/<run>/` |
| T1 | Approved Indian corpus | Multilingual text | IndicBERT fine-tuning | Source/time/event grouping | Macro F1 | Per-language recall, calibration | GPU pilot; measure | GPU recommended | Planned investigation | `backend/data/experiments/<run>/` |
| T2 | PHEME + approved Indian corpus | Multilingual text | XLM-R transfer/domain test | Dataset-held-out and cross-domain | Cross-domain macro F1 | In-domain gap, per-language recall | GPU pilot; measure | GPU recommended | Planned; never merge blindly | `backend/data/experiments/<run>/` |
| T3 | Approved Indian corpus | Multilingual text | mBERT fine-tuning | Source/time/event grouping | Macro F1 | Per-language recall, calibration | GPU pilot; measure | GPU recommended | Planned investigation | `backend/data/experiments/<run>/` |
| V0 | Approved image corpus | Image only | Manipulation/reuse baseline | Image/source/event grouping | Macro F1 or AUROC per task | Robustness, source-disjoint gap | GPU/CPU pilot; measure | GPU optional | Dataset required | `backend/data/experiments/<run>/image/` |
| O0 | Approved screenshot corpus | OCR → text | OCR quality + frozen text baseline | Source/event grouping | OCR-aware macro F1 | OCR CER/WER, OCR confidence, ECE | CPU/GPU pilot; measure | CPU/GPU | Dataset and OCR policy required | `backend/data/experiments/<run>/ocr/` |
| U0 | Approved URL corpus | URL/source | URL metadata baseline | Domain/time grouping | Macro F1 or AUROC per task | Cross-domain gap, calibration | CPU pilot; measure | CPU | Source data required | `backend/data/experiments/<run>/url/` |
| M0 | Approved multimodal corpus | Text + image | Independent baselines + late fusion | Event/source/image grouping | Macro F1 | Ablations, missing-modality gap, ECE | GPU pilot; measure | GPU recommended | After modality baselines | `backend/data/experiments/<run>/multimodal/` |

## Matrix rules

- P0 remains the immutable control; do not overwrite its artifact or reports.
- Every run receives a unique versioned output directory.
- Test data is not read for tuning or threshold choice.
- Group keys are chosen before looking at test performance.
- Dataset-specific labels are never silently mapped to PHEME labels.
- `Result path` is a planned location, not evidence that an experiment has run.
- Actual runtime, package versions, seed, dataset hash, configuration, and
  hardware are mandatory metadata.

## Recommended order

1. P1–P7 controls and calibration/robustness on PHEME.
2. Manual approval and import audit for HinFakeNews.
3. I0–I1 Hindi text baselines.
4. FactDrill schema/access review and I3 social-media baseline.
5. Transformer investigation T0–T3.
6. Image/OCR/URL branches, then multimodal fusion M0.

