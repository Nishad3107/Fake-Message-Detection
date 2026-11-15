# Improvement Roadmap and Research Plan

Status: planning only. No new dataset was downloaded, no experiment was run,
and the frozen PHEME artifact and metrics were not changed.

## 1. Current baseline

The current system is an academic PHEME rumor-veracity prototype:

| Item | Frozen value |
|---|---|
| Dataset | PHEME Rumour Detection and Veracity Classification |
| Processed records | 2,271 |
| Labels | `true`, `false`, `unverified` |
| Split | 1,507 train / 467 validation / 297 test |
| Split control | Event-level, with no event or thread overlap |
| Final model | Tuned Multinomial Naive Bayes |
| Final test accuracy | 0.225589 |
| Final test macro F1 | 0.211379 |
| Validation macro F1 | 0.372437 |
| Grouped CV macro F1 | 0.197411 ± 0.113824 |
| Reproduced experiment runtime | 62.545390 seconds |

The model is a statistical language-pattern classifier, not a fact-checker. Its
probabilities are overconfident and should not be interpreted as factual truth
probabilities.

## 2. Current weaknesses

- Unseen-event performance is substantially below validation performance.
- `false` recall is 0.108108 and `unverified` precision is 0.117188 on test.
- Only nine historical events are available, making event-level estimates
  unstable.
- Feature explanations contain event-specific associations such as names and
  locations; associations are not evidence.
- PHEME is not an Indian-language or Indian-source benchmark.
- The current backend is text-only and exposes one frozen model contract.
- A legacy `PredictionService` can train the old synthetic baseline if called
  directly; production startup uses `FrozenModelService` and does not train.
- The frontend has a fixed three-label text response contract and no modality
  selector.

## 3. Scientific improvement goals

Success must be measured with macro F1, per-class recall and precision,
balanced accuracy, log loss, calibration error, event-level generalization,
cross-dataset generalization, paraphrase/noise robustness, and reproducibility.
Accuracy alone is insufficient. No experiment may mix partitions, tune on test
data, remove difficult examples, or claim that a feature proves truth.

The first improvement goal is not a target such as “near 100% accuracy.” It is
to establish whether a change improves macro F1 and minority-class behavior
without worsening calibration or held-out-event performance.

## 4. Separate research tracks

### Track A — Social-media rumor veracity

Dataset: PHEME. Labels remain `true`, `false`, and `unverified`. Keep event and
thread grouping, and report the existing frozen baseline before every new
comparison.

### Track B — Indian fake-news text classification

Evaluate Indian datasets independently. They differ from PHEME in language,
source type, label meaning, and collection process. Do not merge them with PHEME
without a pre-specified compatible task and a documented domain-shift protocol.

### Track C — Image-only misinformation detection

Input is an image without text. Candidate signals include manipulation evidence,
image reuse, and visual content. Visual appearance alone does not establish
claim truth. This track needs its own labels, split policy, and image-level
leakage audit.

### Track D — Text-in-image detection

```text
Image → OCR → extracted text → text classifier → OCR confidence + warning
```

OCR errors and layout artifacts must be measured separately from classifier
errors. OCR confidence must be shown to users and never hidden as certainty.

### Track E — URL/source analysis

Inputs may include URL structure, redirect behavior, article text where legally
available, and source-history features from a verified dataset. Do not use a
hardcoded “good domains/bad domains” list. Source features must be grouped by
domain and time to prevent source memorization.

### Track F — Multimodal analysis

Text plus image, caption plus attachment, or screenshot plus OCR text should be
combined only after each modality has an independent baseline and evaluation.
Fusion experiments must report whether the gain comes from content or leakage.

## 5. Indian dataset options

The values below are reported only where the cited source documents them. A
license marked “not stated on the reviewed page” requires manual confirmation
before download or redistribution.

| Dataset | Source and documented scope | Language / unit | Labels and modalities | Fit, risks, and manual work |
|---|---|---|---|---|
| **HinFakeNews** | [IndiaAI/AIkosh catalog](https://aikosh.indiaai.gov.in/home/datasets/details/hinfakenews_2.html); approximately 68,000 Hindi real/fake news articles as catalogued | Hindi; article with URL, title, content | Binary `0=FAKE`, `1=REAL`; text and URL fields | **Recommended first feasibility candidate.** Clear catalog description and Indian/Hindi context, but it is article classification rather than PHEME-style social rumor veracity. Exact downloadable files, license terms, duplicate policy, and split manifest must be confirmed manually. |
| **IFND** | [Peer-reviewed paper and dataset description](https://pmc.ncbi.nlm.nih.gov/articles/PMC8520332/), DOI `10.1007/s40747-021-00552-1`; paper reports 56,868 Indian news items and 2013–2021 coverage | Primarily Indian news; article/headline records | Binary `True/Fake`; text, image link, source, category, date | Strong Indian and multimodal context, but the paper documents synthetic/augmented fake text and scraped source structure. This creates duplication, augmentation, outlet/topic leakage, and unclear redistribution/licensing checks. Use only after manual access and provenance review. |
| **FakeNewsIndia** | [Peer-reviewed paper](https://doi.org/10.1016/j.comcom.2022.01.003); 4,803 fake-news incidents from June 2016–December 2019, six Indian fact-checking sites | Indian fact-checking incidents; linked social content | Incident/fake-news collection; 5,031 Twitter links and 866 YouTube links are reported; a balanced real-news label is not documented in the abstract | Strong fit for source/URL/social-impact research, not a ready binary text classifier. Treat as an incident retrieval/impact track unless the released files provide compatible labels. Manual access and terms review required. |
| **FactDrill** | [AAAI/ICWSM dataset paper](https://doi.org/10.1609/icwsm.v16i1.19384); 22,435 fact-checked social-media items, 2013–2020, 13 Indian languages | Multilingual; fact-checked social-media content | Multi-attribute, multi-lingual, multi-media repository; exact label schema and downloadable access must be checked from the release | **High-priority social-media follow-on.** Better conceptual match than article-only data, but task labels, licensing, access, and identity/deletion handling need verification before adoption. |

The first recommended approval decision is **HinFakeNews for a separate Hindi
binary text benchmark**, not a replacement for PHEME. FactDrill should be
investigated in parallel as the preferred social-media/domain-extension option
if its access terms and label schema are suitable.

## 6. Classical text experiments

Run in this order after a dataset is approved:

1. Reproduce the frozen PHEME baseline as a control.
2. Re-run TF-IDF + Logistic Regression with the same grouped split policy.
3. Re-run TF-IDF + LinearSVC with calibration evaluated separately.
4. Re-run Multinomial Naive Bayes.
5. Add character n-gram models for spelling, morphology, and transliteration.
6. Compare word-plus-character feature fusion.
7. Add robustness suites before selecting any “best” model.

Each experiment must preserve group boundaries, fit vocabulary on training data
only, and save configuration, package versions, seed, timing, and result path.

## 7. Multilingual transformer experiments

Potential candidates, to investigate but not download yet:

- [MuRIL](https://huggingface.co/google/muril-base-cased): designed for Indian
  languages and transliterated text; requires fine-tuning and substantial
  memory, with limited simple feature explanations.
- [IndicBERT](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only): Indian
  language coverage; tokenizer and exact checkpoint/license must be confirmed
  before use.
- [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base): broad
  multilingual transfer baseline; not India-specific and computationally
  heavier than TF-IDF.
- [mBERT](https://huggingface.co/google-bert/bert-base-multilingual-cased):
  established multilingual comparison; older model and also requires
  fine-tuning/calibration.

For every transformer record language coverage, checkpoint/version, tokenizer,
GPU/CPU requirements, fine-tuning settings, explainability limitations, and
whether the model can run locally. Do not report pretrained-model claims as
project results.

## 8. Calibration, uncertainty, and robustness

Planned calibration experiments:

- Fit sigmoid or isotonic calibration on training/validation data only.
- Compare expected calibration error, Brier score, log loss, and reliability
  diagrams.
- Select abstention thresholds on validation only.
- Expose `uncertain` as a product state only after defining its policy and
  measuring coverage versus error.

Planned robustness suites:

- Spelling and punctuation noise.
- Hinglish and transliteration.
- Hashtag edits, URL insertion, and emoji insertion.
- Short and long messages.
- Human-reviewed paraphrases.
- Code-mixed Hindi-English.

Robustness test text must not be used to fit vocabulary or tune the final model.

## 9. Image, OCR, URL, and multimodal sequence

The recommended order is:

1. Image-only data audit and manipulation/reuse baseline.
2. OCR-only extraction quality benchmark.
3. OCR text classifier compared with native text baseline.
4. URL/source metadata branch with time/domain grouping.
5. Independent text and image baselines on a multimodal dataset.
6. Late-fusion multimodal model with modality-missing and ablation tests.

This sequence isolates failure modes and prevents an apparently strong fusion
model from hiding a broken modality.

## 10. Evaluation policy

- Dataset-specific test sets remain untouched until final evaluation.
- Event, thread, source, article, domain, and near-duplicate groups remain
  within one partition.
- Duplicate and near-duplicate checks run before splitting.
- Hyperparameters and thresholds use training/validation only.
- Every model reports macro F1, per-class precision/recall/F1, balanced
  accuracy, log loss where valid, calibration error where valid, and timing.
- Accuracy is never reported alone.
- Cross-domain and cross-dataset results are reported separately.
- Indian data is not merged with PHEME unless label/task definitions are shown
  compatible.
- Every run records seed, dataset version/hash, package versions, hardware,
  duration, configuration, and artifact path.
- No model is called a fact-checker, and no source word is presented as proof.
- No near-perfect result is accepted without independent, leakage-audited
  evaluation.

## 11. Backend and frontend policy

No application changes are required for this audit. The current backend can
support additional versions safely only after a model registry/configuration
layer is designed. The current fixed `FINAL_MODEL_ARTIFACT` setting should not
be replaced by ad hoc path selection.

Future changes should likely include:

- Versioned model metadata and an allowlisted artifact registry.
- A model/version field in prediction responses and history.
- Separate endpoint contracts for text, OCR, image, URL, and multimodal input,
  or a carefully validated modality envelope.
- Modality-specific uncertainty, missing-input, and explanation fields.
- Frontend capability discovery from `/model-info` rather than assuming three
  text labels.

Frontend redesign should wait until modality schemas and uncertainty semantics
are finalized. The existing text analyzer should remain unchanged while these
interfaces are researched.

## 12. Compute and storage requirements

Classical TF-IDF experiments are expected to run locally on CPU; exact runtime
must be measured and recorded rather than assumed. Character fusion increases
memory but should remain a CPU pilot. Transformer fine-tuning is expected to
need a compatible GPU or a slower CPU pilot, model caches, and additional
storage; exact requirements depend on checkpoint and sequence length. Image
and multimodal work requires dataset image storage, preprocessing caches, and
larger artifact management.

## 13. Expected manual work

Before the next training phase, the student must approve one Indian dataset,
review its terms, obtain/download it manually where required, record the exact
release/version, and provide its raw files in a documented ignored directory.
GPU access and transformer dependencies are optional until the classical pilot
is complete. Image files or URL access must be supplied only for the relevant
approved track.

## 14. Leakage and inflated-accuracy risks

The highest risks are source/outlet shortcuts, duplicated fact-check articles,
near-identical headlines, synthetic augmentation leakage, event memorization,
URL/domain memorization, and train/test image reuse. A high score from a random
row split or a mixed-source dataset would not demonstrate robust misinformation
detection.

## 15. Definition of success

An improvement is successful only if it:

1. Beats the frozen baseline on predeclared macro F1 or class recall without
   sacrificing calibration.
2. Improves or preserves event-level generalization on untouched events.
3. Shows robustness to documented noisy/paraphrased inputs.
4. Does not rely on duplicate, source, event, or synthetic leakage.
5. Reproduces within documented tolerance on a second run/environment.
6. Reports failures and uncertainty honestly.

## Current recommendation

Do not change the frozen PHEME model yet. Approve and manually obtain
HinFakeNews for a separate Hindi binary feasibility benchmark, then run the
experiment matrix's classical controls before any transformer or multimodal
work. Keep FactDrill as the next social-media-oriented dataset investigation.

