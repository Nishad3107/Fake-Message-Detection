# Misinformation Detection in Social Media Rumor Messages Using NLP

## 1. Abstract

This project develops a reproducible academic prototype for classifying the
veracity labels of social-media rumor source messages. It uses the PHEME Rumour
Detection and Veracity Classification dataset, stateless text preprocessing,
training-only TF-IDF features, and classical machine-learning baselines. A
majority classifier, Multinomial Naive Bayes, Logistic Regression, and Linear
SVM were compared. Grouped cross-validation and an event-level split were used
to reduce event and thread leakage. The frozen final candidate was tuned
Multinomial Naive Bayes. On the untouched test events it achieved accuracy
`0.225589` and macro F1 `0.211379`.

The result is intentionally reported as a weak unseen-event generalization
result. The system learns language patterns associated with historical PHEME
annotations; it does not retrieve evidence, verify claims, or establish
factual truth. Its probabilities are uncalibrated model outputs and should not
be used as factual certainty or as an autonomous moderation decision.

## 2. Introduction

Social platforms allow claims to spread quickly during breaking-news events.
This makes early assessment difficult: posts are short, context is incomplete,
language changes across events, and a post can be true, false, or simply not
yet verifiable. Natural-language processing (NLP) can identify recurring
lexical patterns, but it cannot replace source checking or human judgment.

The project therefore emphasizes uncertainty, event-level generalization,
explainability, and reproducibility rather than a claim of universal truth
detection.

## 3. Problem Statement

Given the text of a rumor source message, predict one of three PHEME veracity
labels:

- `true`
- `false`
- `unverified`

The prototype is scoped to English-oriented PHEME source-tweet text and
historical annotation labels. It does not perform evidence retrieval, source
credibility assessment, bot detection, multilingual analysis, or autonomous
fact-checking.

## 4. Objectives

The project objectives were to:

1. Validate and prepare a real rumor dataset while preserving raw data.
2. Make text cleaning deterministic and explainable.
3. Compare classical sparse-text classifiers fairly.
4. Prevent event, thread, and vocabulary leakage.
5. Report uncertainty and statistical feature associations.
6. Integrate the frozen model into a FastAPI backend and React frontend.
7. Persist local analysis history without external inference APIs.
8. Provide reproducible commands, reports, tests, and Docker/CI support.

## 5. Dataset Description

The project uses the **PHEME dataset for Rumour Detection and Veracity
Classification**. The official source is the [Figshare dataset record](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078)
and its DOI is [10.6084/m9.figshare.6392078](https://doi.org/10.6084/m9.figshare.6392078).
The dataset is organized by event, rumor/non-rumor branch, thread, source
tweet, reactions, and annotation files.

Observed local dataset values from `pheme_validation.json`,
`pheme_quality.json`, and `eda_report.json` are:

| Quantity | Value |
|---|---:|
| Events | 9 |
| Raw threads | 6,425 |
| Rumor threads | 2,402 |
| Non-rumor threads | 4,023 |
| Processed records | 2,271 |
| `true` | 1,064 (46.85%) |
| `false` | 635 (27.96%) |
| `unverified` | 572 (25.19%) |
| Unsupported/unlabelled annotations | 124 |
| Normalized duplicate records removed during preparation | 7 |

The raw validator found no missing source tweets, malformed annotations, empty
source text, or duplicate tweet IDs. Non-rumor threads were counted and
validated but excluded from this three-class veracity CSV. Annotation and
tweet-content rights remain subject to the official source terms: the dataset
card records CC BY 4.0 for annotations while Twitter retains rights to tweet
content. Raw data is not committed to Git.

Text statistics before model preprocessing were 22–148 characters, mean
118.48, median 126, standard deviation 21.77; and 3–29 words, mean 18.82,
median 19, standard deviation 3.92. The preparation script assumes English and
does not perform language detection.

## 6. Data Preparation

The raw event tree was validated before normalization. Only source-tweet text
was used as a record; reactions and future thread content were not used as
features. The mapping used the PHEME annotation fields:

- `true=1` → `true`
- `misinformation=1` → `false`
- both flags equal to zero → `unverified`
- unsupported or missing mappings → excluded and counted

The processed schema is `text`, `label`, `event`, `thread_id`, and
`source_tweet_id`. Raw files remain unchanged. The 20-row synthetic CSV is
retained only for development fallback and was not used for these academic
results.

## 7. Exploratory Data Analysis

The EDA report found no missing values, exact duplicate rows, or exact duplicate
texts after preparation. It found 17 high-similarity near-duplicate pairs.
Observed surface patterns included URLs in 1,447 records, hashtags in 1,420,
numbers in 1,585, emojis in 9, and repeated punctuation in 65. The EDA's broad
`@`-character detector counted mentions in 343 records; the preprocessing
report's stricter valid-mention regex counted 336. These are different
operational definitions, not contradictory dataset sizes.

Common terms and bigrams include URLs, `police`, `breaking`, event names, and
event-specific phrases such as `war memorial`, `massey hall`, `parliament
hill`, and `sydney cafe`. These patterns may be shortcuts for event identity,
not general indicators of veracity.

Generated charts are in `backend/data/reports/eda/`, including label
distribution, text lengths, label-conditioned lengths, top words, bigrams,
missing values, and duplicate summaries. The dataset is not severely imbalanced
under the project's 80%-majority rule, but `true` is the largest class.

## 8. Data Splitting Strategy

A random row split is unsafe because related event vocabulary and conversation
context can appear on both sides of the split. The deterministic manifest keeps
events and threads disjoint:

| Split | Events | Records | `true` | `false` | `unverified` |
|---|---|---:|---:|---:|---:|
| Train | ebola-essien, ferguson, ottawashooting, prince-toronto, sydneysiege | 1,507 | 720 | 399 | 388 |
| Validation | charliehebdo, putinmissing | 467 | 193 | 125 | 149 |
| Test | germanwings-crash, gurlitt | 297 | 151 | 111 | 35 |

The recorded seed is 42. The event allocation is deterministic and exhaustive,
not a random row sample. All three labels occur overall in each split, but
three individual events contain only one label. With only nine events, this is
one held-out allocation rather than a complete estimate of future event
generalization. Grouped cross-validation is therefore important.

## 9. Text Preprocessing

Stateless cleaning preserves labels and metadata and is applied consistently:

- Unicode is normalized.
- Text is lowercased.
- URLs are replaced consistently.
- Mentions are replaced consistently.
- Hashtags are retained as meaningful word content.
- Whitespace is normalized.
- Repeated punctuation is normalized.
- Retweet markers are handled.
- Negation words are preserved.
- Stopwords are not removed blindly.
- Stemming and lemmatization are not applied automatically.

Cleaning changed 2,270 of 2,271 records. Character mean changed from 118.48 to
99.66 and word mean from 18.82 to 16.33. This cleaning does not learn from
labels. TF-IDF vocabulary learning is a separate learned step and is fitted
only on training text (or train plus validation text for the already-frozen
final fit after model selection).

## 10. Feature Extraction

TF-IDF represents each message using weighted word features: terms frequent in
a document but less common across the fitting corpus receive more weight. The
baseline comparison used word n-grams `(1, 2)`, `min_df=2`, `max_df=0.95`,
sublinear TF, and Unicode accent stripping, producing 4,243 features from the
training vocabulary. The frozen final configuration used `(1, 2)`, `min_df=3`,
`max_df=0.95`, `sublinear_tf=false`, and 3,173 features after fitting on train
plus validation. No validation or test text was used to learn the final test
vocabulary.

## 11. Model Comparison

The validation comparison used the same leakage-safe baseline feature pipeline.
Macro F1 was primary because class performance is uneven; weighted F1 and
accuracy were secondary.

| Model | Accuracy | Macro F1 | Weighted F1 | Training seconds | Confidence output |
|---|---:|---:|---:|---:|---|
| Majority baseline | 0.413276 | 0.194949 | 0.241704 | 0.000380 | probabilities |
| Multinomial Naive Bayes | 0.428266 | 0.250895 | 0.292193 | 0.001614 | probabilities |
| Logistic Regression | 0.415418 | **0.343505** | **0.374263** | 0.021566 | probabilities |
| LinearSVC | 0.346895 | 0.292217 | 0.316932 | 0.006420 | decision margins, not probabilities |

Logistic Regression was the strongest original baseline by validation macro F1.
The majority baseline establishes how much a trivial class-frequency strategy
can achieve and exposes the cost of ignoring minority classes.

## 12. Hyperparameter Tuning

The tuning runner used five-fold `StratifiedGroupKFold` on the training split,
with `event` as the group and `label` as the target. Each fold fitted a fresh
TF-IDF vectorizer only on its fold-training text. It tested 144 configurations
(48 per model family), varying n-grams, `min_df`, sublinear TF, and model
parameters such as Logistic Regression `C`/class weighting, Naive Bayes
`alpha`/prior fitting, and LinearSVC `C`/class weighting.

The selected candidate was Multinomial Naive Bayes:

```text
TF-IDF: ngram_range=(1, 2), min_df=3, sublinear_tf=False, max_df=0.95
Model: alpha=0.1, fit_prior=False
Grouped CV macro F1: 0.197411 ± 0.113824
Validation macro F1: 0.372437
```

The large grouped-CV standard deviation signals event instability. The final
test was excluded from tuning and model selection.

## 13. Final Model

The frozen final model is tuned Multinomial Naive Bayes with the configuration
above. It was selected by grouped-CV mean macro F1 and validation macro F1
before test evaluation. The final artifact is
`backend/models/final/final_pipeline.joblib`. The backend loads it once at
startup and returns the original PHEME labels, class scores, confidence
category, statistical feature associations, warnings, and model metadata.

## 14. Final Evaluation

The final pipeline was fitted on 1,507 train plus 467 validation records and
evaluated once on 297 records from two unseen events. The frozen test results
are:

| Metric | Test result |
|---|---:|
| Accuracy | 0.225589 |
| Macro precision | 0.228806 |
| Macro recall | 0.267193 |
| Macro F1 | 0.211379 |
| Weighted precision | 0.278353 |
| Weighted recall | 0.225589 |
| Weighted F1 | 0.232157 |
| Log loss | 1.694920 |
| OVR ROC-AUC | 0.417280 |
| OVR PR-AUC | 0.295843 |
| Multiclass Brier score | 0.314570 |
| Expected calibration error | 0.378936 |

Per-class results:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `false` | 0.184615 | 0.108108 | 0.136364 | 111 |
| `true` | 0.384615 | 0.264901 | 0.313725 | 151 |
| `unverified` | 0.117188 | 0.428571 | 0.184049 | 35 |

With label order `false`, `true`, `unverified`, the confusion matrix is:

```text
[[12, 51, 48],
 [46, 40, 65],
 [ 7, 13, 15]]
```

## 15. Error Analysis

The largest confusion pairs were `true → unverified` (65), `false → true`
(51), `false → unverified` (48), and `true → false` (46). The `false` class
was hardest by F1 and had recall 0.108108. `unverified` precision was only
0.117188 because many true and false messages were assigned to it.

Germanwings accuracy was 0.252101 with macro F1 0.239687. Gurlitt accuracy was
0.118644; three-class macro F1 was unavailable because no false examples were
present in that event. The model saw event-specific topics and vocabulary that
did not transfer reliably.

There were 25 high-confidence wrong predictions (confidence at least 0.80),
181 low-confidence predictions under 0.60, and 38 correct predictions below
0.60. Feature explanations associated terms such as `prince`, `massey`,
`cirillo`, and `nathan` with classes. These are statistical associations and
may be event shortcuts, not evidence or causal explanations.

## 16. System Architecture

```text
User
  → React frontend
  → FastAPI API
  → stateless text preprocessing
  → frozen TF-IDF vectorizer
  → frozen Multinomial Naive Bayes model
  → probabilities and explanation
  → SQLite history
  → dashboard
```

Docker Compose runs the backend and frontend with health checks. GitHub Actions
runs source/API-contract tests, frontend tests/build, linting, formatting, and
repository security checks without requiring private PHEME data or ignored
model artifacts.

## 17. Frontend and Backend

The React application provides a landing page, analyzer, history dashboard,
methodology page, and model-results page. The analyzer shows the original
label, class probabilities, confidence category, warning, influential terms,
model metadata, and an explicit limitation notice. The dashboard reads real
history and summary metrics from FastAPI.

The backend exposes `/health`, `/api/v1/model-info`, `/api/v1/predict`,
`/api/v1/predict/batch`, `/api/v1/history`, and `/api/v1/metrics`. It validates
input, handles missing/corrupt artifacts safely, persists successful analyses
locally, and returns generic errors without internal paths or stack traces.

## 18. Security and Privacy

The API enforces 10–5,000 non-whitespace characters per post, a 50-item batch
limit, and a 1 MB request-body limit. CORS is restricted to configured
development origins and production rejects wildcard origins. Security headers,
strict Pydantic types, parameterized SQLite queries, and safe React text
rendering are enabled.

Successful local predictions are stored in SQLite. Raw message text is not
written to application logs, no external inference API or API key is used, and
there is no automatic retention job. The prototype has no authentication and
SQLite is not encrypted. Production use requires explicit CORS configuration,
HTTPS, access control, retention review, and stronger operational controls.

## 19. Limitations

- Unseen-event test macro F1 is low at 0.211379.
- Event-specific distribution shift is substantial.
- Only nine historical events are available.
- PHEME annotations and available tweets reflect historical sampling and
  annotation judgments.
- Model probabilities are not calibrated factual probabilities.
- The system does not retrieve evidence or verify sources.
- The prototype is English-oriented and has no multilingual support.
- Some event splits have highly uneven or single-label composition.
- SQLite history lacks encryption, authentication, and automatic retention.
- The result must not be used for censorship or high-stakes decisions.

## 20. Future Work

Future experiments should add newer and more diverse events, evidence retrieval,
source-credibility and temporal features, calibrated models, human-in-the-loop
review, multilingual evaluation, temporal and cross-dataset testing, and a
managed database with authentication. Transformer comparisons may be added only
after preserving event-level leakage controls and establishing a fair compute
and evaluation protocol.

## 21. Conclusion

The main contribution is a reproducible and transparent NLP workflow rather
than a claim of reliable truth detection. The project validates real PHEME data,
uses event-level evaluation, compares explainable classical models, freezes a
candidate before one-time test evaluation, exposes its weaknesses through error
analysis, and integrates the result into a tested full-stack prototype. The
low unseen-event score is an important finding: a stronger validation result
does not guarantee generalization. Human review and independent evidence remain
necessary.

