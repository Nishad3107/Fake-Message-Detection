# Project Audit and ML Development Plan

Audit date: 2026-11-14

## Current project objective

The repository is a prototype misinformation-detection system for short, social-media-style English posts. It predicts one of two language-pattern classes: `misinformation` or `likely_true`. The name and product copy describe misinformation detection, while the current training examples are synthetic claim-like posts written for development.

This is not currently a fact-checker. The model does not retrieve sources, compare claims with evidence, or establish whether a statement is true. Its output is a probabilistic prediction based on text patterns learned from the training corpus. It is also not a spam classifier: there is no `spam`/`ham` label, sender metadata, or SMS-specific task. It is adjacent to fake-news classification, but the current dataset is not a real news-article corpus and does not contain provenance or fact-check evidence.

## Current technology stack

- Backend: Python 3.11 virtual environment, FastAPI, Pydantic, Uvicorn.
- ML/NLP: pandas, NumPy, scikit-learn, TF-IDF word n-grams, Logistic Regression, joblib.
- Persistence: SQLite through a small repository layer in `backend/app/db/database.py`.
- Frontend: React, TypeScript, Vite, Tailwind CSS, Lucide icons, Framer Motion.
- Testing/quality: pytest, Vitest, React Testing Library, Black, Ruff, TypeScript build.
- Operations: Dockerfiles and Docker Compose with model/history volumes and health checks.

## Current folder structure

```text
.
├── .env.example
├── .gitignore
├── Makefile
├── README.md
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── app/
│   │   ├── api/routes.py
│   │   ├── core/config.py
│   │   ├── db/database.py
│   │   ├── main.py
│   │   ├── ml/model_utils.py
│   │   ├── ml/predict.py
│   │   ├── ml/train.py
│   │   ├── models/schemas.py
│   │   └── services/prediction_service.py
│   ├── data/sample_dataset.csv
│   ├── data/history.db
│   ├── models/classifier.joblib
│   ├── models/vectorizer.joblib
│   ├── models/metrics.json
│   ├── requirements.txt
│   └── tests/
└── frontend/
    ├── Dockerfile
    ├── package.json
    ├── package-lock.json
    ├── vite.config.ts
    ├── vitest.config.ts
    └── src/
        ├── App.tsx
        ├── App.test.tsx
        ├── components/SiteShell.tsx
        ├── pages/{LandingPage,AnalyzerPage,DashboardPage,AboutPage}.tsx
        ├── services/api.ts
        └── styles.css
```

There is no accidental `misinformation-detection-system/` nested folder and no nested Git repository. There are generated caches and Python bytecode directories locally; they are ignored and are not source components.

## Existing ML workflow

1. `backend/app/ml/train.py` reads a CSV with `text` and `label` columns.
2. Text is lowercased, URLs and punctuation are removed, whitespace is normalized, and hashtags are converted to `hashtag_<term>` tokens.
3. The data is split with a fixed random seed using a stratified 70/30 holdout.
4. `TfidfVectorizer(ngram_range=(1, 2), max_features=5000)` is fit only on training text.
5. A class-balanced `LogisticRegression(max_iter=1000)` classifier is fit on the training vectors.
6. Accuracy, misinformation precision, misinformation recall, F1, confusion matrix, and train/test sample counts are written to `backend/models/metrics.json`.
7. The classifier and vectorizer are serialized to `backend/models/classifier.joblib` and `vectorizer.joblib`.
8. At startup, `PredictionService` loads the artifacts and automatically retrains if they are missing or incompatible.
9. Prediction uses `predict_proba`, detected keyword terms, and coefficient/vectorizer contributions to produce an explanation. The result is persisted to SQLite.

The current workflow avoids a common leakage problem because the vectorizer is fit after the train/test split. It is still only a single holdout experiment, not a robust cross-validation or time/event-separated evaluation.

## Existing dataset details

The only repository dataset is `backend/data/sample_dataset.csv`.

Facts measured directly from the file:

- 20 data records.
- Columns: `text`, `label`.
- Labels: 10 `likely_true`, 10 `misinformation`.
- English text, inferred from inspection of all rows.
- Average raw text length: approximately 81.7 characters.
- Format: short, authored claim-like/social-media-style sentences.
- Source/provenance: local synthetic development examples; no external source, annotator, fact-check report, license, or collection date is recorded.

The current saved metrics report 1.0 for accuracy, precision, recall, and F1 on six holdout records. This is not evidence of real-world performance. The test set is extremely small and the examples are stylistically separable. The metrics file correctly includes a warning that they are not representative.

## Existing model details

The only trained model is `TF-IDF Logistic Regression Baseline`, version `0.1.0`.

- Feature representation: sparse TF-IDF unigrams and bigrams.
- Classifier: class-balanced Logistic Regression.
- Confidence: the largest `predict_proba` value, exposed as model confidence rather than truth probability.
- Explainability: detected heuristic terms plus the highest positive feature contributions for the predicted class.
- Training duration: not recorded by the current script. No measured training-time claim should be made.
- Model comparison: none has been run in this repository.

## Existing frontend and backend status

### Backend

Implemented endpoints:

- `GET /health`
- `GET /api/v1/model-info`
- `POST /api/v1/predict`
- `POST /api/v1/predict/batch`
- `GET /api/v1/history`
- `GET /api/v1/metrics`

The API validates empty, short, and overlong posts; reports model-unavailable errors; persists successful predictions; exposes OpenAPI through FastAPI; and has CORS configuration. SQLite history currently contains local runtime records and is ignored by Git.

### Frontend

The frontend has a landing page, analyzer, dashboard, and methodology page. It includes responsive Tailwind styling, Framer Motion transitions, confidence/keyword/explanation UI, low-confidence messaging, history summaries, a classification chart, a recent-confidence chart, and a basic Vitest/React Testing Library shell test.

The routing is a small hash-based router in `App.tsx`, not React Router. The analyzer imports `getModelInfo` but currently displays model metadata returned with each prediction rather than loading model information separately. There is only one frontend test, so analyzer success/error rendering and dashboard API behavior are not directly component-tested.

## Current application flow

```text
Landing page
   ↓ hash navigation
Analyzer page
   ↓ fetch POST /api/v1/predict
FastAPI validation
   ↓
PredictionService → clean_text → saved TF-IDF vectorizer → Logistic Regression
   ↓
keywords + influential terms + confidence + explanation
   ↓
SQLite history write
   ↓
JSON response → analyzer result card
   ↓
Dashboard GET /api/v1/history → summary and recent analysis views
```

## Current run and test status

Verified during this audit:

- `make test`: 12 backend tests passed, with one upstream Starlette/httpx deprecation warning.
- `make train`: completed and wrote model artifacts and metrics.
- `make format`: Black passed.
- `make lint`: Ruff passed.
- `cd frontend && npm test`: one frontend test passed.
- `cd frontend && npm run build`: Vite/TypeScript production build passed.
- `docker compose config`: passed.
- The repository is on branch `main`.
- The working tree contains intentional uncommitted work from the existing development session; this audit did not discard or overwrite it.

The current Makefile now prefers `.venv/bin/python`, which prevents the earlier `ModuleNotFoundError` caused by invoking a different system Python. Live HTTP checks previously confirmed backend health, real prediction, invalid-input handling, SQLite history, frontend serving, and CORS. Full browser automation was not available in this environment.

## Problems and incomplete academic components

1. The dataset is synthetic, tiny, and not traceable to real fact-checking annotations.
2. The current label `likely_true` is a development label, not a verified truth judgment.
3. Only one model has been tested; there is no baseline comparison table.
4. Metrics are from a six-record holdout and are not statistically meaningful.
5. No cross-validation, confidence calibration, bootstrap intervals, event split, or time split exists.
6. Training duration, hardware/runtime, inference latency, and memory use are not recorded.
7. There is no notebook or experiment registry for reproducible academic analysis.
8. Dataset license, collection date, annotation protocol, class definitions, and provenance are absent.
9. Source-level and topic leakage risks have not been addressed because no real corpus is present.
10. The database layer has lightweight startup migration logic rather than a formal migration tool.
11. Docker Compose validates structurally, but actual image/runtime verification depends on a running Docker daemon and available image/build storage.
12. Frontend test coverage is minimal; there are no direct analyzer, error-state, or dashboard component tests.
13. The model explanation is feature contribution, not causal evidence or source verification.

## Dataset options to evaluate next

Do not adopt any option until its license, access method, exact release, preprocessing, and label mapping are documented. Counts below are source-reported or locally measured; they are not claims about a downloaded copy.

### 1. LIAR

- Source: PolitiFact-derived dataset described in Wang, “Liar, Liar Pants on Fire,” and mirrored with documentation by the UCSB AI repository: <https://arxiv.org/abs/1705.00648>.
- Reported size: approximately 12.8K human-labeled short statements.
- Labels: six truthfulness levels: pants-fire, false, barely-true, half-true, mostly-true, true.
- Language/format: English, short political claims/statements.
- Advantages: professionally fact-checked claims, balanced enough for a meaningful academic baseline, close to the current short-text input shape.
- Limitations: politics-heavy, not ordinary social posts, labels are ordinal/multi-class and collapsing them to binary requires a defensible rule, possible source/topic leakage, and the original release/licensing must be checked before redistribution.
- Fit: strongest candidate if the project is reframed as short fact-checked claim classification rather than general social-media misinformation.

### 2. FakeNewsNet

- Source: official repository and paper: <https://github.com/KaiDMML/FakeNewsNet> and <https://arxiv.org/abs/1809.01286>.
- Reported structure: PolitiFact and GossipCop news samples with news content, social context, and dynamic information; exact usable record counts vary by release and crawl completeness.
- Labels: fake/real or source-specific labels depending on the derived release.
- Language/format: English news articles plus social-media context, not just isolated posts.
- Advantages: closest conceptual match to misinformation spreading through social media; supports studying content and social context.
- Limitations: access/recollection complexity, platform/API availability, missing/deleted content, privacy and terms-of-use concerns, domain differences between politics and celebrity news, and possible label/source leakage.
- Fit: strong research option if the project expands beyond text-only classification and can satisfy access/licensing requirements.

### 3. PHEME Rumour Dataset

- Source: Figshare release linked by the PHEME project: <https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078>.
- Reported corpus: nine breaking-news events, conversation threads with source tweets and replies; published summaries report 6,425 threads and 105,354 tweets, with 2,402 rumor and 4,023 non-rumor threads. Verify exact files and counts after download.
- Labels: rumor/non-rumor, with additional true/false/unverified veracity labels for rumor threads.
- Language/format: English Twitter/X conversation trees.
- Advantages: directly aligned with social-media rumor detection, event structure, and temporal context.
- Limitations: historical platform data, event/domain imbalance, label definitions differ from “misinformation,” missing/deleted content, and thread-level leakage risk.
- Fit: best fit if the project objective becomes rumor detection on social-media conversations rather than generic truth classification.

### 4. NELA-GT-2019

- Source: dataset documentation and Harvard Dataverse DOI are linked from the official repository: <https://github.com/mgruppi/nela-gt-2019> and <https://doi.org/10.7910/DVN/O7FWPO>.
- Reported size: 1,118,821 articles from 261 sources collected during 2019.
- Labels: source-level Reliable, Mixed, and Unreliable aggregates based on media-bias/fact-check assessments.
- Language/format: English news articles with title, body, source, date, author, and URL metadata.
- Advantages: large scale, rich metadata, useful for source-level reliability and domain-shift experiments.
- Limitations: source labels are proxies, not article-level truth; label aggregation and source identity can leak the answer; not social-media posts; large storage/compute requirements.
- Fit: useful for a news-source reliability study, not the current short-post objective without a deliberate reframing.

### 5. ISOT Fake News Dataset

- Source: commonly distributed research dataset; documented in the research literature, including the review at <https://pmc.ncbi.nlm.nih.gov/articles/PMC7303704/>.
- Reported size: 44,898 articles: 23,481 fake and 21,417 real.
- Labels: fake/real.
- Language/format: English news articles with title/body/date/subject; real articles are reported as Reuters-sourced and fake articles from multiple unreliable sources.
- Advantages: easy binary baseline, larger than the current sample, straightforward TF-IDF experiments.
- Limitations: political/news-domain bias, source/style leakage, old web content, unclear redistribution/license handling across mirrors, and not social-media text.
- Fit: acceptable for a fake-news article baseline, but weaker alignment than LIAR or PHEME for this project’s social-post framing.

### 6. UCI SMS Spam Collection — contrast option, not recommended

- Source: UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>.
- Reported size: 5,574 English SMS messages.
- Labels: `ham` and `spam`.
- Language/format: short English SMS messages.
- Advantages: clean, accessible, and ideal for demonstrating text classification mechanics.
- Limitations: spam is not misinformation; labels concern unwanted messaging rather than truthfulness; strong domain mismatch.
- Fit: not suitable for the current misinformation objective except as a separate spam-classification project or preprocessing tutorial.

## Model options to compare

### Logistic Regression — current baseline

Use TF-IDF features with class-balanced Logistic Regression. It is fast, strong on sparse text, gives probability estimates, and exposes coefficients that support simple explanations. It is the correct first baseline, but its probabilities should eventually be calibrated and its linear boundary cannot model all context.

### Multinomial Naive Bayes

Compare Multinomial Naive Bayes on count or TF-IDF features. It is extremely fast and a useful classical text baseline, but its conditional-independence assumption is simplistic and its probabilities can be poorly calibrated. It is valuable for an academic comparison because it provides a distinct modeling assumption.

### Linear SVM

Compare LinearSVC on the same TF-IDF features. It is often a strong sparse-text classifier and can provide useful coefficient-based explanations. Raw SVM margins are not probabilities, so confidence must be calibrated with a held-out calibration split or `CalibratedClassifierCV` without leaking test data.

### Random Forest or another tree-based model

Use only as a controlled comparison, ideally on reduced/dense engineered features rather than the full high-dimensional sparse vocabulary. Trees can capture nonlinear interactions but are usually less efficient and less competitive than linear models on raw TF-IDF text. Their explanation is also less direct.

### Optional transformer

DistilBERT or a suitable small encoder could be a later experiment if compute, model downloads, licensing, and dataset size are available. It should not replace the classical baseline. It adds training cost, reproducibility and explainability challenges, and may still fail under topic or event shift. No transformer should be introduced before the dataset and evaluation protocol are fixed.

## Recommended academic direction

1. Choose one clearly defined task: recommended choice is English short-claim or rumor classification, not “truth detection” in the absolute sense.
2. Select LIAR for fact-checked short claims or PHEME for social-media rumor threads. FakeNewsNet is a later extension if social-context data access is feasible.
3. Define label mapping before training. Preserve original multi-class labels where possible; if binary mapping is necessary, justify every merged class.
4. Create a documented data card: source, license, collection period, fields, class distribution, language, missingness, duplicates, and known biases.
5. Compare Logistic Regression, Multinomial Naive Bayes, and calibrated Linear SVM using the same leakage-safe preprocessing.
6. Use stratified cross-validation on the training set and a locked test set. For PHEME or news datasets, also use event/source/time-aware splits to measure generalization.
7. Report accuracy, macro and weighted precision/recall/F1, per-class metrics, confusion matrix, calibration/Brier score where probabilities are used, training time, inference latency, and resource assumptions.
8. Add an experiment table or notebook after the dataset is selected. Record seeds, package versions, feature settings, model hyperparameters, and artifact hashes.
9. Present the output as a learned language-pattern or rumor-risk signal. Do not describe it as a final fact-check or a probability that the claim is objectively true.

## Recommended next steps

- Decide between the LIAR and PHEME directions before downloading data.
- Provide or approve the selected dataset and its license/usage terms.
- Add a reproducible dataset ingestion script rather than replacing `sample_dataset.csv` manually.
- Add an academic experiment runner that compares the classical models and records timing/metrics.
- Add a data card and model card under `docs/`.
- Add cross-validation and event/source-aware evaluation where the selected dataset supports it.
- Extend frontend tests for analyzer success/error states and dashboard rendering.
- Consider formal SQLite migrations and Docker health-gated `depends_on` after the ML protocol is fixed.

