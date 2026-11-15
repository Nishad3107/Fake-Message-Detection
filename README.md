# Misinformation Detection System

An explainable full-stack application for analyzing social-media posts with a TF-IDF + Logistic Regression baseline. It returns a probabilistic language-pattern signal, not a factual verdict or substitute for source verification.

## Stack

- Backend: Python 3.11+, FastAPI, Pydantic, pandas, NumPy, scikit-learn
- NLP: cleaned text, TF-IDF word n-grams, Logistic Regression
- Frontend: React, TypeScript, Vite, Tailwind CSS
- Quality: pytest, Black, Ruff, type hints
- Persistence: SQLite by default, with a repository boundary for a future PostgreSQL adapter
- Operations: Docker and Docker Compose with health checks

## Quick start

```bash
cd <repository-root>
cp .env.example .env
python3.11 -m venv .venv
source .venv/bin/activate
make install
make test
```

In two terminals:

```bash
make backend
make frontend
```

Open http://localhost:5173. The app includes an overview landing page, analyzer, SQLite-backed dashboard, and methodology page. The backend loads the frozen model artifact from `backend/models/`; it does not train during application startup.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

Frontend: http://localhost:5173 · API: http://localhost:8000 · Swagger UI: http://localhost:8000/docs

For end-to-end verification, run `docker compose config`, start detached with
`docker compose up --build -d`, then run `tools/verify_stack.sh`. The backend
mounts the ignored frozen model and research reports read-only; Docker never
trains or downloads a model. See `docs/docker-and-end-to-end-verification.md`.

## API

`GET /health` reports service and model readiness.

`GET /api/v1/model-info` returns the baseline name, version, algorithm, dataset name, readiness, and evaluation metrics.

`GET /api/v1/history` returns recent SQLite-backed analyses and summary counts. `GET /api/v1/metrics` exposes saved evaluation metrics.

`POST /api/v1/predict`

```json
{"text":"Paste a social media post with at least ten characters."}
```

The response contains the original PHEME `classification` (`true`, `false`, or
`unverified`), class probabilities, a confidence category, detected feature
associations, a short model explanation, and a disclaimer. These are
probabilistic language-pattern outputs, not factual verification. The model
does not browse or retrieve sources.

`POST /api/v1/predict/batch` accepts `{"posts":[{"text":"..."}]}` for up to 50 posts. Every successful prediction is saved to local history.

Requests must contain 10–5000 non-whitespace characters. Batch requests are
limited to 50 posts and request bodies to 1 MB by default. Invalid input returns
HTTP 422; oversized bodies return HTTP 413. If model artifacts cannot be
loaded, prediction returns a sanitized HTTP 503 error. Successful submissions
are stored in local SQLite history; see `docs/privacy-and-data-handling.md`.

## Training

The sample CSV at `backend/data/sample_dataset.csv` requires `text` and `label` columns. To retrain:

```bash
make train
```

The script cleans text, performs a fixed-seed 70/30 stratified holdout, fits TF-IDF and Logistic Regression with class balancing, reports accuracy/precision/recall/F1 plus a confusion matrix and sample counts, and saves `classifier.joblib`, `vectorizer.joblib`, and `metrics.json` in `backend/models/`. Place a real labeled dataset in `backend/data/` and run `make train`; do not treat the included sample metrics as real-world performance.

## Academic dataset status

The current `backend/data/sample_dataset.csv` is a 20-row synthetic development dataset only. It must not be used for final academic evaluation.

The project now uses the manually downloaded PHEME Rumour Detection and Veracity Classification dataset. Its official record is <https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078>. The original archive is preserved outside the repository; the copied raw event tree is ignored by Git.

The raw event tree is under `backend/data/raw/pheme/`. Validate, prepare, and analyze it with:

```bash
make validate-pheme
make prepare-dataset
make analyze-dataset
make create-splits
```

The preparation script creates `backend/data/processed/pheme_veracity.csv` with `text`, `label`, `event`, `thread_id`, and `source_tweet_id` columns. It preserves the raw files, excludes non-rumor branches from the veracity task, and does not train a model. See `docs/dataset-card.md` and `docs/exploratory-data-analysis.md` for the actual counts, label policy, bias risks, and split recommendation.

## Reproducible end-to-end workflow

After placing the manually obtained PHEME event tree under
`backend/data/raw/pheme/`, run the isolated workflow:

```bash
make reproduce-experiment-safe
make verify-experiment
```

The runner executes validation, preparation, EDA, event-level splitting,
preprocessing, baseline comparison, grouped tuning, final evaluation, and
error analysis in order. It stops on the first failure and records the complete
run under `backend/data/experiments/run_<UTC_TIMESTAMP>/`. It never accepts the
synthetic sample dataset for the academic workflow. See
`docs/reproducibility-guide.md` for fresh-clone setup, Docker, generated-file
policy, and limitations.

## Baseline model comparison

Run `make train-baselines` after `make prepare-splits`. The experiment fits one TF-IDF vectorizer on training text only, compares a majority baseline, Multinomial Naive Bayes, class-balanced Logistic Regression, and LinearSVC on validation data, and keeps the test split unevaluated for model selection. Results are written to `backend/data/reports/model_comparison/`; model artifacts and the vectorizer are written to the ignored `backend/models/baselines/` directory. See `docs/baseline-model-comparison.md` for the configuration, validation results, and confidence semantics.

## Leakage-safe hyperparameter tuning

Run `make tune-models` after the baseline experiment. Tuning uses only the training split with event-grouped `StratifiedGroupKFold`; each fold fits its own TF-IDF vectorizer. The selected candidates are then evaluated on validation only. The final test split is not loaded or scored. Reports are written to `backend/data/reports/model_comparison/` and selected artifacts to the ignored `backend/models/tuned/` directory. See `docs/hyperparameter-tuning.md` for the search space, results, and limitations.

## Final evaluation

Run `make evaluate-final` only after the candidate configuration is frozen. The command fits the tuned Multinomial Naive Bayes pipeline on train plus validation, evaluates the unseen test events once, and writes final metrics, predictions, confusion matrices, event-wise results, and calibration diagnostics under `backend/data/reports/model_comparison/`. It does not retune or use test results for model selection. See `docs/final-model-evaluation.md`.

## Error analysis and explainability

Run `make analyze-errors` after `make evaluate-final`. This reads the frozen final artifact and saved test predictions without retraining or changing the model. It validates prediction alignment, analyzes confusion pairs and held-out events, groups confidence bands, extracts statistical Naive Bayes feature associations, and writes JSON reports plus charts under `backend/data/reports/model_comparison/error_analysis/`. These associations are language-pattern correlations, not evidence that a claim is true or false. See `docs/error-analysis-and-explainability.md`.

## Leakage-safe data splitting

Run `make create-splits` after preparation. The deterministic script selects five training events, two validation events, and two test events while requiring all three labels in each subset. It keeps events and threads disjoint, writes `backend/data/reports/grouping_report.json`, and writes the ignored manifest to `backend/data/splits/pheme_split_manifest.json`. It does not train a model. See `docs/data-splitting-strategy.md` for the exact allocation and limitations.

## NLP preprocessing

Run `make prepare-splits` after creating the event-level split. The stateless pipeline preserves `text_original`, creates deterministic `text_clean`, and writes separate `train.csv`, `validation.csv`, and `test.csv` files under `backend/data/processed/splits/`. It preserves labels, events, threads, and source tweet IDs, and writes `backend/data/reports/preprocessing_report.json`. It does not fit a vectorizer or train a model. See `docs/text-preprocessing.md` for the transformation choices and leakage controls.

## Exploratory data analysis

Run `make analyze-dataset` to analyze prepared PHEME data when `backend/data/processed/pheme_veracity.csv` exists. If PHEME is unavailable, the command uses only `backend/data/sample_dataset.csv` and labels every output as synthetic development data; the datasets are never combined. The machine-readable report is written to `backend/data/reports/eda_report.json` and charts are written to `backend/data/reports/eda/`. See `docs/exploratory-data-analysis.md` for interpretation and leakage risks. This workflow does not train a model.

## Persistence

SQLite history is stored at `backend/data/history.db` for local runs. Docker stores it in the `history-data` named volume. The database layer is isolated in `backend/app/db/database.py`, so a PostgreSQL repository can be introduced later without changing the API contract.

## Privacy and security

Submitted text is stored locally in SQLite after a successful prediction. The
analyzer shows this notice, application logs omit raw text, and no external
inference API or API key is used. History has no automatic retention job and
should be treated as sensitive local data. Configure explicit production CORS
origins and keep `DEBUG=false`. See `docs/privacy-and-data-handling.md` and
`docs/security-audit.md`.

## Development commands

```bash
make test       # backend unit/API tests
make format     # Black
make lint       # Ruff
make build      # frontend production build
make repo-checks # tracked-file and secret checks
make docker-up
make docker-down
```

GitHub Actions runs the same backend, frontend, and repository quality gates on
pushes and pull requests targeting `main`. See `docs/ci-and-quality-gates.md`.

Additional workflow checks:

```bash
make validate-dataset
make reproduce-experiment
make verify-experiment
make check-environment
```

The backend uses the frozen PHEME artifact in `backend/models/final/` when it
is available. Start it locally with `make backend`, then verify integration:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/model-info
curl -X POST http://localhost:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"A public report describes the latest event and evidence."}'
```

If the generated artifact is absent, the API remains available for health and
documentation endpoints but prediction returns HTTP 503. Recreate it with the
reproducible workflow documented in `docs/backend-model-integration.md`; the
API never trains a replacement model automatically.

Frontend checks:

```bash
cd frontend
npm test
npm run build
```

The frontend includes Overview, Analyze, History, Methodology, and Results
pages. Results are sourced from `frontend/src/data/research-results.json`,
while live predictions and history always come from FastAPI. Set `VITE_API_URL`
when the backend is not running at `http://localhost:8000`. See
`docs/frontend-ml-results.md` for the page/API contract and accessibility
decisions.

The frontend test suite covers the application shell and CTA rendering. The backend suite covers validation, model availability, preprocessing, predictions, endpoints, and SQLite history.

The frontend sends requests to `VITE_API_URL` when provided, otherwise it uses `http://localhost:8000`.

## Limitations and ethical considerations

This tiny illustrative dataset is not suitable for real-world moderation or high-stakes decisions. Language patterns vary by topic, culture, time, and platform; confidence is model certainty, not probability that a claim is true. False positives and false negatives are expected. Do not use this output to censor people, target protected groups, make employment/education/health decisions, or label individuals. For important claims, inspect primary sources and consult qualified experts. Production use needs representative, licensed data, bias and calibration evaluation, human review, monitoring, privacy controls, and a clear appeal process. Hugging Face Transformers can be added later behind the same service interface after careful benchmarking and governance.
