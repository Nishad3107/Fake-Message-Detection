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
cd /Users/nishadtembhurnekar/Desktop/Fake-Message-Detection
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

Open http://localhost:5173. The app includes an overview landing page, analyzer, SQLite-backed dashboard, and methodology page. The backend automatically trains and persists a model under `backend/models/` when artifacts are absent or incompatible.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

Frontend: http://localhost:5173 · API: http://localhost:8000 · Swagger UI: http://localhost:8000/docs

## API

`GET /health` reports service and model readiness.

`GET /api/v1/model-info` returns the baseline name, version, algorithm, dataset name, readiness, and evaluation metrics.

`GET /api/v1/history` returns recent SQLite-backed analyses and summary counts. `GET /api/v1/metrics` exposes saved evaluation metrics.

`POST /api/v1/predict`

```json
{"text":"Paste a social media post with at least ten characters."}
```

The response contains `classification` (`misinformation` or `likely_true`), a 0–1 `confidence`, detected heuristic keywords, a short model explanation, and a disclaimer. The model does not browse, retrieve sources, or establish truth.

`POST /api/v1/predict/batch` accepts `{"posts":[{"text":"..."}]}` for up to 50 posts. Every successful prediction is saved to local history.

Requests must contain 10–5000 non-whitespace characters. Invalid input returns HTTP 422. If model artifacts cannot be trained or loaded, prediction returns HTTP 503 with a clear error message.

## Training

The sample CSV at `backend/data/sample_dataset.csv` requires `text` and `label` columns. To retrain:

```bash
make train
```

The script cleans text, performs a fixed-seed 70/30 stratified holdout, fits TF-IDF and Logistic Regression with class balancing, reports accuracy/precision/recall/F1 plus a confusion matrix and sample counts, and saves `classifier.joblib`, `vectorizer.joblib`, and `metrics.json` in `backend/models/`. Place a real labeled dataset in `backend/data/` and run `make train`; do not treat the included sample metrics as real-world performance.

## Academic dataset status

The current `backend/data/sample_dataset.csv` is a 20-row synthetic development dataset only. It must not be used for final academic evaluation.

The preferred real-data direction is the PHEME Rumour Detection and Veracity Classification dataset. Its official record is <https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078>. The current environment can reach the record page, but the official archive download returned HTTP 403, so no PHEME data has been added automatically. Do not substitute LIAR or another dataset without documenting that decision.

After manually downloading and extracting PHEME, place the untouched event tree under `backend/data/raw/pheme/`, then run:

```bash
make prepare-dataset
cat backend/data/reports/pheme_quality.json
```

The preparation script creates `backend/data/processed/pheme_veracity.csv` with `text`, `label`, `source_event`, and `source_tweet_id` columns, and writes a quality report. It preserves the raw files, skips non-rumor branches for the preferred `true`/`false`/`unverified` veracity task, and does not train a model. See `docs/dataset-card.md` for the license, label policy, expected raw tree, bias risks, and split recommendation.

## Persistence

SQLite history is stored at `backend/data/history.db` for local runs. Docker stores it in the `history-data` named volume. The database layer is isolated in `backend/app/db/database.py`, so a PostgreSQL repository can be introduced later without changing the API contract.

## Development commands

```bash
make test       # backend unit/API tests
make format     # Black
make lint       # Ruff
make docker-up
make docker-down
```

Frontend checks:

```bash
cd frontend
npm test
npm run build
```

The frontend test suite covers the application shell and CTA rendering. The backend suite covers validation, model availability, preprocessing, predictions, endpoints, and SQLite history.

The frontend sends requests to `VITE_API_URL` when provided, otherwise it uses `http://localhost:8000`.

## Limitations and ethical considerations

This tiny illustrative dataset is not suitable for real-world moderation or high-stakes decisions. Language patterns vary by topic, culture, time, and platform; confidence is model certainty, not probability that a claim is true. False positives and false negatives are expected. Do not use this output to censor people, target protected groups, make employment/education/health decisions, or label individuals. For important claims, inspect primary sources and consult qualified experts. Production use needs representative, licensed data, bias and calibration evaluation, human review, monitoring, privacy controls, and a clear appeal process. Hugging Face Transformers can be added later behind the same service interface after careful benchmarking and governance.
