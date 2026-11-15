# Docker and End-to-End Verification

## Prerequisites

- Docker Desktop with the daemon running.
- `.env` copied from `.env.example`.
- The ignored artifact `backend/models/final/final_pipeline.joblib`.
- Matching final reports under `backend/data/reports/model_comparison/`.

Docker never trains or downloads the model. On a fresh clone, provide PHEME
and run `make reproduce-experiment-safe` first.

## Start

```bash
cp .env.example .env
docker compose config
docker compose up --build -d
docker compose ps
```

The frontend is at `http://localhost:5173`; the backend is at
`http://localhost:8000`. Models and final reports are mounted read-only.
SQLite history persists in the named `history-data` volume at
`/app/runtime/history.db`.

## Verify

Use `tools/verify_stack.sh` after startup. It checks backend health and model
readiness, a real prediction, probability normalization, history response
shape, and frontend availability. It exits non-zero on failure.

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/model-info
curl -X POST http://localhost:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"A public report describes the latest event and evidence."}'
curl http://localhost:8000/api/v1/history
tools/verify_stack.sh
```

Missing/corrupt models produce degraded health and `503` predictions. Invalid
input produces `422`. The frontend displays errors without fake success.
Set `VITE_API_URL` at frontend build time when the backend is not exposed at
`http://localhost:8000`.

## Local versus Docker

Local development uses `make backend` and `make frontend`; Docker serves the
compiled frontend with Nginx and runs FastAPI in Python. Both use the same
frozen artifact and API contracts. Raw PHEME data is not copied into images.

Stop without deleting history:

```bash
docker compose stop
docker compose down
```

Do not pass `--volumes` when history must be retained. Regenerate research
artifacts with the safe workflow, never during Docker startup.
