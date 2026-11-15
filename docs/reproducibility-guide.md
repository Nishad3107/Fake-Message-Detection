# Reproducibility Guide

This repository separates committed source code from the manually supplied
PHEME dataset and generated research artifacts. A fresh clone can reproduce
the academic workflow after the dataset is obtained under its official terms.

## Fresh clone setup

```bash
git clone https://github.com/Nishad3107/Fake-Message-Detection.git
cd Fake-Message-Detection
cp .env.example .env
make install
make check-environment
```

Download the PHEME Rumour Detection and Veracity Classification archive from
the official Figshare record documented in `docs/dataset-card.md`. Extract the
actual event directories under:

```text
backend/data/raw/pheme/
```

The raw tree should contain event folders with PHEME rumor threads. Do not
combine it with `backend/data/sample_dataset.csv`; that 20-row file is only a
development fallback.

## Reproduce the ML workflow

Run the complete staged experiment:

```bash
make reproduce-experiment
```

The order is:

1. Validate the raw PHEME tree.
2. Prepare `backend/data/processed/pheme_veracity.csv`.
3. Run real-data EDA.
4. Create the deterministic event-level split.
5. Prepare cleaned train, validation, and test files.
6. Train baseline models on training data and compare on validation data.
7. Tune candidates with event-grouped cross-validation.
8. Fit and evaluate the frozen candidate once on the untouched test events.
9. Generate error-analysis and explainability reports.

Every stage stops on failure. The runner records timestamps, durations,
package versions, split sizes, report references, and the current Git commit in
`backend/data/reports/experiment_summary.json`.

Individual commands are available when investigating a stage:

```bash
make validate-dataset
make prepare-dataset
make analyze-dataset
make create-splits
make prepare-splits
make train-baselines
make tune-models
make evaluate-final
make analyze-errors
```

`make verify-experiment` checks the generated outputs, label consistency,
event/thread separation, frozen test-selection policy, final artifact path,
and frontend research metrics. It writes a diagnostic report to
`backend/data/reports/experiment_validation.json` and fails clearly when an
input or output is missing.

## Application workflow

Start the backend and frontend locally in separate terminals:

```bash
make backend
make frontend
```

The frontend uses `VITE_API_URL`, defaulting to `http://localhost:8000`.
Prediction requests use the frozen artifact; history is stored in SQLite.

Verify the backend directly:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/model-info
curl -X POST http://localhost:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"A public report describes the latest event and evidence."}'
```

## Docker workflow

```bash
cp .env.example .env
make docker-up
```

Compose builds the frontend and backend, exposes ports 5173 and 8000, and
checks both services. The backend bind-mounts `backend/models` read-only. A
fresh clone therefore needs the generated final artifact before prediction
requests can be ready; Docker does not download data or train during startup.
SQLite history is stored in the `history-data` named volume.

## What is included in Git and what must be regenerated

| Item | Git policy |
| --- | --- |
| Python, FastAPI, React, Docker, and Make source | Included |
| Academic documentation and tests | Included |
| `.env.example` | Included; copy to `.env` locally |
| Raw PHEME dataset | Not included; licensing and size require manual provision |
| Processed PHEME CSV and split CSVs | Generated and ignored |
| Baseline/tuned/final model artifacts | Generated and ignored |
| EDA, training, tuning, evaluation, and error reports | Generated and ignored |
| SQLite history database | Local/generated and ignored |
| `frontend/node_modules` and build output | Installed/generated and ignored |
| `frontend/src/data/research-results.json` | Included static research-results snapshot, checked against generated final metrics |

## Reproducibility limitations

Results can vary slightly across Python, scikit-learn, BLAS, operating-system,
and hardware versions. The workflow records versions and uses seed `42`, but
the test set must remain locked and must never be used for tuning. The PHEME
annotations, event distribution, and small number of held-out events limit
generalization. The model is an academic prototype, not a factual verifier.

## Verification and tests

```bash
make verify-experiment
make test
cd frontend && npm test -- --run && npm run build
make lint
make format
git diff --check
```
