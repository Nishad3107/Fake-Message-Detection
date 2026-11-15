.PHONY: install backend frontend test build train train-baselines tune-models evaluate-final analyze-errors validate-pheme validate-dataset prepare-dataset analyze-dataset create-splits prepare-splits reproduce-experiment reproduce-experiment-safe verify-experiment check-environment format lint repo-checks docker-up docker-down

PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

install:
	@test -x .venv/bin/python || python3 -m venv .venv
	$(PYTHON) -m pip install -r backend/requirements.txt
	cd frontend && npm install
backend:
	PYTHONPATH=backend $(PYTHON) -m uvicorn app.main:app --reload --reload-dir backend --host 127.0.0.1 --port 8000
frontend:
	cd frontend && npm run dev
test:
	PYTHONPATH=backend $(PYTHON) -m pytest -q backend/tests
build:
	cd frontend && npm run build
train:
	PYTHONPATH=backend $(PYTHON) -m app.ml.train --dataset backend/data/sample_dataset.csv --model-dir backend/models
train-baselines:
	PYTHONPATH=backend $(PYTHON) -m app.ml.train_baselines
tune-models:
	PYTHONPATH=backend $(PYTHON) -m app.ml.tune_models --manifest backend/data/splits/pheme_split_manifest.json
evaluate-final:
	PYTHONPATH=backend $(PYTHON) -m app.ml.evaluate_final_model
analyze-errors:
	PYTHONPATH=backend $(PYTHON) -m app.ml.error_analysis
validate-pheme:
	PYTHONPATH=backend $(PYTHON) -m app.ml.validate_pheme --raw-dir backend/data/raw/pheme --report backend/data/reports/pheme_validation.json
validate-dataset: validate-pheme
prepare-dataset:
	PYTHONPATH=backend $(PYTHON) -m app.ml.prepare_dataset --raw-dir backend/data/raw/pheme --output backend/data/processed/pheme_veracity.csv --report backend/data/reports/pheme_quality.json
analyze-dataset:
	PYTHONPATH=backend $(PYTHON) -m app.ml.analyze_dataset
create-splits:
	PYTHONPATH=backend $(PYTHON) -m app.ml.create_splits
prepare-splits:
	PYTHONPATH=backend $(PYTHON) -m app.ml.prepare_splits
reproduce-experiment:
	PYTHONPATH=backend $(PYTHON) -m app.ml.run_experiment
reproduce-experiment-safe:
	@test -d backend/data/raw/pheme || (echo "PHEME dataset missing at backend/data/raw/pheme" && exit 1)
	PYTHONPATH=backend $(PYTHON) -m app.ml.check_environment --strict
	PYTHONPATH=backend $(PYTHON) -m app.ml.run_experiment --output-dir backend/data/experiments/run_$$(date -u +%Y%m%dT%H%M%SZ)
verify-experiment:
	PYTHONPATH=backend $(PYTHON) -m app.ml.validate_experiment
check-environment:
	PYTHONPATH=backend $(PYTHON) -m app.ml.check_environment
format:
	$(PYTHON) -m black backend/app backend/tests
lint:
	$(PYTHON) -m ruff check backend/app backend/tests
repo-checks:
	./tools/ci_repository_checks.sh
docker-up:
	@test -f .env || cp .env.example .env
	docker compose up --build
docker-down:
	docker compose down
