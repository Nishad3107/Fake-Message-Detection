.PHONY: install backend frontend test train train-baselines tune-models validate-pheme prepare-dataset analyze-dataset create-splits prepare-splits format lint docker-up docker-down

PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

install:
	@test -x .venv/bin/python || python3 -m venv .venv
	.venv/bin/python -m pip install -r backend/requirements.txt
	cd frontend && npm install
backend:
	PYTHONPATH=backend $(PYTHON) -m uvicorn app.main:app --reload --reload-dir backend --host 127.0.0.1 --port 8000
frontend:
	cd frontend && npm run dev
test:
	PYTHONPATH=backend $(PYTHON) -m pytest -q backend/tests
train:
	PYTHONPATH=backend $(PYTHON) -m app.ml.train --dataset backend/data/sample_dataset.csv --model-dir backend/models
train-baselines:
	PYTHONPATH=backend $(PYTHON) -m app.ml.train_baselines
tune-models:
	PYTHONPATH=backend $(PYTHON) -m app.ml.tune_models --manifest backend/data/splits/pheme_split_manifest.json
validate-pheme:
	PYTHONPATH=backend $(PYTHON) -m app.ml.validate_pheme --raw-dir backend/data/raw/pheme --report backend/data/reports/pheme_validation.json
prepare-dataset:
	PYTHONPATH=backend $(PYTHON) -m app.ml.prepare_dataset --raw-dir backend/data/raw/pheme --output backend/data/processed/pheme_veracity.csv --report backend/data/reports/pheme_quality.json
analyze-dataset:
	PYTHONPATH=backend $(PYTHON) -m app.ml.analyze_dataset
create-splits:
	PYTHONPATH=backend $(PYTHON) -m app.ml.create_splits
prepare-splits:
	PYTHONPATH=backend $(PYTHON) -m app.ml.prepare_splits
format:
	$(PYTHON) -m black backend/app backend/tests
lint:
	$(PYTHON) -m ruff check backend/app backend/tests
docker-up:
	docker compose up --build
docker-down:
	docker compose down
