.PHONY: install backend frontend test train format lint docker-up docker-down
install:
	python -m pip install -r backend/requirements.txt
	cd frontend && npm install
backend:
	cd backend && uvicorn app.main:app --reload
frontend:
	cd frontend && npm run dev
test:
	cd backend && pytest -q
train:
	cd backend && python -m app.ml.train
format:
	black backend/app backend/tests
lint:
	ruff check backend/app backend/tests
docker-up:
	docker compose up --build
docker-down:
	docker compose down
