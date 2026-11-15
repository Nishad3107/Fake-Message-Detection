# CI and Quality Gates

## Triggers

`.github/workflows/ci.yml` runs on pushes to `main` and pull requests targeting
`main`. It has backend, frontend, and repository-policy jobs.

## Backend checks

The backend job uses Python 3.11, installs `backend/requirements.txt`, runs
syntax compilation, Ruff, Black check, and pytest. With `CI=true`,
`backend/tests/conftest.py` creates a temporary three-class model fixture
outside the repository. This verifies API contracts without requiring the
ignored final PHEME artifact. It is not a research model.

The real PHEME dataset, final artifact, final reports, and full ML reproduction
are excluded from CI because the dataset is manually supplied and generated
artifacts are ignored.

## Frontend checks

The frontend job uses Node.js 22, installs exactly from `package-lock.json` with
`npm ci`, runs Vitest, and runs `npm run build`, including TypeScript checking
and the Vite production build. Tests do not require a live backend.

## Repository policy checks

The repository job runs `git diff --check` and
`tools/ci_repository_checks.sh`. It rejects tracked `.env` files, raw PHEME
files, model artifacts, SQLite databases, generated experiment directories,
user-specific absolute paths, and common credential formats. It only inspects
tracked files inside the repository.

## Local equivalents

```bash
make test
make lint
make format
make build
make repo-checks
git diff --check
```

Docker runtime verification remains artifact-aware and is run locally after the
frozen model exists:

```bash
docker compose up --build -d
tools/verify_stack.sh
```

The complete PHEME workflow remains a local/manual operation:
`make reproduce-experiment-safe`.

## Troubleshooting and merge requirements

- Backend fixture failures indicate an API contract or dependency issue.
- Frontend failures indicate a lockfile, TypeScript, or build issue.
- Repository failures indicate tracked private, generated, or credential-like content.
- Docker failures require Docker Desktop and the ignored final artifact locally.

All three CI jobs must pass before merging. No repository secret or API key is
required by this workflow.
