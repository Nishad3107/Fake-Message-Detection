# Final Release Checklist

Use this checklist before academic submission. Generated datasets, model
artifacts, reports, SQLite history, and experiment outputs remain local and are
not automatically added to Git.

## Academic dataset and ML workflow

- [x] PHEME source and usage terms documented.
- [x] Raw dataset validation completed.
- [x] Processed dataset preparation documented.
- [x] Exploratory data analysis completed.
- [x] Event-level train/validation/test split validated.
- [x] Stateless text preprocessing documented.
- [x] Baseline model comparison completed.
- [x] Grouped hyperparameter tuning completed.
- [x] Frozen final model evaluated once on unseen events.
- [x] Error analysis and explainability completed.
- [x] Safe versioned experiment reproduction documented.

## Application

- [x] FastAPI health, model-info, prediction, history, and metrics endpoints.
- [x] Frozen PHEME model loading and missing-artifact handling.
- [x] Input validation and safe error responses.
- [x] React landing, analyzer, dashboard, methodology, and results pages.
- [x] SQLite history persistence.
- [x] Docker Compose configuration and health checks.
- [x] GitHub Actions CI workflow.
- [x] Security and privacy documentation.

## Quality and security

- [x] Backend tests pass.
- [x] Frontend tests pass.
- [x] Frontend production build passes.
- [x] Ruff passes.
- [x] Black passes.
- [x] Python syntax checks pass.
- [x] Repository secret/artifact checks pass.
- [x] `git diff --check` passes.
- [x] Frozen model and research metrics remain unchanged.
- [ ] Manual browser review completed on the target screen sizes.
- [ ] Docker runtime verification completed with Docker Desktop available.

## Submission materials

- [x] Final academic report created.
- [x] Viva preparation created.
- [x] Presentation outline created.
- [ ] Final screenshots reviewed and captured.
- [ ] Final PDF/export formatting reviewed.
- [ ] Mentor or department formatting requirements confirmed.
- [ ] Final Git diff reviewed.
- [ ] Git commit created.
- [ ] GitHub push completed.

The project is ready for academic submission after the student reviews the final report, presentation, screenshots, and local Git diff.

