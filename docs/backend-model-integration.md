# Backend Model Integration

The FastAPI application now uses the frozen PHEME artifact at
`backend/models/final/final_pipeline.joblib`. The artifact is loaded once in
the application lifespan by `FrozenModelService`; requests never retrain or
reload it. Missing or corrupt artifacts leave the service degraded and return
HTTP 503 for prediction requests.

## Inference flow

1. `PredictionRequest` rejects non-string, empty, too-short, and over-5,000
   character input.
2. The service reads the preprocessing configuration stored with the artifact.
3. Stateless preprocessing produces the same `text_clean` representation used
   by final evaluation.
4. The stored TF-IDF vectorizer transforms the cleaned text. Its vocabulary is
   never fitted by the API.
5. Multinomial Naive Bayes returns probabilities for the original PHEME labels:
   `false`, `true`, and `unverified`.
6. The response includes the winning label, all class probabilities, a
   confidence category, statistical feature associations, model metadata, and
   an uncertainty warning.

The confidence categories are configurable with `LOW_CONFIDENCE_THRESHOLD`
(default `0.6`) and `HIGH_CONFIDENCE_THRESHOLD` (default `0.8`). They describe
model output thresholds only; they are not calibrated probabilities of factual
truth. Every response states that it is not factual verification. High
confidence adds an overconfidence warning, low confidence adds an uncertainty
warning, and `unverified` adds a source-verification warning.

## Endpoints

- `GET /health` reports API and artifact readiness.
- `GET /api/v1/model-info` reports the frozen model, PHEME references, TF-IDF
  configuration, final test macro F1, and limitations.
- `POST /api/v1/predict` analyzes one post.
- `POST /api/v1/predict/batch` analyzes at most 50 independently validated
  posts.
- `GET /api/v1/history` returns SQLite-backed analyses.
- `GET /api/v1/metrics` returns saved final evaluation metrics.

History stores the original and cleaned text, label, probabilities, confidence
category, feature terms, explanation summary, model version, and timestamp. It
does not store unnecessary request metadata.

## Reproducibility and replacement

The artifact is ignored by Git because it is generated. To reproduce it, keep
the real PHEME data and generated splits available, run the documented grouped
tuning workflow, freeze the selected configuration, and run:

```bash
make evaluate-final
```

The current integration must not be used to silently replace the frozen model.
Any future model replacement should update the artifact metadata, evaluation
reports, API model information, and academic documentation together.
