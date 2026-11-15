# Frontend ML Results

The React frontend presents the frozen PHEME workflow without inventing
predictions or research metrics. It uses the FastAPI API at `VITE_API_URL`
(default `http://localhost:8000`) for health, model information, prediction,
history, and metrics requests.

## Pages

- **Overview** introduces the academic question, PHEME dataset, event-level
  evaluation, explainable NLP, and ethical boundaries.
- **Analyze** sends messages to `POST /api/v1/predict` and displays the PHEME
  label, class probabilities, confidence category, feature associations,
  warnings, model version, and timestamp.
- **History** reads `GET /api/v1/history`, displays only stored backend
  analyses, and supports refresh, empty/error states, and detail inspection.
- **Methodology** explains the dataset, event splitting, preprocessing,
  TF-IDF, model selection, final evaluation, and limitations.
- **Results** uses one versioned `research-results.json` source for baseline,
  validation, grouped-CV, and final test results.

## Interpretation rules

The interface uses neutral labels: `Model prediction: true`, `false`, or
`unverified`. It does not present `true` as verified fact. Probabilities are
uncalibrated model outputs, and feature associations are statistical patterns,
not evidence. Low-confidence, high-confidence, and unverified warnings remain
visible with the result.

## Accessibility and resilience

Inputs have explicit labels, navigation uses buttons with visible focus styles,
results use live regions, detail inspection uses dialog semantics, and reduced-
motion CSS disables decorative animation when requested. Loading, empty, HTTP
error, unavailable-model, and network-failure states are handled without
showing a success result.

## Local startup

```bash
cp .env.example .env
make install
make backend
# in another terminal
make frontend
```

Set `VITE_API_URL` when the backend is not running at `http://localhost:8000`.
The backend must have the generated frozen artifact available; otherwise
predictions correctly show the model-unavailable error.
