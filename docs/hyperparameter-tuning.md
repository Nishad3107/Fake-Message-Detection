# Leakage-Safe Hyperparameter Tuning

## Strategy

Tuning uses only `backend/data/processed/splits/train.csv`. The five training events are divided with `StratifiedGroupKFold` using `event` as the group and `label` as the target. Ordinary random cross-validation is unsafe for PHEME because event-specific vocabulary and context could appear in both fold partitions.

For every fold and configuration, a new TF-IDF vectorizer is fitted only on that fold's training text. The fold validation text is transformed with that fitted vectorizer. After selecting one configuration per model family by mean cross-validation macro F1, the configuration is fitted on the complete training split and evaluated once on the existing validation split. The final test split is not loaded, transformed, or scored.

## Search space

The experiment tested 144 configurations total: 48 per model family.

TF-IDF variations:

- Word n-grams `(1, 1)` and `(1, 2)`
- `min_df`: 1, 2, and 3
- `sublinear_tf`: enabled and disabled
- `max_df=0.95`, `lowercase=False`, and `strip_accents="unicode"`

Model parameters:

- Logistic Regression: `C` in `{0.5, 1.0}`, `class_weight` in `{None, "balanced"}`, `max_iter=1000`
- Multinomial Naive Bayes: `alpha` in `{0.1, 1.0}`, `fit_prior` in `{True, False}`
- LinearSVC: `C` in `{0.5, 1.0}`, `class_weight` in `{None, "balanced"}`, `max_iter=2000`

The fixed seed is 42. Macro F1 is the primary metric; weighted F1 and accuracy are secondary diagnostics.

## Selected configurations and results

| Family | Best CV configuration | CV macro F1 mean ± std | Validation macro F1 | Baseline validation macro F1 |
|---|---|---:|---:|---:|
| Logistic Regression | `(1,2)`, `min_df=3`, sublinear off, `C=1.0`, balanced | 0.180518 ± 0.124920 | 0.330556 | 0.343505 |
| Multinomial Naive Bayes | `(1,2)`, `min_df=3`, sublinear off, `alpha=0.1`, `fit_prior=false` | **0.197411 ± 0.113824** | **0.372437** | 0.250895 |
| LinearSVC | `(1,2)`, `min_df=2`, sublinear off, `C=1.0`, balanced | 0.171292 ± 0.145954 | 0.293472 | 0.292217 |

The tuned Naive Bayes candidate is the provisional candidate for final test evaluation because it has the highest selected-candidate CV mean and validation macro F1. Tuning improved Naive Bayes substantially and produced only a small improvement for LinearSVC. It reduced the validation macro F1 of Logistic Regression relative to its original baseline configuration. This does not establish that Naive Bayes is the final best model until the held-out test phase is performed.

The CV standard deviations are large relative to the means. Several folds hold out single events with distinctive label distributions, so the results show meaningful event-shift instability and a risk of overfitting the small set of five training events.

## Runtime and artifacts

The run used Python 3.11.14, scikit-learn 1.9.1, and macOS arm64. Selected-candidate grouped CV durations were approximately 0.165223 seconds for Logistic Regression, 0.102776 seconds for Naive Bayes, and 0.128890 seconds for LinearSVC. Complete fold-level timing is stored in `backend/data/reports/model_comparison/tuning_results.json`.

Reports are written to:

- `backend/data/reports/model_comparison/tuning_results.json`
- `backend/data/reports/model_comparison/tuning_results.csv`
- `backend/data/reports/model_comparison/tuning_summary.json`

Selected model and vectorizer artifacts are written to the ignored `backend/models/tuned/` directory.

## Limitations and academic interpretation

This is grouped cross-validation over only five training events, followed by one validation evaluation. It is not final generalization evidence. The test set must remain untouched until the model family, TF-IDF settings, preprocessing, and any confidence policy are frozen. A final report should include grouped cross-validation variability, per-class metrics, confusion matrices, and one pre-registered test evaluation. LinearSVC margins remain uncalibrated decision scores rather than probabilities.

## Reproduce

```bash
make tune-models
```
