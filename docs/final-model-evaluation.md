# Final Model Evaluation

## Frozen candidate

The final candidate was frozen before test evaluation:

- Model: tuned Multinomial Naive Bayes
- TF-IDF word n-grams: `(1, 2)`
- `min_df`: `3`
- `sublinear_tf`: `false`
- `alpha`: `0.1`
- `fit_prior`: `false`
- Random seed: `42`

The frozen configuration is stored in `backend/data/reports/model_comparison/final_model_config.json`. It records the preprocessing options, split manifest reference, library versions, and confirms that test metrics were not used for selection.

## Fitting and leakage control

The final pipeline was fitted on the 1,507 training records plus 467 validation records, for 1,974 fitting records total. The test split contains 297 records from two unseen events and was not used for fitting, vocabulary learning, model selection, or threshold selection.

The pipeline applies the recorded stateless preprocessing to `text_original`, fits TF-IDF on the combined train/validation text only, and fits Multinomial Naive Bayes on the resulting features. The artifact is saved at `backend/models/final/final_pipeline.joblib`.

## Final test results

| Metric | Test result |
|---|---:|
| Accuracy | 0.225589 |
| Macro precision | 0.228806 |
| Macro recall | 0.267193 |
| Macro F1 | 0.211379 |
| Weighted precision | 0.278353 |
| Weighted recall | 0.225589 |
| Weighted F1 | 0.232157 |
| Log loss | 1.694920 |
| One-vs-rest ROC-AUC macro | 0.417280 |
| One-vs-rest PR-AUC macro | 0.295843 |
| Multiclass Brier score | 0.314570 |

All three labels are present in the overall test set, so the multiclass metrics are defined. The full classification report and confusion matrix are stored in the model-comparison report directory.

## Per-class performance

The exact per-class values are stored in `final_test_results.json` and `final_classification_report.json`. The report uses label order `false`, `true`, `unverified` and includes precision, recall, F1, and support for every class. Errors are substantial across classes, so the model should not be described as a reliable fact checker.

## Event-wise performance

| Event | Records | Labels present | Accuracy | Macro F1 |
|---|---:|---|---:|---:|
| `germanwings-crash` | 238 | all three | 0.252101 | 0.239687 |
| `gurlitt` | 59 | `true`, `unverified` | 0.118644 | unavailable |

Gurlitt does not contain any `false` test records, so a full three-class macro F1 is marked unavailable rather than fabricated. Event-specific topic and label composition explain why event performance differs from the aggregate result.

## Confidence and calibration

Multinomial Naive Bayes supplies class probability outputs. On the test set:

- Confidence minimum: 0.362465
- Confidence maximum: 0.983769
- Confidence mean: 0.593928
- Confidence median: 0.569365
- Low-confidence threshold: 0.6
- Low-confidence predictions: 181 of 297
- Expected calibration error: 0.378936

The probabilities are model outputs, not probabilities that a claim is factually true. The high calibration error and low test accuracy show that these outputs should not be treated as verified confidence. The reliability diagram is descriptive for this one held-out split and does not establish calibrated probabilities.

## Validation versus test interpretation

| Evaluation source | Macro F1 |
|---|---:|
| Majority baseline validation | 0.194949 |
| Original Logistic Regression validation | 0.343505 |
| Tuned Naive Bayes grouped CV mean ± std | 0.197411 ± 0.113824 |
| Tuned Naive Bayes validation | 0.372437 |
| Frozen final test | 0.211379 |

Validation performance substantially overestimated the final held-out event result. The test macro F1 is only modestly above the majority validation baseline and below the tuned validation score. The grouped CV standard deviation and event-wise results provide evidence of event-level instability.

## Academic interpretation and limitations

The model is suitable as an academic prototype demonstrating a reproducible NLP pipeline, event-level split, TF-IDF features, and classical probabilistic classification. It is not suitable as an autonomous fact checker, moderation authority, or high-stakes decision system. It learns language patterns associated with historical PHEME annotations and does not verify claims against evidence.

Future work should use more events or grouped cross-validation, investigate class and event shift, evaluate calibration on independent data, compare feature representations, inspect errors manually, and only then consider a carefully governed deployment. No parameter tuning should be performed using this test result.

## Reproduce the one-time evaluation

```bash
make evaluate-final
```

The command refuses a changed frozen configuration and writes the final reports without fitting on test rows.
