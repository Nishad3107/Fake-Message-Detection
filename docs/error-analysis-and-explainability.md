# Error Analysis and Explainability

This analysis uses the frozen tuned Multinomial Naive Bayes artifact and saved
predictions for the held-out PHEME test events. It does not retrain, retune,
calibrate, or modify the final model.

## Reproducible command

```bash
make analyze-errors
```

The command validates that test rows, original and cleaned text, labels,
events, threads, and source tweet IDs align with `final_predictions.csv`.
Reports are written below `backend/data/reports/model_comparison/` and charts
below its `error_analysis/` directory.

## Findings

The test set contains 297 predictions. The largest confusion pairs are
`true → unverified` (65), `false → true` (51), `false → unverified` (48), and
`true → false` (46). The hardest class by F1 is `false` (0.136364), followed by
`unverified` (0.184049). False recall is low because many false-labelled posts
are assigned to true or unverified. Unverified precision is low because the
model assigns that label to many posts whose gold label is true or false.

Performance is event-sensitive. Germanwings has accuracy 0.252101 and macro F1
0.239687. Gurlitt has accuracy 0.118644; macro F1 is unavailable because that
event's test rows do not contain all three labels. Topic, names, phrasing, and
label availability differ between held-out events, creating distribution shift.

## Confidence and feature associations

The saved probabilities are Multinomial Naive Bayes outputs, not calibrated
probabilities of factual truth. There are 25 high-confidence wrong predictions
(at least 0.80) and 38 correct predictions below 0.60 confidence. The product
should therefore show an uncertainty warning and never present the score as a
verified fact-check.

Feature explanations use class log-probability differences. Strong associations
include topic- and event-specific terms such as `prince`, `massey`, `cirillo`,
and `nathan`. These are learned language associations, not causal explanations
or proof of truth status. Event-specific vocabulary is a likely contributor to
poor performance on unseen events.

## Limitations and future work

The test set is small and contains only two held-out events. Event metrics can
be unstable, and some per-event metrics are undefined when a label is absent.
Calibration is also limited by sample size and the lack of a separate
calibration set. Future work can evaluate grouped cross-validation, richer
features, probability calibration using training-only validation data, and
transformers only after careful leakage controls. Human review and source
verification remain necessary for consequential claims.

The low test score is a valuable academic finding: a strong-looking validation
result does not guarantee generalization to unseen events. No model change was
made during this analysis.
