# Baseline NLP Model Comparison

## Experiment scope

This experiment compares classical models on the real PHEME processed split files. The event-level split manifest is reused exactly; no random row split was created. The test text was transformed only to verify feature shape and was not predicted or scored.

Dataset sizes:

- Train: 1,507 records
- Validation: 467 records
- Test: 297 records, held out

The validation set contains 193 `true`, 125 `false`, and 149 `unverified` records. Macro metrics are the primary comparison because the class counts are not equal.

## Shared TF-IDF pipeline

One `TfidfVectorizer` was fitted only on `train.csv` `text_clean`, then used to transform validation and test text. The configuration was:

```text
word n-grams: (1, 2)
min_df: 2
max_df: 0.95
sublinear_tf: true
lowercase: false  # text_clean is already deterministically case-folded
strip_accents: unicode
features: 4,243
vocabulary size: 4,243
```

The vectorizer was never fitted on validation or test text. This is feature learning, distinct from stateless text cleaning.

## Models

- **Majority baseline:** `DummyClassifier(strategy="most_frequent")`; establishes the score from always selecting the most common training label.
- **Multinomial Naive Bayes:** `alpha=1.0`; a natural sparse count/TF-IDF text baseline based on class-conditional feature likelihoods.
- **Logistic Regression:** class-balanced, `max_iter=1000`, `solver="lbfgs"`, random state 42; provides multiclass probabilities and interpretable linear feature weights.
- **LinearSVC:** class-balanced, `max_iter=2000`, random state 42; a strong sparse text baseline, but its decision margins are not probabilities.

## Validation results

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Training seconds | Features | Confidence output |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Majority baseline | 0.413276 | 0.137759 | 0.333333 | 0.194949 | 0.241704 | 0.000372 | 4,243 | probabilities from dummy model |
| Multinomial Naive Bayes | 0.428266 | 0.531313 | 0.353691 | 0.250895 | 0.292193 | 0.001525 | 4,243 | probabilities |
| Logistic Regression | 0.415418 | 0.359082 | 0.368535 | **0.343505** | **0.374263** | 0.021774 | 4,243 | probabilities |
| LinearSVC | 0.346895 | 0.295512 | 0.311603 | 0.292217 | 0.316932 | 0.006778 | 4,243 | decision margins |

The selected validation comparison is Logistic Regression by macro F1. It is clearly better than the majority baseline on macro F1, while its accuracy is only slightly higher. Naive Bayes has the highest validation accuracy in this run but lower macro F1, illustrating why accuracy alone is insufficient. These are validation results from one fixed event-level allocation, not final generalization results.

The baseline has imperfect label balance rather than severe imbalance. The majority baseline predicts only the most common class and has zero F1 for the other two classes. Class weighting was used for Logistic Regression and LinearSVC; the effect should be studied in later controlled experiments.

## Confidence semantics

DummyClassifier, MultinomialNB, and Logistic Regression expose class probabilities. Their report records the mean maximum predicted probability, but these probabilities are model outputs and are not factual truth probabilities. LinearSVC exposes decision margins only. The runner labels them `decision_margins` and explicitly does not call them probabilities; calibration would be a separate later experiment.

## Runtime and reproducibility

The run used Python 3.11.14, scikit-learn 1.9.1, and macOS arm64. Training durations are machine-dependent and are recorded in `backend/data/reports/model_comparison/baseline_results.json`. The fixed experiment seed is 42, although the shared vectorizer and split assignments are the primary reproducibility controls.

## Limitations and next steps

The comparison uses one fixed validation allocation across only nine historical events. It does not establish future performance, factual correctness, or production readiness. The test split must remain untouched until the model configuration and preprocessing choices are finalized. A later academic phase should compare grouped cross-validation or leave-one-event-out results, inspect per-class errors, evaluate calibration, and then perform one final test evaluation.
