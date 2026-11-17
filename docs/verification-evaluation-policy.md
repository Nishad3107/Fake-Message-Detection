# Verification Evaluation Policy

Each task is evaluated separately with a task-specific, leakage-safe test set.
Accuracy is never reported without naming the task, data split, and evidence
protocol.

## Metrics by task

| Task | Primary metrics | Secondary metrics |
|---|---|---|
| Pattern classifier | Macro F1 | Per-class precision/recall, balanced accuracy, calibration error, log loss |
| Evidence retrieval | Recall@k | Precision@k, MRR, nDCG |
| Entailment | Macro F1 | Per-class recall, confusion matrix |
| Full verification | Verdict macro F1 | Evidence-supported accuracy, citation validity, evidence sufficiency, abstention quality, temporal correctness |

## Rules

1. Dataset-specific test sets remain untouched until final evaluation.
2. Event, thread, source, article, claim, and near-duplicate groups stay in one
   partition as appropriate.
3. Duplicate and near-duplicate checks happen before splitting.
4. Retrieval, reranking, entailment, and aggregation are evaluated separately
   before end-to-end fusion.
5. Cross-domain results are reported separately from in-domain results.
6. Abstentions and unavailable verifications are measured, not counted as
   silent failures or forced labels.
7. Citation validity requires that a cited URL and excerpt came from the
   recorded retrieval result.
8. Temporal evaluation declares its cutoff and allowed evidence window.
9. Every experiment records dataset version, seed, package versions, model
   configuration, runtime, and artifact paths.
10. No model is called a fact-checker, and no result is called near-perfect
    without independent, leakage-safe evaluation.

## Robustness plan

Future evaluation should include spelling variation, Hinglish, transliteration,
hashtags, URLs, emojis, paraphrases, short/long inputs, code-mixed Hindi-English,
OCR noise, image compression, and stale or unavailable sources. These tests
remain separate from the clean benchmark score.
