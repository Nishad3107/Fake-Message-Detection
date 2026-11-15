# Presentation Outline

Suggested 12-slide academic presentation. All numbers below come from the
repository reports and should be presented as research-prototype results.

## Slide 1 — Title

**Main points:** Misinformation Detection in Social Media Rumor Messages Using
NLP; name, course, and date.

**Recommended visual:** Minimal title slide with a message-to-model pipeline.

**Speaker notes:** Introduce this as probabilistic language-pattern analysis,
not factual verification.

## Slide 2 — Problem statement

**Main points:** Classify a rumor source message as `true`, `false`, or
`unverified`; short social posts lack context and shift across events.

**Recommended visual:** Three-label task diagram.

**Speaker notes:** State non-goals: no evidence retrieval, source verification,
or autonomous moderation.

## Slide 3 — Objectives

**Main points:** Real-data preparation, explainable preprocessing, model
comparison, leakage-safe evaluation, API/frontend integration, and reproducible
artifacts.

**Recommended visual:** Objective checklist or workflow timeline.

**Speaker notes:** Emphasize that reproducibility and honest failure analysis are
first-class objectives.

## Slide 4 — Dataset

**Main points:** PHEME; 9 events; 6,425 raw threads; 2,271 processed records;
labels 1,064 true, 635 false, 572 unverified.

**Recommended visual:** Label distribution chart and event count table.

**Speaker notes:** Explain that 4,023 non-rumor threads and 124 unsupported
annotations were excluded from the three-class veracity task.

## Slide 5 — Data preprocessing

**Main points:** Unicode normalization, lowercasing, URL/mention replacement,
hashtag retention, whitespace/punctuation normalization, negation preservation.

**Recommended visual:** Before/after text example and preprocessing flow.

**Speaker notes:** Cleaning is stateless; vocabulary learning happens later on
training text only.

## Slide 6 — Event-level splitting

**Main points:** Train 1,507, validation 467, test 297; 5/2/2 event allocation;
seed 42; no event or thread overlap.

**Recommended visual:** Colored event-to-split allocation diagram.

**Speaker notes:** Explain why a random row split would leak event-specific
language and why nine events make the estimate unstable.

## Slide 7 — Models compared

**Main points:** Majority, Multinomial Naive Bayes, Logistic Regression,
LinearSVC; baseline macro F1 values 0.194949, 0.250895, 0.343505, 0.292217.

**Recommended visual:** Baseline macro F1 bar chart with confidence-type labels.

**Speaker notes:** Macro F1 is primary; LinearSVC margins are not probabilities.

## Slide 8 — Final model

**Main points:** Tuned Multinomial Naive Bayes; `(1,2)` TF-IDF, `min_df=3`,
`sublinear_tf=false`, `alpha=0.1`, `fit_prior=false`; 144 configurations and
five grouped folds.

**Recommended visual:** Configuration card and grouped-CV fold schematic.

**Speaker notes:** Test data was not used for tuning or model selection.

## Slide 9 — Final results

**Main points:** Test accuracy 0.225589, macro F1 0.211379, weighted F1
0.232157, log loss 1.694920, OVR ROC-AUC 0.417280.

**Recommended visual:** Final confusion matrix and validation-versus-test bar.

**Speaker notes:** Make the validation/test gap explicit; do not present the
result as good real-world fact-checking.

## Slide 10 — Error analysis

**Main points:** Largest pairs: true→unverified 65, false→true 51,
false→unverified 48; false F1 0.136364; Gurlitt accuracy 0.118644; 25
high-confidence errors.

**Recommended visual:** Normalized confusion matrix and event accuracy chart.

**Speaker notes:** Event-specific names and topics act as shortcuts and fail on
unseen events.

## Slide 11 — System architecture

**Main points:** React → FastAPI → preprocessing → TF-IDF → frozen MNB →
probabilities/explanations → SQLite history → dashboard; Docker and CI support.

**Recommended visual:** Architecture flow diagram.

**Speaker notes:** Mention model health checks, sanitized errors, CORS, and local
privacy notice.

## Slide 12 — Limitations and future work

**Main points:** Low unseen-event performance, nine events, calibration limits,
English scope, no evidence retrieval, no authentication; future work includes
more events, calibration, evidence, human review, temporal/cross-dataset tests.

**Recommended visual:** Limitations-to-future-work two-column layout.

**Speaker notes:** Close with the main contribution: reproducible and honest
academic methodology, not an autonomous truth detector.

