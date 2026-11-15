# Viva Preparation

The answers below use the repository's actual PHEME reports and frozen-model
results. They describe a research prototype, not an autonomous fact-checker.

## Questions and answers

### 1. What is misinformation?

In this project, misinformation refers to rumor messages whose PHEME annotation
is `false`; the task also includes `true` and `unverified`. The model predicts
historical annotation labels, not universal truth.

### 2. Why was PHEME selected?

PHEME contains real social-media rumor source messages organized into events,
threads, and veracity annotations. That event structure makes leakage-aware
evaluation possible and is more appropriate than the 20-row synthetic sample.

### 3. What are the PHEME labels?

The processed task has three labels: `true`, `false`, and `unverified`.

### 4. Why not use a random split?

Messages from the same event can share names, topics, and phrasing. A random
row split can put related context in both training and test data and inflate
generalization estimates.

### 5. What is event-level leakage?

It occurs when records from one event appear in multiple partitions. The model
can then memorize event-specific vocabulary instead of learning transferable
patterns.

### 6. What is TF-IDF?

TF-IDF weights a term by its importance in a document while down-weighting terms
common across the fitting corpus. It creates sparse numeric features for text
classification.

### 7. Why use unigrams and bigrams?

Unigrams capture individual terms and bigrams capture short phrases. The
baseline used word n-grams `(1, 2)` as a simple, interpretable starting point.

### 8. Why compare multiple models?

Different classifiers make different assumptions about sparse text. The project
compared a majority baseline, Multinomial Naive Bayes, Logistic Regression, and
LinearSVC instead of assuming one model would be best.

### 9. Why did Naive Bayes win validation after tuning?

The selected Naive Bayes configuration had validation macro F1 `0.372437` and
grouped-CV mean `0.197411`, higher than the selected candidates in the tuning
run. This is an experiment result, not proof of universal superiority.

### 10. Why was macro F1 used?

Macro F1 gives each class equal weight. That matters because `true`, `false`,
and `unverified` have different supports and a majority strategy can hide poor
minority-class performance.

### 11. What is the difference between validation and test data?

Validation data supports model comparison and tuning. The test data is held back
for one final evaluation after the configuration is frozen.

### 12. Why was test performance lower?

The test events were unseen and had different topic vocabulary and label
composition. Validation macro F1 was `0.372437`, while final test macro F1 was
`0.211379`, showing event distribution shift.

### 13. Why is `false` recall low?

Only 12 of 111 false examples were correctly identified. Many false messages
were predicted as true (51) or unverified (48), suggesting that surface
language did not transfer across the held-out Germanwings event.

### 14. Why is `unverified` precision low?

The model predicted unverified for many messages whose labels were true or
false. Its final precision was `0.117188`, despite recall `0.428571`.

### 15. Why is Gurlitt performance weak?

Gurlitt had 59 records, mostly true, with only two unverified records and no
false records. Its accuracy was `0.118644`, and three-class macro F1 was not
defined for that event.

### 16. What does confidence mean?

It is the highest class score output by Multinomial Naive Bayes. It is not a
calibrated probability that the underlying claim is factually true.

### 17. Are model probabilities factual truth probabilities?

No. They are statistical model outputs based on learned text patterns. The test
expected calibration error was `0.378936`, and the product displays warnings.

### 18. How does the backend load the model?

`FrozenModelService` loads `backend/models/final/final_pipeline.joblib` once at
startup, validates the artifact, and keeps the model and vectorizer in memory.

### 19. How does the frontend communicate with the backend?

The React frontend calls the typed FastAPI endpoints using the configured
`VITE_API_URL`, defaulting to the local backend at port 8000.

### 20. How is history stored?

Successful predictions are stored in a local SQLite database. Docker uses a
named volume so history survives a safe container restart.

### 21. What happens if the model artifact is missing?

Health reports degraded readiness and prediction returns a sanitized HTTP 503.
The API does not train a replacement during startup.

### 22. What are the security limitations?

The prototype has no authentication, rate limiting, or encrypted database.
Production requires HTTPS, explicit CORS, access controls, and a retention
policy.

### 23. What is data leakage?

Data leakage is unintended information from evaluation data entering training,
feature learning, model selection, or preprocessing decisions.

### 24. What is overfitting?

Overfitting is learning idiosyncrasies of the available training events that do
not generalize. The gap between validation and unseen-event test results is a
warning sign.

### 25. What is distribution shift?

Distribution shift means the test data has different topics, vocabulary,
label proportions, or writing patterns from training data. PHEME's event
structure makes this especially important.

### 26. Why was LinearSVC not selected?

Its baseline validation macro F1 was `0.292217`, below Logistic Regression, and
the selected tuned candidate had grouped-CV macro F1 `0.171292`. Its decision
margins also are not probabilities without calibration.

### 27. What is hyperparameter tuning?

It is systematic evaluation of a documented parameter grid, such as TF-IDF
`min_df`, n-gram range, Naive Bayes `alpha`, and classifier regularization.

### 28. Why use grouped cross-validation?

`StratifiedGroupKFold` keeps event groups together while attempting label
stratification. Ordinary folds would mix event context and risk optimistic
results.

### 29. Why not use a transformer model?

The project prioritized a small explainable baseline and reproducibility. A
transformer should be a separately governed experiment after fair grouped
evaluation and resource checks.

### 30. What are the ethical limitations?

The labels are historical annotations, not objective truth. The system can be
wrong or biased and must not be used to censor people, target groups, or make
high-stakes decisions.

### 31. What would you improve next?

I would add more diverse and recent events, evidence retrieval, calibrated
uncertainty, temporal and cross-dataset evaluation, human review, multilingual
support, and stronger production security.

### 32. Why preserve the synthetic dataset?

It supports development and CI fallback without pretending to represent real
misinformation. It is never combined with PHEME or used for final research
claims.

### 33. How many records were evaluated finally?

The final test contains 297 records from two held-out events. The complete
processed PHEME file contains 2,271 records.

### 34. What is the final confusion matrix?

For label order `false`, `true`, `unverified`, it is
`[[12,51,48],[46,40,65],[7,13,15]]`.

### 35. What is the main academic contribution?

It is a reproducible, leakage-aware, explainable workflow with honest
event-level evaluation and full-stack integration, including the finding that
validation performance did not transfer reliably to unseen events.

