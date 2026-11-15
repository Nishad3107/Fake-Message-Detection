# Experiment Runtime

Recorded from safe isolated run `run_20261115T042010Z` on the current local
environment using Python 3.11.14. Durations are measured wall-clock seconds by
the runner and are not estimates.

| Stage | Duration |
| --- | ---: |
| Dataset validation | 3.210981 s |
| Dataset preparation | 1.103160 s |
| EDA | 15.769007 s |
| Split creation | 0.836209 s |
| Text preprocessing | 0.267532 s |
| Baseline training | 0.980718 s |
| Hyperparameter tuning | 17.980939 s |
| Final evaluation | 11.072369 s |
| Error analysis | 11.150546 s |
| **Total** | **62.545390 s** |

The run used 2,271 processed PHEME records and produced split sizes of
1,507 train, 467 validation, and 297 test records. The reproduced final test
metrics matched the frozen report, including macro F1 `0.211379` and accuracy
`0.225589`; no frozen artifact hash changed.

## Safe reruns

Use `make reproduce-experiment-safe`. Each run receives a new UTC timestamped
directory under `backend/data/experiments/`. Existing model artifacts,
research reports, and frontend metrics are never used as write targets by the
isolated runner. The generated summary records package versions, Git commit,
stage timings, configurations, and before/after SHA-256 hashes.

Runtime can vary with Python, scikit-learn, BLAS, operating-system, disk, and
matplotlib cache state. Future runs should be compared using their own
`experiment_summary.json` and `reproduction_comparison.json` files.
