# Price-only optimisation

Run from the repository root using the pinned Python 3.11 environment. Each run
requires an empty output directory. Existing results, artifacts and the staged
submission are never replaced by the research runner.

```bash
python -m unittest discover -s code/tests -v
python code/run_price_experiments.py --output results/price_features_full
python code/run_price_experiments.py --phase policies --feature-set both --output results/price_policies_full
python code/run_price_experiments.py --phase tuning --feature-set both --output results/price_tuning_full
python code/run_price_experiments.py --phase complementary --feature-set both --output results/price_complementary_full
```

`both` above is an example, not a selected winner. Use the feature set supported
by the preceding experiment. `--quick` uses 120 rather than 600 trees for pipeline
checks; it is not an accuracy comparison. `--hours` bounds admission of subsequent
candidates; a fit already in progress is allowed to complete.

The runner caps the data at market day 2024-12-31 before feature construction.
Initial validation trains through 2013 and scores 2014. Historical predictions
start with a fit through 2014, then update in 2016/2018/2020/2022/2024. There are
five historical replacements through 2024, leaving room for the sixth in 2026.
2015-2022 is development; 2023-2024 is previously inspected confirmation.
Neither is an untouched test. No 2025-2026 outcomes are scored.

Features are base (240), geometry (+12), recency (+7), both (+19). Geometry
normalises seven existing margins by yesterday's IQR floored at $1/MWh, measures
mean/median and trimmed-mean distortion, scaled volatility, spike fraction and
negative-price fraction. Recency uses 7/28-day same-slot and 28-day all-slot
exceedances above the CURRENT threshold, yesterday's adjacent-slot margin,
late-day change, daily mean momentum and weekly threshold anomaly. Targets
always use the actual previous-day arithmetic mean. See price_context.py for
precise definitions. The demand path retains its original inputs.

Policy comparisons isolate threshold refresh, class weighting, 2/4-year windows
and 2/4-year half-life weights. Refresh uses two chronological blocks covering
the previous 365 days, training auxiliary models strictly before each block.
The deployed model is refitted only on scheduled occasions. Thresholds maximise
past balanced accuracy with ties resolved near the preceding cutoff. Time-decay
weights are balanced once using weighted class totals.

Tuning screens 16 sampled configurations plus the incumbent on 2012/2013 folds;
it evaluates at most four screened candidates and the control on later dates.
Incompatible depth/leaf samples are discarded. Sampled boosted candidates use
90-day chronological early stopping then refit the fitting prefix at the chosen
iteration count. Tuning currently retains the 600-tree ceiling to control cost.
Complementary runs compare the control, four logistic settings and six CPU
histogram XGBoost settings. They do not automatically ensemble or deploy models.

Outputs include predictions/probabilities, per-class recall, pooled/year/month
metrics, fit provenance, validation scores and paired seven-day moving-block
bootstrap intervals on development balanced accuracy. Metrics also include both
required price baselines. Error cohorts based on actual future price margins are
retrospective diagnostics only. `selection.json` records Stage 1 and development
leaders separately; it is NOT an automatic promotion decision.

Promotion review: at least +0.25 pp development BA, positive changes in six of
eight years, no pooled accuracy loss over 0.25 pp, a supportive paired interval,
and confirmation of the direction of change in 2023-2024. Do not keep tuning on
2025-2026. Repeat finalists with additional seeds before final selection.

For a reviewed fixed-threshold candidate, build a separate package:

```bash
python code/train_price_candidate.py --research results/price_features_full --candidate both --output /tmp/price_candidate_package
```

This inherits the incumbent demand configuration unchanged, rebuilds both tasks
from the released CSV, and includes every required helper module. The generated
bare `train.py CSV` invocation reproduces the configuration without research
files. It does not overwrite or activate the existing submission. Refreshed
threshold policies cannot be exported until final-cutoff calibration is added.
Verify CPU limits, peak memory, package size, fresh-process inference and rebuild
reproducibility before using a candidate submission.

Deferred conditional work: demand OOF stacking, continuous margin regression,
calibration, probability ensembles, specialist regimes and automated promotion
remain research extensions. The present implementation provides the controlled
feature, policy and tuning experiments needed to decide whether they are justified.

To queue the policy, tuning and complementary-model experiments after a full
feature run completes (or while it is running):

```bash
python code/run_price_research.py --features-results results/price_features_full --output results/price_research_queue --hours 24
python code/analyse_price_experiments.py results/price_features_full
```

The queue selects its feature representation using 2014 validation only. It
records progress in queue.json and individual log files. Each subsequent phase
admits candidates for at most four hours within the overall admission budget.
An already-running candidate completes; this is not a hard process timeout.
The queue never replaces the submission or automatically approves a candidate.
