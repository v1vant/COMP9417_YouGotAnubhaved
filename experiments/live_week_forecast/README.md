# Live-week performance estimate

This folder freezes the current recipe and sets up historical simulations of a
model trained through **31 October** and evaluated on **7–13 December**.
It does not generate actual December 2026 forecasts before the required daily
history and supplied next-day weather become available.

## Frozen model and daily inputs

- Demand: 50% one LightGBM, 50% the mean of five ten-epoch MLPs, seeds 9417–9421.
- Price: five-seed C2, weather shape enabled, threshold 0.50.
- Fit once through each declared cutoff. Freeze models and scalers all week.
- Each day receives actual history through the preceding day and that day's
  permitted weather. November/December observations update inputs, not weights.
- The yesterday baseline uses exactly the same target and input conventions.
- Source code, the official harness and recipe are copied here; their hashes
  are checked before runs. Production packages and artifacts are never written.

## Windows

- **Seasonal:** 2015–2025 inclusive. Train through 31 October, score 7–13 December.
- **Recent:** ten month-end cutoffs, December 2024 through September 2025.
  Begin a seven-day test 37 days after each cutoff. The first week is
  6–12 February 2025; the last is 6–12 November 2025.
- All 21 evaluation weeks are non-overlapping. Training windows may overlap and
  contain outcomes from earlier tests, as chronological deployment permits.
- Seasonal and recent estimates stay separate. Earlier seasonal years may be
  less representative of current conditions.

## Estimates and uncertainty

Each week supplies demand MAPE/MAE and price accuracy/balanced accuracy over all
336 half-hours. Balanced accuracy uses the official mean recall over classes
present in that week. Group means are **means of weekly scores**; in particular,
mean weekly balanced accuracy is not pooled balanced accuracy across all weeks.

- Report every week, mean, median, minimum and maximum.
- Report empirical 10th–90th percentile weekly ranges. These describe observed
  variation, **not a calibrated prediction interval for December 2026**.
- For a **95% confidence interval for the historical mean weekly score**, resample
  whole evaluation weeks with replacement 5,000 times (seed 9417); use the
  2.5th/97.5th percentiles of the means.
- Baseline comparisons are paired: identical sampled weeks for both methods.
  Positive improvement means baseline minus current for errors, and current
  minus baseline for accuracy metrics.
- Sensitivity analysis for recent weeks resamples circular blocks of two adjacent
  test windows. Those windows are roughly monthly, not adjacent calendar weeks.
- No mean CI is emitted with fewer than five completed weeks in a group.

These intervals are approximate: small samples, regime changes, overlapping
training sets and residual dependence limit interpretation. They do not account
for model-selection uncertainty, repeated historical/holdout inspection or new
training seeds. No hyperparameters, thresholds or blend weights are selected
from these results. No half-hours are treated as independent bootstrap units.

## Commands (from the repository root)

Inventory, schedule validation and cost estimate only; **no historical training**:

```bash
python experiments/live_week_forecast/run.py
```

Fast tests, including a synthetic clean-package training/inference smoke test:

```bash
python -m unittest discover -s experiments/live_week_forecast/tests -v
```

Explicitly fit and evaluate one pilot window:

```bash
python experiments/live_week_forecast/run.py --execute --window seasonal_2025
```

Run the complete historical study (potentially many hours):

```bash
python experiments/live_week_forecast/run.py --execute
```

Use `--group seasonal` or `--group recent` to limit execution. Completed snapshots
and predictions are reused only if their recipe/source/data fingerprints and
artifact checksums match. Quick smoke caches are distinct and cannot be used
by a real evaluation. Resume is at the completed-cutoff level, not within a fit.

## Outputs

- `windows.csv`: all declared historical windows.
- `inventory.csv`, `inventory.json`: eligible caches, missing fits, measured-timing
  extrapolation and whether the existing production package matches the actual
  live cutoff. That production package is never reused for historical weeks.
- `cache/`: fitted packages per cutoff (ignored by Git; about 24 MB per snapshot).
- `logs/`: training logs (ignored by Git).
- `predictions/`: 336 half-hour forecasts, truths and baselines per window.
- `results/weekly_metrics.csv`: scores per week/model.
- `results/summary.csv`: group means, medians, empirical ranges and mean CIs.
- `results/paired_improvement.csv`: paired differences and 95% intervals.
- `results/recent_pair_block_sensitivity.csv`: two-window block sensitivity.
- `figures/`: separate seasonal/recent score plots plus `summary_intervals`,
  which distinguishes 95% CIs for the historical mean from the empirical
  10th–90th percentile weekly range. PNG and PDF outputs.

Results files and figures appear only after historical evaluation. The synthetic
smoke test uses temporary directories and must not create historical scores.

## December 2024 standalone task

From the repository root, with project dependencies installed and the released
CSV at `data/nem_weather_halfhourly_2010_2026.csv`:

```bash
python experiments/live_week_forecast/run.py --execute --window seasonal_2024
```

This trains the frozen current demand blend (one LightGBM plus five ten-epoch
MLPs) and five-seed C2 price model on data through **31 October 2024**, then
forecasts **7–13 December 2024** using daily updated history. Budget roughly
30–40 minutes on the original machine; hardware changes runtime. It writes only
inside this experiment folder and does not replace production models. Each
person should use their own checkout/output folder, rather than running two
writers against the same cache. `--data /path/to/file.csv` overrides the CSV path.
Python 3.12.12 was used locally; NumPy, pandas, scipy, scikit-learn, LightGBM,
PyTorch, joblib, threadpoolctl and matplotlib are required. See repository
requirements for the project environment.

The source/recipe hashes are checked before execution. The command can resume
completed snapshots, but an interrupted incomplete snapshot is fitted again.
