# Understanding and extending the forecasting project


| File | Purpose |
| --- | --- |
| `forecast_package/model.py` | Shared feature construction and the live `Forecaster` interface |
| `forecast_package/train.py` | Estimator construction, chronological model selection, and artifact saving |
| `run_experiments.py` | Baselines, frozen models, retraining policies, and chronological evaluation |
| `analysis_tools.py` | Metrics, uncertainty intervals, drift detection, day clustering, and plots |
| `tests/test_forecasting.py` | Checks for features, leakage, targets, baselines, metrics, and retraining timing |
| `run_project.py` | Environment check followed by tests, experiments, and final training |
| `../results/` | Saved configurations, metrics, predictions, plots, and experiment records |
| `submission_package/` | Separately staged copy of the existing model, training code, and artifacts |

There are two tasks: predict NSW demand in MW and classify whether each NSW
half-hour price exceeds **the previous market day's mean price**. This is not
classification against the previous half-hour's price.

Features include calendar and NSW holidays, forecast-day weather, market lags
of 1/2/3/7/14/21/28 days, and historical summaries across NSW/VIC/QLD. Forecast-day
market observations are excluded. The first 28 days cannot supply all lags.
Demand models learn a correction to the average of yesterday's and last week's
same-slot demand. Price models learn class probabilities; the decision threshold
is tuned using 2014 validation data.

The two baselines use yesterday's and last week's observations.
The original algorithms are Ridge regression / logistic regression (`linear`)
and LightGBM regression / classification (`boosted`). Initial fitting uses
2010–2013, model and threshold selection uses 2014, and simulation starts in
2015. Retraining strategies are frozen, periodic expanding history, periodic
two-year history, and drift-triggered four-year history. Simulation allows up to
six retrainings per task and strategy. Strategy selection uses 2015–2024 by default;
2025–October 2026 is the later holdout.

The saved `results/final_config.json` selects LightGBM for both tasks with periodic
expanding-history retraining. Saved holdout results in `aggregate_metrics.csv`:

| Setup | Demand MAPE (%) ↓ | Demand MAE (MW) ↓ | Price accuracy (%) ↑ | Price balanced accuracy (%) ↑ |
| --- | ---: | ---: | ---: | ---: |
| Yesterday baseline | 7.53 | 532.35 | 76.96 | 76.97 |
| Last-week baseline | 10.00 | 703.82 | 73.07 | 73.03 |
| Selected LightGBM + periodic expanding | 3.67 | 253.49 | 82.61 | 82.54 |

Final training refits the selected estimator configuration through the dataset's
end (or the policy's explicit training cutoff); it does not replay the simulation's
historical retraining dates. The staged submission directory is a separate copy
and does not update automatically.

## Temporal leakage safeguards

Feature construction requires consecutive complete market days (00:30 through
the following midnight). This keeps row shifts aligned with past days, including
at year boundaries. Market lags and rolling summaries end on the previous day;
forecast-day weather and its daily summaries are allowed by the project protocol.
The supplied weather is reanalysis treated as available in advance, so these
scores describe that idealised scenario rather than actual weather forecasts.

Every newly fitted model records its training date range and rejects predictions
on or before its last training day. `Forecaster` reads the companion manifest for
older artifacts without embedded dates and rejects artifacts with no provenance.
Use `run_experiments.py` for historical testing: final artifacts fitted through
October 2026 cannot be used to score 2015–2026. The same safeguards are included
in the staged submission code.

The 2014 scores are tuning scores because they select both models and price
thresholds. Strategy selection uses 2015–2024; assess the selected policy on the
later holdout without choosing a different policy based on its holdout score.
The holdout is a chronological deployment simulation: past holdout observations
can enter subsequent scheduled or drift-triggered retraining. Each day's outcome
is observed after its forecast, and triggered replacements start the next day.
PCA and clustering remain fixed from 2010–2013; completed-day cluster distances
feed the detector after that day finishes.

Regression tests check batch/live feature agreement, invariance to future-data
changes, training cutoffs, midnight boundaries, and adaptive forecast timing.

## Train random forests

Run commands from the repository root, in the **Python 3.11** environment with
`requirements.txt` installed (see the main README for setup). `python` below means
the activated environment's interpreter.

The algorithm lives in `code/forecast_package/random_forest.py`. It supplies a
`RandomForestRegressor` for demand corrections and a `RandomForestClassifier`
for price direction. Both use two worker threads and seed 9417. Price training
uses balanced class weights, and demand retains the shared inverse-demand sample
weights. The full candidate set tries 200 trees with depth/leaf-size pairs of
12/20 and 20/40, using 70% of features per split.

For a first training check:

```bash
python code/forecast_package/train.py data/nem_weather_halfhourly_2010_2026.csv --families random_forest --quick --output code/forecast_package/artifacts_rf_quick
```

This actually fits both forests, selects settings on 2014, and saves
`models.joblib` and `manifest.json`. Quick mode uses 20 trees and one candidate.
It still processes the full CSV. Without a simulation configuration, final fitting
uses the most recent four years; the script labels this window as unvalidated.
Use it to check training, not to claim that random forest outperforms LightGBM.

For a full forest-only experiment and final training:

```bash
python code/run_experiments.py --families random_forest --output results/random_forest --resume
python code/forecast_package/train.py data/nem_weather_halfhourly_2010_2026.csv --selection results/random_forest/final_config.json --output code/forecast_package/artifacts_rf
```

The first command evaluates both forest candidates, baselines, and all existing
retraining strategies. The second fits the selected settings and writes separate
artifacts. `--selection` controls model families when supplied; `--families` only
controls candidate search when no selection file is supplied. Full experiments
perform many fits and can take substantial time; `--resume` reuses completed,
matching cached stages. Quick experiments support `--quick`, but use a separate
output directory for them.

Load the resulting forests using the existing interface:

```python
import sys
sys.path.insert(0, "code/forecast_package")
from model import Forecaster

forecaster = Forecaster("code/forecast_package/artifacts_rf")
# history and weather_next follow the kit's day-ahead interface:
# predictions = forecaster.predict(history, weather_next)
```

## Compare algorithms fairly

To compare linear models, LightGBM, and random forest in one run:

```bash
python code/run_experiments.py --families linear boosted random_forest --output results/model_comparison --resume
```

Read these files inside the chosen output directory:

- `selection.json`: each candidate's 2014 validation scores and the selected family
  for each task. Demand uses lowest MAPE; price uses highest balanced accuracy.
- `aggregate_metrics.csv`: simulation, strategy-selection, and holdout scores.
- `period_metrics.csv`: performance over time and uncertainty intervals.
- `final_config.json`: selected models and retraining policies for final fitting.
- `feature_importance.csv`: importance for each candidate. Forest importance is
  mean decrease in impurity; its scale differs from LightGBM gain and linear coefficients.

The `frozen_*` rows compare candidate algorithms trained through 2014. The
`adapted_*` rows apply retraining policies to the **2014-selected winner for each
task**; a combined run does not apply every policy to every algorithm. To compare
adapted forest and LightGBM systems, run each family separately with the same
date boundaries and compare corresponding policy rows. Existing saved results
are useful reference measurements, but a fresh full comparison avoids mixing
quick and full settings or different feature versions.

Keep chronological splits; random train/test splitting would mix past and future.
Use validation/strategy-selection scores to make choices, then report the holdout.
Repeatedly choosing settings based on holdout scores turns that period into
another validation set. Final fitted artifacts are for inference; scoring them
on their own training rows is not a performance comparison.

## Add another algorithm

Follow `random_forest.py`: put candidate dictionaries and estimator construction
in a new module. Register its family in `train.py`'s `FAMILIES`, add its candidates
to `candidates()`, and add its construction branch to `fit_model()`. Regression
must predict demand residuals; classification must provide two-class
`predict_proba`. The shared pipeline handles sample weights, thresholds, targets,
and the live interface. Add the new module to the source fingerprint in
`run_experiments.py` so cached runs invalidate when its code changes. Extend
feature-importance reporting if the estimator exposes a different API.

Check fitting, saving/loading, and live prediction as well as the existing tests:

```bash
python -m unittest discover -s code/tests -v
```

Default commands still compare only linear and boosted models. The new family
is enabled explicitly with `--families`. A submission trained using this updated
trainer needs `random_forest.py` alongside `model.py` and `train.py`, because the
trainer imports it. Put the chosen output artifacts in the submission's
`artifacts/` directory and check the kit's total artifact-size and runtime limits
before submission. The commands above leave the original artifacts and results
files in place.
