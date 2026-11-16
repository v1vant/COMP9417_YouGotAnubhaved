# Optimised demand splines

Run from the repository root in `conda activate comp9417`:

```bash
python code/forecast_package/optimise_spline_ridge.py --output results/spline_optimisation_repeat
```

The runner preserves existing files and requires a new, empty output directory.
It caps the data at the last market day of 2024 **before feature construction**.
It compares eight structural settings on 2015–2022, then tests two nearby
regularisation/complexity settings for the best structural candidate. The winner
is fixed before evaluating it and the original model on 2023–2024. All periods
are exploratory: these data have been inspected before. Neither stage uses
2025–2026 outcomes. All preprocessing is refitted from each training window.

Candidate definitions (all start with alpha 10 and six weather knots):

| Name | History | Retraining | Added features |
| --- | --- | --- | --- |
| structure_0 | Expanding | Biennial | Original model |
| structure_1 | 2 years | Biennial | Original model |
| structure_2 | 4 years | Biennial | Original model |
| structure_3 | 2 years | Annual | Original model |
| structure_4 | 4 years | Annual | Original model |
| structure_5 | 2 years | Annual | Solar-response trend and anchor changes |
| structure_6 | 2 years | Annual | Above, regional weather/demand and weekday × hour |
| structure_7 | 4 years | Annual | Same as structure_6 |

Biennial fitting is effective in 2015, 2016, 2018, 2020, 2022, 2024;
annual fitting is effective every January. Rolling windows include the preceding
two or four calendar years. The expanded features include solar changes versus
the anchor, their interaction with elapsed calendar years, temperature changes,
and recent demand minus its four-week same-slot average. These changes also
interact with the hour splines. This is a limited demand-shape representation,
not the full midday/evening summary experiment. The rich variant additionally
includes VIC/QLD demand lags and weather from Melbourne/Brisbane.

`development_metrics.csv` and `confirmation_metrics.csv` report MAPE, MAE,
and signed bias (positive means overprediction), pooled and by year, for all
hours, 10:00–15:00 interval starts, and other hours. `fits.csv` and
`confirmation_fits.csv` record training dates and runtime. Each candidate's
predictions are saved as Parquet. `selection.json` records the selected model.
Timing guards stop adding search candidates after one hour and tuning after
75 minutes; these are search guards rather than a hard process timeout.

The experiment also fits `demand.joblib` through 2024 using the selected window.
For an independent rebuild without repeating the experiments:

```bash
python code/forecast_package/optimise_spline_ridge.py \
  --train-selection results/spline_optimisation/selection.json \
  --output results/spline_selected_rebuild
```

To fit through the released dataset's end **after freezing the configuration**,
add `--cutoff 2026-10-31` to that rebuild command. This does not score the later
outcomes. Default cutoff is 2024-12-31. This is demand-only, not a full submission.

Load with `code/forecast_package` on `sys.path`, import
`optimise_spline_ridge`, then call `joblib.load` and the model's
`predict_day(history, weather_next)`. Forecast dates must follow the training
cutoff. Both the original spline source and shared `model.py` must remain
importable. The estimator uses coefficient Ridge penalties, not a separate
curvature penalty.

Checks:

```bash
python -m unittest discover -s code/tests -p 'test_optimise_spline.py' -v
```
