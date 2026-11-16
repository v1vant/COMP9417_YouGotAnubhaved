# Spline + Ridge demand model

Run commands from the repository root in the project environment:

```bash
conda activate comp9417
python code/forecast_package/spline_ridge.py --quick --output results/spline_ridge_quick
```

This tries one setting (`alpha=100`, six weather knots), scores it on 2014,
then fits a final demand model on all supplied data. Quick mode reduces the
search, not the final training history. Timings are printed and saved.

For the full six-setting search and a chronological development backtest:

```bash
python code/forecast_package/spline_ridge.py --evaluate --output results/spline_ridge_full
```

Both commands use the released CSV by default. An alternative CSV can be passed
as the first positional argument. Output directories must be empty to protect
existing results. Use `--threads 4` to change the default two-thread limit.
There is no enforced two-hour timeout.

The full search tries four/six weather knots and Ridge alpha 10/100/1000.
Settings are selected by 2014 demand MAPE after fitting on 2010–2013. Optional
development evaluation starts with a fit through 2014 and refits on expanding
history for 2016/2018/2020/2022/2024, matching the saved LightGBM schedule.
It scores 2015–2024 only. Supplied 2025–2026 observations enter only the final
artifact fit, after selection and development evaluation; no holdout score is
computed. Comparisons should use the same evaluation period and schedule.

Outputs:

- `demand.joblib`: fitted `SplineRidgeDemand`, including preprocessing and training dates.
- `validation.csv`: 2014 tuning scores and fit timings.
- `manifest.json`: chosen settings, training span, feature count and timings.
- With `--evaluate`: `development_metrics.csv`, `development_predictions.parquet`
  and `fits.csv`.

The model predicts a correction to the average of yesterday's and last week's
same-slot demand. It uses NSW demand lags, Sydney weather, holiday context,
periodic time-of-day and annual splines, and temperature/solar interactions
with time-of-day. Sparse cubic spline features feed a scaled Ridge fit using
inverse-demand sample weights. Ridge penalizes coefficient magnitude; it does
not apply a separate spline curvature penalty. Only training data determine
scalers and weather knot locations.

Load the model from Python with the source directory importable:

```python
import sys
import joblib

sys.path.insert(0, "code/forecast_package")
import spline_ridge

model = joblib.load("results/spline_ridge_full/demand.joblib")
demand = model.predict_day(history, weather_next)
```

`history` and `weather_next` use the same format as the existing forecasting
package: complete market days, at least 28 days of past observations, and
48 next-day weather rows. Predictions must be after the artifact's training
cutoff. Use the backtest for historical scores rather than the final artifact.

This is a **demand-only experiment**, not a complete submission package.
It does not train price classification or alter existing LightGBM artifacts.

Verification:

```bash
python -m unittest discover -s code/tests -p 'test_spline_ridge.py' -v
```
