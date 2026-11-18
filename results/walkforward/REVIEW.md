# Best price-direction configuration so far: review notes

This is a research result for review. It is not wired into `forecast_package` or `submission_package`.

## The configuration

```
P(up) = 0.7 * P_LightGBM + 0.3 * P_NN        predict "up" when P(up) > 0.5
```

| Part | Model | Inputs |
|---|---|---|
| `lgbm63_A5_blend` | LightGBM classifier with the package's settings except capacity: 63 leaves, 900 trees, learning rate 0.025, `class_weight="balanced"` | The 571 package features, plus Anubhav's 5 demand-forecast features (**A5**) |
| `nn256_D4_blend` | Our numpy MLP (`train.fit_neural_net`): 256-256 ReLU, dropout 0.5, AdamW lr 3e-4 and weight decay 1e-4, 10 epochs, 5-seed ensemble | The package NN inputs, plus the **D4** demand-forecast features (15 day-shape features and the 48-slot forecast profile) |

Both parts train from 2011, because the first out-of-sample demand forecasts are for 2011. Both use **Anubhav's day-ahead demand forecast**: 0.5 LightGBM + 0.5 ten-epoch MLP.

### Demand forecast inputs

`build_anubhav_blend.py` rebuilds Anubhav's forecast and aligns it with the price rows.

- **2011–2014:** yearly out-of-sample folds from commit `4f32442`. The file is `results/forecast_demand_price_2014/demand_forecasts_Y.csv`, and each year is trained on all earlier years.
- **2015–2024:** `0.5 * results/feature_comparison_2015_2024/current_demand.npy + 0.5 * results/demand_mlp_10epoch_2015_2024`. Both streams are refitted on the team schedule, with training ends 2014, 2015, 2017, 2019, 2021 and 2023.
- **Freshness:** every demand value comes from a model whose training end is no later than the price model's training end, so no input is fresher than the price model.
- **Accuracy:** MAPE is 2.29% over 2015–2024, against 2.53% for our LightGBM-only demand forecast. It is lower in every year.

### The A5 and D4 features

A5 is the five features from Anubhav's `added_features`, with `f` the demand forecast for slot t of day D+1:

- `f`
- `f - nsw_demand_lag_1`
- `f - nsw_demand_lag_7`
- `f - nsw_demand_previous_mean`
- `f - nsw_demand_previous_max`

D4 contains:

- **D2 features:** `f`, `f - demand_anchor`, `f - lag_1`, `f - lag_7` and `f / lag_1`.
- **Day-shape features of tomorrow's forecast:** mean, max, min and std; tomorrow's mean minus today's mean; slot minus day mean; slot minus day max; within-day rank; and the ramps to the previous and next slot.
- **The 48-slot forecast profile,** each slot minus today's mean demand.

See `add_demand_features` in `harness.py`.

## Protocol

- **Walk-forward:** the team schedule, fitting up to 2013 to predict 2014, then ≤2014 → 2015, ≤2015 → 2016-17, ≤2017 → 2018-19, ≤2019 → 2020-21, ≤2021 → 2022-23 and ≤2023 → 2024. Every prediction is out-of-sample.
- **Threshold:** fixed at 0.5, never tuned.
- **Data:** only rows up to market day 2024-12-31 are read. The 2025-2026 holdout was not read, trained on or scored.
- **Confidence intervals:** paired day-block bootstrap (whole market days resampled, 2000 draws), pooled over 2015-2024.

## Results, 2015-2024, threshold 0.5

| Configuration | BA 2015-24 | BA 2015-19 | BA 2020-24 | Accuracy 2015-24 | BA 2014 (tuning year) |
|---|---|---|---|---|---|
| **Best so far: 0.7 LGBM-63+A5 + 0.3 NN+D4** | **80.572** | **79.750** | **81.300** | **81.112** | 83.366 |
| LightGBM-63 + A5 (part 1) | 80.329 | 79.541 | 81.031 | 80.953 | 83.275 |
| 70/30 LightGBM-63 + NN 256x256, no demand | 80.227 | 79.428 | 80.938 | 80.730 | 83.104 |
| NN 256x256 + D4 (part 2) | 80.021 | 79.239 | 80.678 | 80.316 | 82.630 |
| LightGBM-63 from 2011, no demand | 80.021 | 79.182 | 80.784 | 80.631 | 82.985 |
| LightGBM-63, no demand | 80.006 | 79.154 | 80.776 | 80.599 | 83.142 |
| Package LightGBM-31, no demand | 79.886 | 78.995 | 80.693 | 80.415 | 82.940 |

### Gain of the best configuration over each alternative

| Alternative | BA gain [95% CI] | Accuracy gain [95% CI] | Years better (of 10) |
|---|---|---|---|
| Package LightGBM-31 | **+0.69** [+0.51, +0.86] | +0.70 [+0.52, +0.86] | 10 |
| LightGBM-63, no demand | **+0.57** [+0.39, +0.74] | +0.51 [+0.34, +0.68] | 9 |
| 70/30 LightGBM-63 + NN, no demand | **+0.35** [+0.19, +0.50] | +0.38 [+0.23, +0.53] | 9 |
| LightGBM-63 + A5 alone | **+0.24** [+0.15, +0.33] | +0.16 [+0.07, +0.24] | 9 |
| NN + D4 alone | **+0.55** [+0.38, +0.73] | +0.80 [+0.62, +0.97] | 10 |

`reproduce_best.py` writes the year-by-year BA. Among the models in this table, the best configuration has the highest BA in 8 of 10 years. It ties with NN+D4 in 2017 and is 0.07 behind LightGBM-63+A5 (and LightGBM-63) in 2016.

### What the demand inputs contribute

| Comparison | BA gain [95% CI] | Years better |
|---|---|---|
| LightGBM-63 + A5 (your forecast) vs LightGBM-63 from 2011 | +0.31 [+0.14, +0.48] | 8/10 |
| LightGBM-63 + A5 vs LightGBM-63 from 2010, the current-style model | +0.32 [+0.15, +0.49] | 10/10 |
| NN + D4 (your forecast) vs NN from 2011 | +0.42 [+0.28, +0.56] | 10/10 |
| A5 with your forecast vs A5 with our LightGBM-only forecast (LightGBM-63) | +0.10 [−0.02, +0.22] | 7/10 |

## Caveats

1. **Choices were made by looking at 2015-2024.** That covers the A5 vs D3/D4 feature sets, your forecast vs ours, and the NN partner. Only the 0.5 threshold is fixed a priori. The 0.7/0.3 weight was carried over from earlier LightGBM+NN blends; on 2014, log-loss picks about 0.7 for comparable pairs, and BA is flat from 0.4 to 0.7. Gains under about 0.15 BA over this configuration should be treated as noise.
2. **Seed noise.** A single LightGBM varies by about 0.08 BA (sd) across seeds over 2015-2024. The bootstrap intervals cover only day-to-day sampling, not refitting randomness.
3. **Weather.** The demand forecast, like every package feature, uses the supplied next-day weather.
4. **Not deployable as is.** The package would need to:
   - produce out-of-sample demand forecasts for every price training row from 2011, with yearly folds for 2011-2014 and then the refit schedule, inside `train.py`;
   - predict demand before price at inference;
   - add the two-model blend.

   Training the price model on in-sample demand values would over-trust the feature.
5. **Related results not included here:**
   - **Suhan's 5-member LightGBM ensemble** (63 leaves, 1200 trees, colsample 0.5, min_child 300, plus weather shape) scores 80.23 alone without demand features. Blended with NN+D4 it scores 80.56, a tie with this configuration (−0.01 [−0.13, +0.11]).
   - **His ensemble with A5 demand features** is being fitted now. It is the next candidate.

## Reproduce

From the repo root with the project venv:

```
python code/walkforward/reproduce_best.py
```

This rebuilds the data cache (rows up to 2024-12-31 only, about a minute) and scores the shipped predictions. It prints every table above.

To refit a part and compare it with the shipped predictions:

```
python code/walkforward/reproduce_best.py --refit lgbm63_A5_blend nn256_D4_blend --fit-ends 2013
```

That run took a few minutes and gave a maximum |difference| of 0.00e+00 for both parts. A full refit of every configuration takes about 1-2 hours on 4 cores.

## Files

- `code/walkforward/harness.py`: data cache (truncated to 2024), team schedule, demand-feature builders (`add_demand_features`), and one walk-forward fit (`fit_job`). Branches for other experiment kinds (torch, kNN, layered, extended NN) reference modules that are not part of this review.
- `code/walkforward/build_anubhav_blend.py`: rebuilds your forecast. It needs commit `4f32442` (branch `feature/price-spike-features`) for the 2011-2014 folds.
- `code/walkforward/evaluate.py`: labels, market-day codes and the saved-prediction loader.
- `code/walkforward/best_so_far.py`: definition of the best so far, plus the paired bootstrap comparison table.
- `code/walkforward/reproduce_best.py`: the entry point above.
- `results/walkforward/runs/`: out-of-sample probabilities for every fit (`<config>__<fit_end>.npz`, keys `y<year>`), fit metadata, `configs.json` (exact specs) and `anubhav_blend_forecast.npy`.
- `results/walkforward/results_table.csv`: the results table.
