# Strong-baseline price research v2

This isolated source snapshot starts at commit f92a38e, including the 571-feature
price model and the 405-feature, 50/50 LightGBM/MLP demand recipe. Existing code,
artifacts, staged submissions and v1 results are not overwritten. SOURCE.json
identifies the upstream snapshot; reference_model.py/reference_train.py retain
its original definitions. Changed inference adds price inputs after computing
the unchanged demand blend.

Run from the repository root in the pinned environment:

```bash
python -m unittest discover -s code/price_v2 -p test_v2.py -v
python code/price_v2/pipeline.py --output results/price_v2 --hours 24
```

The full pipeline runs P0 (571), P1 (590), P2 (571 + threshold refresh), P3
(590 + threshold refresh), and a compact 259-feature component. All comparisons
use complete training days and harness-consistent per-day means. Thresholds use
strictly earlier out-of-time predictions, with replacements in 2016/2018/2020/
2022/2024; the initial 2015 fit is not a replacement. Raw future market values
never enter inputs. Data are capped at 2024 before feature engineering.

An ensemble weight from 0/.25/.5/.75/1 is selected using 2014 predictions only.
Its cutoff is refreshed using historical component probabilities. If development
gains remain under one BA point, the pipeline generates OOF demand predictions
and runs paired stacked/control price models trained from 2011. MLP fitting runs
in isolated subprocesses (pinned CPU torch 2.15 is required); price inference
uses NumPy. Annual auxiliary folds cover 2011-2014; subsequent folds follow the
same biennial schedule. Audit files document every fold. Actual future demand is
never used. Main demand artifacts are not replaced by these auxiliary models.

status.json reports progress, including dependency failures. metrics.csv combines
all candidates and periods; summary.csv compares every candidate to P0, with
paired 7-day intervals, year counts, confirmation changes and promotion flags.
Predictions, raw probabilities, thresholds, calibration scores, validation data
and fit audits are saved per candidate. 2023-2024 confirmation is exploratory.
The hours limit stops admission of new stages; existing fits complete.

No automatic promotion, additional-seed verification, or 2025-2026 evaluation is
performed. After reviewing summary.csv, freeze one policy from configurations.json,
check two additional seeds and run the final later-period comparison before
adoption. Do not compare earlier-period BA directly with the published 85.03%
holdout result. Published legacy predictions were previously independently audited;
this pipeline establishes a corrected Python 3.11 control instead of borrowing it.

To build a candidate in a separate copy of this directory, put its policy dictionary
in candidate.json and run `python train.py CSV`. This first rebuilds the original
demand recipe and then replaces only the price bundle. Thresholds are calibrated
at the actual final cutoff. Ensemble and stacked policies have matching inference
paths. Disposable demand training caches live in /tmp, not artifacts. Rebuild
reproducibility, 2-hour training, 100 MB artifacts, 4 GB memory and fresh-process
latency must be verified before adopting the exported package. The original
submission remains the default until those checks pass.

## Final 2025–October 2026 historical comparison

After freezing one candidate, run:

```bash
python code/price_v2/holdout.py --research results/price_v2 --candidate stacked --output results/price_v2_holdout_stacked
```

Use a new output directory. The command evaluates P0 and the single selected
candidate on identical 669-day targets. The 2025 forecasts use a model fitted
through 2023 (the scheduled January-2024 fit); the next replacement is January
2026 using data through 2025. All threshold calibration and auxiliary demand
fits precede the days they forecast. The 2024 research cutoff is retained to
resolve threshold ties consistently. Candidate and source configuration are
frozen in frozen_selection.json before evaluation.

Open metrics.csv and use period=holdout for the full period, or 2025/2026 for
yearly results. The CSV also contains monthly metrics. paired_comparison.json
reports candidate-minus-control BA and a paired block interval; status.json says
completed only after both candidates and the paired comparison finish. Candidate
predictions and fit/calibration audits are saved separately. No artifacts or
submission files are replaced. This later period was inspected previously;
do not use this command as a loop for tuning many alternatives on it.
