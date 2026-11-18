# Integration notes: where each piece came from

Branch `integration` (local; no upstream) is based on `origin/feature/demand-mlp`
(f92a38e, Anubhav Ammangi: 405-feature demand / 571-feature price LightGBM, aligned
profile features, 50/50 LightGBM + MLP demand blend). Everything below was copied from other
remote branches with `git cherry-pick -n` (changes staged, not committed) and then
edited by hand where the branches conflicted. Original hashes are given so authorship can
be traced with `git show <hash>`.

## Copied

| What | Source branch | Commit | Author | Where it landed |
| --- | --- | --- | --- | --- |
| Day-index validation (`validate_day_index`), `training_span` (records `training_start`/`training_end` in each model bundle), consecutive-day checks in `simulate_policy`, and tests | `feature_engineering` / `splines` / `random_forest` lineage | 46aadb7 | Suhan Karaca | `code/forecast_package/model.py`, `train.py`, `code/run_experiments.py`, `code/tests/test_forecasting.py` (+ mirrored to `submission_package/`) |
| Holiday-context features (`holiday_yesterday`, `holiday_tomorrow`, `christmas_shutdown`), opt-in via `config["holiday_context"]`; paired ablation script, results and tests | `feature_engineering` / `splines` | d2ba430 | Suhan Karaca | `model.py` (`holiday_context_features`), `train.py`, `code/ablate_holiday_context.py`, `code/tests/test_holiday_context.py`, `results/holiday_ablation/` |
| Mixed Naive Bayes and RBF-kernel SVM price-direction classifiers, with feature-selection scripts | `Thomas` | e690e40 | NotaByzantophile | `model.py` (`NAIVE_BAYES`, `SVM`, `fit_naive_bayes`, `price_probability`, ...), `train.py`, `code/naive_bayes_selection.py`, `code/svm_selection.py`, tests in `test_forecasting.py` |

The holiday context is **opt-in and currently off** in `final_config.json`. The classifiers
are added as price candidates in `candidates()` but are not selected for the final package
by this copy alone: re-run the selection and simulation before relying on them.

## Edited while copying (deviations from the originals)

1. **Dropped the `training_end` refusal in `predict_task` and the legacy-manifest block in
   `Forecaster.__init__`** (both from 46aadb7). They make a package raise when asked to
   forecast any day on or before its training end. The final package is trained through
   31 Oct 2026, and the spec says `check_submission.py` runs it on the last 14 days of the
   dataset, so with that guard every day would count as failed. `training_end` is still
   recorded in the bundle as metadata. The live week (7-13 Dec 2026) is unaffected either way.
2. **`training_table` now drops a leading partial market day.** The aligned features use
   1/2/4 half-hour lags, so the first complete row is mid-day (02:30 on synthetic data).
   `validate_day_index` correctly rejects that, so the partial day (at most 47 rows from 2010)
   is trimmed. This slightly changes the training set, so artifacts are no longer
   bit-identical to the ones built on `feature/demand-mlp` until retrained.
3. **Tests adapted.** 46aadb7's tests used the `random_forest` branch's
   `candidates(..., families=...)` option and RandomForest tests, which do not exist here.
   They were rewritten to use the boosted/linear candidates (and stub candidates in the
   selection test, because NB/SVM have their own scoring path). The RandomForest-specific
   tests were not copied.
4. `submission_package/model.py` and `train.py` were re-synced as exact copies of
   `forecast_package/`, as they were identical before.

## Not copied (deliberately)

- **`random_forest` branch** (Suhan Karaca, 96317b6 lineage): not needed for the current
  families; 46aadb7 depends on it, hence the manual resolution above.
- **Solar-trend features** (71b3267, 8b72061): overall demand MAPE gain only -0.03 pp, worse on
  holidays and on "similar solar" middays. Marginal; left out.
- **Spline Ridge demand model** (`splines`, 12a3e40, 504d1b5): a separate demand model, not a
  feature change; no integration decision made.
- **`feature/price-regression`** (Anubhav): its `weather_shape` / aligned-profile code is
  already in the base branch (same code as `feature/aligned-demand-features`).
- **Regenerated results / artifacts** from `Thomas` (65d7871) and from other branches: generated
  files, rebuild with `python code/run_project.py` instead.

## Status when this note was written

- `python -m unittest discover -s code/tests`: 30 tests, all pass.
- `submission_package` `Forecaster` still produces a valid 48-row prediction with the
  existing artifacts on the last days of the released CSV.
- Nothing committed; nothing run through the experiments or `train.py` yet.
