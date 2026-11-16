"""Standalone sparse spline + Ridge demand experiment; run with --help.

Uses the shared leakage-safe features and demand anchor. This is a demand-only
artifact, not a complete submission package (which also needs price predictions).
"""
import argparse
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from model import (forecast_features, holiday_context_features, load_data,
                   market_days, training_span, training_table)


class SplineRidgeDemand:
    """Additive cubic splines with selected interactions and an L2 penalty."""

    weather_columns = ["syd_temp", "syd_solar", "syd_rh", "syd_cloud",
                       "syd_temp_change_1d", "syd_solar_change_1d"]

    def __init__(self, alpha=100.0, n_knots=6):
        self.alpha = alpha
        self.n_knots = n_knots

    @staticmethod
    def _periodic(period, knots):
        return SplineTransformer(
            degree=3, knots=np.linspace(0, period, knots + 1)[:, None],
            extrapolation="periodic", include_bias=True, sparse_output=True)

    @staticmethod
    def _calendar(features):
        days = market_days(features.index)
        # Fixed calendar boundaries, including leap years; no fitted date range.
        year_length = np.where(days.is_leap_year, 366.0, 365.0)
        return np.asarray((days.day_of_year - 1) / year_length).reshape(-1, 1)

    def _design(self, features, fit=False):
        x = holiday_context_features(features)
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError("Nonfinite demand features")
        if fit:
            self.linear_columns_ = [c for c in x if (
                c.startswith("nsw_demand_") or c.startswith("dow_")
                or c.startswith("syd_") or c in (
                    "demand_anchor", "weekend", "holiday", "holiday_yesterday",
                    "holiday_tomorrow", "christmas_shutdown"))]
            self.linear_scaler_ = StandardScaler()
            self.weather_ = SplineTransformer(
                degree=3, n_knots=self.n_knots, knots="uniform",
                extrapolation="linear", include_bias=False, sparse_output=True)
            self.slot_ = self._periodic(48, 12)
            self.annual_ = self._periodic(1, 8)
        method = "fit_transform" if fit else "transform"
        linear = getattr(self.linear_scaler_, method)(x[self.linear_columns_])
        weather = getattr(self.weather_, method)(x[self.weather_columns]).tocsr()
        slot = getattr(self.slot_, method)(x[["slot"]]).tocsr()
        annual = getattr(self.annual_, method)(self._calendar(x)).tocsr()
        blocks = [sparse.csr_matrix(linear), weather, slot, annual]
        # Temperature/solar responses can vary across the day. Only two
        # tensor products are added, rather than crossing every input feature.
        width = weather.shape[1] // len(self.weather_columns)
        for feature_number in (0, 1):
            basis = weather[:, feature_number * width:(feature_number + 1) * width]
            blocks.extend(basis.multiply(slot[:, [j]]) for j in range(slot.shape[1]))
        for flag in ("weekend", "holiday", "christmas_shutdown"):
            blocks.append(slot.multiply(x[flag].to_numpy()[:, None]))
        design = sparse.hstack(blocks, format="csr", dtype=np.float32)
        if fit:
            self.design_scaler_ = StandardScaler(with_mean=False)
        return getattr(self.design_scaler_, method)(design)

    def fit(self, features, truth):
        if not np.isfinite(self.alpha) or self.alpha <= 0 or self.n_knots < 2:
            raise ValueError("alpha must be positive and n_knots at least two")
        span = training_span(features, truth)
        actual = truth["demand"].to_numpy()
        if not np.isfinite(actual).all() or (actual <= 0).any():
            raise ValueError("Demand targets must be finite and positive")
        started = time.perf_counter()
        design = self._design(features, fit=True)
        self.estimator_ = Ridge(alpha=self.alpha, solver="lsqr", tol=1e-4,
                                max_iter=2000)
        weights = 1 / actual
        self.estimator_.fit(design, actual - features["demand_anchor"].to_numpy(),
                            sample_weight=weights / weights.mean())
        if np.max(self.estimator_.n_iter_) >= self.estimator_.max_iter:
            warnings.warn("Ridge reached its iteration limit; inspect validation scores")
        self.training_start_ = span["training_start"]
        self.training_end_ = span["training_end"]
        self.fit_seconds_ = time.perf_counter() - started
        self.design_columns_ = design.shape[1]
        return self

    def predict(self, features):
        if (market_days(features.index) <= pd.Timestamp(self.training_end_)).any():
            raise ValueError("Forecast days must be after training_end")
        residual = self.estimator_.predict(self._design(features))
        return np.clip(features["demand_anchor"].to_numpy() + residual, 0, 20000)

    def predict_day(self, history, weather_next):
        """Return a timestamp-indexed demand Series for the next complete day."""
        features = forecast_features(history, weather_next)
        return pd.Series(self.predict(features), index=features.index,
                         name="nsw_demand_mw")


def metrics(actual, predicted):
    error = np.abs(np.asarray(actual) - predicted)
    return {"MAPE": float(np.mean(error / np.asarray(actual)) * 100),
            "MAE": float(error.mean())}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", default=str(root / "data/nem_weather_halfhourly_2010_2026.csv"))
    parser.add_argument("--output", type=Path, default=root / "results/spline_ridge")
    parser.add_argument("--quick", action="store_true", help="Try one configuration instead of six; final fit still uses all data")
    parser.add_argument("--evaluate", action="store_true", help="Also run a 2015-2024 expanding-history development backtest")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("Output directory must be empty; choose a new --output path")
    started = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        frame = load_data(args.data)
        features, truth = training_table(frame)
        days = market_days(features.index)
        train, valid = days.year <= 2013, days.year == 2014
        if not train.any() or not valid.any():
            parser.error("Data must contain 2010-2013 training days and 2014 validation days")
        scores = []
        for knots in ([6] if args.quick else [4, 6]):
            for alpha in ([100.0] if args.quick else [10.0, 100.0, 1000.0]):
                model = SplineRidgeDemand(alpha, knots).fit(features.loc[train], truth.loc[train])
                row = {"alpha": alpha, "n_knots": knots,
                       **metrics(truth.loc[valid, "demand"], model.predict(features.loc[valid])),
                       "fit_seconds": model.fit_seconds_, "design_columns": model.design_columns_}
                scores.append(row)
                print(f"2014 validation: {row}", flush=True)
        pd.DataFrame(scores).to_csv(args.output / "validation.csv", index=False)
        best = min(scores, key=lambda row: row["MAPE"])
        config = {key: best[key] for key in ("alpha", "n_knots")}
        if args.evaluate:
            predictions, fits = [], []
            # Matches the saved expanding-history LightGBM development schedule.
            for start, stop in [(2015, 2015), (2016, 2017), (2018, 2019),
                                (2020, 2021), (2022, 2023), (2024, 2024)]:
                fitting = days.year < start
                scoring = (days.year >= start) & (days.year <= stop)
                if not scoring.any():
                    continue
                model = SplineRidgeDemand(**config).fit(features.loc[fitting], truth.loc[fitting])
                prediction = model.predict(features.loc[scoring])
                predictions.append(pd.DataFrame({"actual": truth.loc[scoring, "demand"],
                                                 "prediction": prediction}))
                fits.append({"effective_year": start, "training_end": model.training_end_,
                             "rows": int(fitting.sum()), "fit_seconds": model.fit_seconds_})
                print(f"Development {start}-{stop}: {metrics(truth.loc[scoring, 'demand'], prediction)}", flush=True)
            if not predictions:
                parser.error("--evaluate requires development data from 2015-2024")
            evaluated = pd.concat(predictions)
            evaluated.to_parquet(args.output / "development_predictions.parquet")
            pd.DataFrame(fits).to_csv(args.output / "fits.csv", index=False)
            reports = [{"period": "development", **metrics(evaluated.actual, evaluated.prediction)}]
            for year, group in evaluated.groupby(market_days(evaluated.index).year):
                reports.append({"period": str(year), **metrics(group.actual, group.prediction)})
            pd.DataFrame(reports).to_csv(args.output / "development_metrics.csv", index=False)
        print("Fitting selected configuration on all supplied data...", flush=True)
        model = SplineRidgeDemand(**config).fit(features, truth)
        joblib.dump(model, args.output / "demand.joblib", compress=3)
        manifest = {"model": "spline_ridge_demand", "config": config,
                    "data": str(Path(args.data).resolve()), "rows": len(features),
                    "training_start": model.training_start_, "training_end": model.training_end_,
                    "final_fit_seconds": model.fit_seconds_, "total_seconds": time.perf_counter() - started,
                    "design_columns": model.design_columns_, "validation_year": 2014,
                    "development_evaluated": args.evaluate, "holdout_evaluated": False}
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Saved demand model to {args.output}; total {manifest['total_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    # Store an importable class name in joblib even when invoked as a script.
    from spline_ridge import main as run
    run()
