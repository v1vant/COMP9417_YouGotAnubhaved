"""Leakage-safe features shared by training and live day-ahead prediction."""
import os
import json

import holidays
import joblib
import numpy as np
import pandas as pd

MARKET = [f"{state}_{kind}" for state in ("nsw", "vic", "qld")
          for kind in ("demand", "price")]
WEATHER = [f"{city}_{kind}" for city in ("syd", "mel", "bne")
           for kind in ("temp", "rh", "cloud", "solar")]
LAGS = (1, 2, 3, 7, 14, 21, 28)


def market_days(index):
    return (index - pd.Timedelta(minutes=30)).normalize()


def validate_day_index(index):
    """Row lags are day lags only on consecutive, complete market days."""
    if (not isinstance(index, pd.DatetimeIndex) or index.empty or index.hasnans
            or not index.is_unique or not index.is_monotonic_increasing):
        raise ValueError("Expected nonempty, ordered, unique timestamps")
    expected = pd.date_range(index[0], index[-1], freq="30min")
    if (not index.equals(expected)
            or index[0] != market_days(index[:1])[0] + pd.Timedelta(minutes=30)
            or index[-1] != market_days(index[-1:])[0] + pd.Timedelta(days=1)):
        raise ValueError("Expected consecutive complete market days")


def training_span(features, truth):
    if not features.index.equals(truth.index):
        raise ValueError("Features and targets must have identical training indices")
    validate_day_index(features.index)
    days = market_days(features.index)
    return {"training_start": str(days.min()), "training_end": str(days.max())}


def load_data(path):
    frame = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"]).sort_index()
    if frame.empty or not frame.index.is_unique:
        raise ValueError("Empty data or duplicate timestamps")
    expected = pd.date_range(frame.index[0], frame.index[-1], freq="30min")
    if not frame.index.equals(expected) or frame[MARKET + WEATHER].isna().any().any():
        raise ValueError("Expected complete consecutive half-hour observations")
    counts = pd.Series(1, index=frame.index).groupby(market_days(frame.index)).sum()
    if not counts.eq(48).all():
        raise ValueError("Each market day must contain 48 intervals")
    return frame


def feature_frame(frame):
    """Current-day weather is allowed; all market features use completed days."""
    validate_day_index(frame.index)
    days = market_days(frame.index)
    start = frame.index - pd.Timedelta(minutes=30)
    slot = np.asarray(start.hour * 2 + start.minute // 30)
    calendar = holidays.Australia(
        subdiv="NSW", years=range(days.min().year, days.max().year + 1))
    holiday_map = {day: float(day.date() in calendar) for day in days.unique()}
    f = {
        "slot": slot,
        "weekday": np.asarray(days.day_of_week),
        "month": np.asarray(days.month),
        "weekend": np.asarray(days.day_of_week >= 5, dtype=float),
        "holiday": np.asarray([holiday_map[day] for day in days]),
        "hour_sin": np.sin(2 * np.pi * slot / 48),
        "hour_cos": np.cos(2 * np.pi * slot / 48),
        "annual_sin": np.sin(2 * np.pi * days.day_of_year / 365.25),
        "annual_cos": np.cos(2 * np.pi * days.day_of_year / 365.25),
    }
    for weekday in range(7):
        indicator = np.asarray(days.day_of_week == weekday, dtype=float)
        f[f"dow_{weekday}"] = indicator
        f[f"dow_{weekday}_sin"] = indicator * f["hour_sin"]
        f[f"dow_{weekday}_cos"] = indicator * f["hour_cos"]
    lookup = pd.Series(days, index=frame.index)
    for col in WEATHER:
        values = frame[col]
        f[col] = values
        f[f"{col}_change_1d"] = values - values.shift(48)
        f[f"{col}_change_7d"] = values - values.shift(336)
        daily = values.groupby(days).agg(["mean", "min", "max"])
        for stat in daily:
            f[f"{col}_today_{stat}"] = lookup.map(daily[stat])
    for city in ("syd", "mel", "bne"):
        temp = frame[f"{city}_temp"]
        for label, values in (
            ("heat", (18 - temp).clip(lower=0)),
            ("cool", (temp - 22).clip(lower=0)),
        ):
            f[f"{city}_{label}"] = values
            f[f"{city}_{label}_sin"] = values * f["hour_sin"]
            f[f"{city}_{label}_cos"] = values * f["hour_cos"]
    for col in MARKET:
        for lag in LAGS:
            lagged = frame[col].shift(48 * lag)
            f[f"{col}_lag_{lag}"] = lagged
            if col.endswith("price"):
                f[f"{col}_loglag_{lag}"] = np.sign(lagged) * np.log1p(np.abs(lagged))
        daily = frame[col].groupby(days).agg(["mean", "min", "max", "std"])
        for stat in daily:
            f[f"{col}_previous_{stat}"] = lookup.map(daily[stat].shift(1))
        for window in (7, 28):
            f[f"{col}_mean_{window}d"] = lookup.map(
                daily["mean"].shift(1).rolling(window, min_periods=window).mean())
        f[f"{col}_weekly_mean"] = sum(f[f"{col}_lag_{lag}"] for lag in (7, 14, 21, 28)) / 4
        f[f"{col}_recent_change"] = f[f"{col}_lag_1"] - f[f"{col}_lag_7"]
    f["price_threshold"] = f["nsw_price_previous_mean"]
    for lag in LAGS:
        f[f"price_margin_{lag}"] = f[f"nsw_price_lag_{lag}"] - f["price_threshold"]
    f["demand_anchor"] = 0.5 * (f["nsw_demand_lag_1"] + f["nsw_demand_lag_7"])
    return pd.DataFrame(f, index=frame.index, dtype=np.float32)


def training_table(frame):
    features = feature_frame(frame)
    keep = features.notna().all(axis=1)
    features = features.loc[keep]
    # Targets retain the original precision for the strict price comparison.
    means = frame["nsw_price"].groupby(market_days(frame.index)).mean().shift(1)
    threshold = pd.Series(market_days(features.index), index=features.index).map(means)
    truth = pd.DataFrame({
        "demand": frame.loc[keep, "nsw_demand"],
        "price": (frame.loc[keep, "nsw_price"] > threshold).astype(int),
    }, index=features.index)
    return features, truth


def forecast_features(history, weather_next):
    if len(weather_next) != 48 or set(weather_next.columns) != set(WEATHER):
        raise ValueError("Expected 48 rows and the 12 weather columns")
    if history.empty or not history.index.is_monotonic_increasing or not history.index.is_unique:
        raise ValueError("History must be nonempty, ordered and unique")
    validate_day_index(history.index)
    expected = pd.date_range(history.index[-1] + pd.Timedelta(minutes=30),
                             periods=48, freq="30min")
    if not weather_next.index.equals(expected) or len(market_days(weather_next.index).unique()) != 1:
        raise ValueError("Weather must be the next complete market day")
    if len(history) < 28 * 48:
        raise ValueError("At least 28 complete history days are required")
    past = history.iloc[-35 * 48:][MARKET + WEATHER]
    future = weather_next.reindex(columns=MARKET + WEATHER)
    features = feature_frame(pd.concat([past, future])).loc[weather_next.index]
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Nonfinite forecast features")
    return features


def holiday_context_features(features):
    """Calendar-only context; independent of batch length and market outcomes."""
    days = market_days(features.index)
    calendar = holidays.Australia(
        subdiv="NSW", years=range(days.min().year - 1, days.max().year + 2))
    extra = {}
    for name, offset in (("holiday_yesterday", -1), ("holiday_tomorrow", 1)):
        values = {day: float((day + pd.Timedelta(days=offset)).date() in calendar)
                  for day in days.unique()}
        extra[name] = [values[day] for day in days]
    extra["christmas_shutdown"] = ((days.month == 12) & (days.day >= 20)) | (
        (days.month == 1) & (days.day <= 5))
    return pd.concat([features, pd.DataFrame(extra, index=features.index,
                                            dtype=np.float32)], axis=1)


def predict_task(bundle, features):
    if (bundle.get("training_end") is not None
            and (market_days(features.index) <= pd.Timestamp(bundle["training_end"])).any()):
        raise ValueError("Forecast days must be after the model's training_end; use chronological backtests")
    if bundle.get("holiday_context"):
        features = holiday_context_features(features)
    x = features.loc[:, bundle["features"]]
    if bundle["task"] == "demand":
        residual = bundle["estimator"].predict(x)
        return np.clip(features["demand_anchor"].to_numpy() + residual, 0, 20000)
    probability = bundle["estimator"].predict_proba(x)[:, 1]
    return (probability > bundle["threshold"]).astype(np.int64)


class Forecaster:
    def __init__(self, artifact_dir: str):
        self.models = joblib.load(os.path.join(artifact_dir, "models.joblib"))
        # Older artifacts store provenance only in the companion manifest.
        manifest_path = os.path.join(artifact_dir, "manifest.json")
        manifest = None
        for task, bundle in self.models.items():
            if bundle.get("training_end") is None:
                if manifest is None:
                    if not os.path.exists(manifest_path):
                        raise ValueError("Missing training provenance; rebuild artifacts before forecasting")
                    with open(manifest_path, encoding="utf-8") as handle:
                        manifest = json.load(handle)
                end = pd.Timestamp(manifest["data_end"])
                cutoff = manifest.get("selection", {}).get(task, {}).get("training_end")
                if cutoff:
                    end = min(end, pd.Timestamp(cutoff))
                bundle["training_end"] = str(end)
        for bundle in self.models.values():
            estimator = bundle["estimator"]
            if "n_jobs" in estimator.get_params():
                estimator.set_params(n_jobs=2)

    def predict(self, history: pd.DataFrame, weather_next: pd.DataFrame) -> pd.DataFrame:
        features = forecast_features(history, weather_next)
        return pd.DataFrame({
            "nsw_demand_mw": predict_task(self.models["demand"], features),
            "nsw_price_up": predict_task(self.models["price"], features),
        }, index=weather_next.index)
