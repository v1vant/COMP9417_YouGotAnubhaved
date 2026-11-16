"""Leakage-safe features shared by training and live day-ahead prediction."""
import os

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


def feature_frame(frame, weather_shape=False):
    """Current-day weather is allowed; all market features use completed days."""
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
        if weather_shape and col in ("syd_temp", "syd_solar"):
            for offset in (2, 4):  # One and two hours.
                # Clamp neighbours to this market day's first/last value.
                before = values.groupby(days).shift(offset).groupby(days).bfill()
                after = values.groupby(days).shift(-offset).groupby(days).ffill()
                f[f"{col}_before_{offset}"] = before
                f[f"{col}_after_{offset}"] = after
                ramp = values - before
                f[f"{col}_ramp_{offset}"] = ramp
                f[f"{col}_ramp_{offset}_change_1d"] = ramp - ramp.shift(48)
            for stat in ("mean", "min", "max"):
                f[f"{col}_daily_{stat}_change_1d"] = lookup.map(
                    daily[stat] - daily[stat].shift(1))
            peak_slot = values.groupby(days).agg(
                lambda group: int(np.argmax(group.to_numpy())))
            f[f"{col}_peak_slot"] = lookup.map(peak_slot)
            f[f"{col}_peak_slot_change_1d"] = lookup.map(
                peak_slot - peak_slot.shift(1))
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
    # Align observed demand shapes to the forecast slot. Backward offsets
    # may cross midnight, but always remain within completed historical days.
    opening_demand = frame["nsw_demand"].groupby(days).first()
    for lag in (1, 7):
        aligned = f[f"nsw_demand_lag_{lag}"]
        for offset in (1, 2, 4):
            previous = frame["nsw_demand"].shift(48 * lag + offset)
            f[f"nsw_demand_lag_{lag}_before_{offset}"] = previous
            f[f"nsw_demand_lag_{lag}_ramp_{offset}"] = aligned - previous
        opening = lookup.map(opening_demand.shift(lag))
        f[f"nsw_demand_lag_{lag}_change_from_open"] = aligned - opening
    # Fixed-slot profiles: every forecast row sees all 48 historical slots.
    profiles = pd.DataFrame({"day": days, "slot": slot,
                             "demand": frame["nsw_demand"].to_numpy()}).pivot(
                                 index="day", columns="slot", values="demand")
    weekly_profile = sum(profiles.shift(lag) for lag in (7, 14, 21, 28)) / 4
    for profile_slot in range(48):
        for lag in (1, 7):
            f[f"nsw_demand_profile_lag_{lag}_slot_{profile_slot}"] = lookup.map(
                profiles[profile_slot].shift(lag))
        f[f"nsw_demand_profile_weekly_slot_{profile_slot}"] = lookup.map(
            weekly_profile[profile_slot])
    # Four-week averages of the aligned shape features; same-slot mean exists above.
    for offset in (1, 2, 4):
        previous = sum(frame["nsw_demand"].shift(48 * lag + offset)
                       for lag in (7, 14, 21, 28)) / 4
        f[f"nsw_demand_weekly_before_{offset}"] = previous
        f[f"nsw_demand_weekly_ramp_{offset}"] = f["nsw_demand_weekly_mean"] - previous
    weekly_opening = sum(lookup.map(opening_demand.shift(lag))
                         for lag in (7, 14, 21, 28)) / 4
    f["nsw_demand_weekly_change_from_open"] = f["nsw_demand_weekly_mean"] - weekly_opening
    f["price_threshold"] = f["nsw_price_previous_mean"]
    for lag in LAGS:
        f[f"price_margin_{lag}"] = f[f"nsw_price_lag_{lag}"] - f["price_threshold"]
    # Historical price profiles relative to the forecast's previous-day mean.
    price = frame["nsw_price"]
    threshold = f["price_threshold"]
    price_profiles = pd.DataFrame({
        "day": days, "slot": slot, "price": price.to_numpy(),
    }).pivot(index="day", columns="slot", values="price")
    weekly_prices = sum(price_profiles.shift(k) for k in (7, 14, 21, 28)) / 4
    for s in range(48):
        for k in (1, 7):
            f[f"nsw_price_profile_lag_{k}_margin_slot_{s}"] = (
                lookup.map(price_profiles[s].shift(k)) - threshold)
        f[f"nsw_price_profile_weekly_margin_slot_{s}"] = (
            lookup.map(weekly_prices[s]) - threshold)

    opening_price = price.groupby(days).first()
    for k in (1, 7):
        aligned = f[f"nsw_price_lag_{k}"]
        for offset in (1, 2, 4):
            previous = price.shift(48 * k + offset)
            f[f"nsw_price_lag_{k}_before_margin_{offset}"] = previous - threshold
            f[f"nsw_price_lag_{k}_ramp_{offset}"] = aligned - previous
        f[f"nsw_price_lag_{k}_change_from_open"] = (
            aligned - lookup.map(opening_price.shift(k)))

    weekly_aligned = f["nsw_price_weekly_mean"]
    f["nsw_price_weekly_margin"] = weekly_aligned - threshold
    for offset in (1, 2, 4):
        previous = sum(price.shift(48 * k + offset)
                       for k in (7, 14, 21, 28)) / 4
        f[f"nsw_price_weekly_before_margin_{offset}"] = previous - threshold
        f[f"nsw_price_weekly_ramp_{offset}"] = weekly_aligned - previous
    weekly_opening = sum(lookup.map(opening_price.shift(k))
                         for k in (7, 14, 21, 28)) / 4
    f["nsw_price_weekly_change_from_open"] = weekly_aligned - weekly_opening
    f["demand_anchor"] = 0.5 * (f["nsw_demand_lag_1"] + f["nsw_demand_lag_7"])
    return pd.DataFrame(f, index=frame.index, dtype=np.float32)


def training_table(frame, weather_shape=False):
    features = feature_frame(frame, weather_shape=weather_shape)
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


def forecast_features(history, weather_next, weather_shape=False):
    if len(weather_next) != 48 or set(weather_next.columns) != set(WEATHER):
        raise ValueError("Expected 48 rows and the 12 weather columns")
    if history.empty or not history.index.is_monotonic_increasing or not history.index.is_unique:
        raise ValueError("History must be nonempty, ordered and unique")
    expected = pd.date_range(history.index[-1] + pd.Timedelta(minutes=30),
                             periods=48, freq="30min")
    if not weather_next.index.equals(expected) or len(market_days(weather_next.index).unique()) != 1:
        raise ValueError("Weather must be the next complete market day")
    if len(history) < 28 * 48:
        raise ValueError("At least 28 complete history days are required")
    past = history.iloc[-35 * 48:][MARKET + WEATHER]
    future = weather_next.reindex(columns=MARKET + WEATHER)
    features = feature_frame(pd.concat([past, future]), weather_shape=weather_shape).loc[weather_next.index]
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Nonfinite forecast features")
    return features


def predict_task(bundle, features):
    x = features.loc[:, bundle["features"]]
    if bundle["task"] == "demand":
        residual = bundle["estimator"].predict(x)
        return np.clip(features["demand_anchor"].to_numpy() + residual, 0, 20000)
    probability = bundle["estimator"].predict_proba(x)[:, 1]
    return (probability > bundle["threshold"]).astype(np.int64)


class Forecaster:
    def __init__(self, artifact_dir: str):
        self.models = joblib.load(os.path.join(artifact_dir, "models.joblib"))
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
