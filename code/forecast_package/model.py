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


def feature_frame(frame):
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


# ---- Mixed Naive Bayes (price direction) -------------------------------------------
# Discrete features use frequency tables, continuous ones use bell curves (or a hurdle
# likelihood when they are often exactly 0). The fitted model is a dict of plain numpy
# arrays, so loading it needs no custom class.

# Number of possible values per discrete feature (month is 1-12, so index 0 is unused).
NB_CATEGORIES = {"weekday": 7, "holiday": 2, "slot": 48, "weekend": 2, "month": 13,
                 **{f"dow_{day}": 2 for day in range(7)}}
NB_ZERO_SHARE = 0.10   # continuous features exactly 0 this often get a hurdle likelihood

# Features chosen by code/naive_bayes_selection.py: fit 2010-2012, every choice made on
# 2013 (correlation screen |r| < 0.4 ranked by stand-alone balanced accuracy, then impact
# pruning to >= 0.1 points each; price_margin_7 kept manually). Test on 2014: 78.06 BA.
NAIVE_BAYES = {
    "family": "naive_bayes",
    "discrete": ["slot", "weekend"],
    "continuous": [
        "price_margin_1", "price_margin_7", "price_threshold",          # price vs. threshold
        "qld_demand_recent_change", "vic_demand_previous_min",          # interstate demand
        "vic_price_recent_change",                                      # interstate price
        "dow_0_sin", "dow_0_cos", "dow_5_sin", "dow_6_sin",             # weekday daily shapes
        "syd_temp_anomaly", "mel_cool_sin", "mel_solar",                # weather
        "mel_cloud_today_mean", "bne_temp_change_7d", "mel_temp_change_7d",
    ],
    "alpha": 1.0,   # Laplace smoothing for the frequency tables
}


def seasonal_basis(features):
    """Smooth time-of-year x time-of-day terms used to model the 'normal' temperature."""
    a_s, a_c, h_s, h_c = (features[col].to_numpy(np.float64)
                          for col in ("annual_sin", "annual_cos", "hour_sin", "hour_cos"))
    return np.column_stack([np.ones_like(a_s), a_s, a_c, h_s, h_c,
                            a_s * h_s, a_s * h_c, a_c * h_s, a_c * h_c])


def naive_bayes_continuous(params, features):
    """Matrix of continuous NB inputs, including the derived temperature anomaly."""
    columns = []
    for name in params["continuous"]:
        if name == "syd_temp_anomaly":
            values = (features["syd_temp"].to_numpy(np.float64)
                      - seasonal_basis(features) @ params["temp_normal"])
        elif "price" in name and "loglag" not in name:
            raw = features[name].to_numpy(np.float64)   # raw $/MWh values have huge spikes:
            values = np.sign(raw) * np.log1p(np.abs(raw))  # signed log makes them bell-shaped
        else:
            values = features[name].to_numpy(np.float64)
        columns.append(values)
    return np.column_stack(columns) if columns else np.empty((len(features), 0))


def fit_naive_bayes(config, features, labels):
    """Closed-form fit: class prior, a frequency table per discrete feature, and a mean and
    variance per continuous feature, all separately for 'down' (0) and 'up' (1).
    Used by train.py only; Forecaster never calls it."""
    def available(name):  # derived/ablated features need their raw inputs present
        return ("syd_temp" if name == "syd_temp_anomaly" else name) in features
    y = labels.to_numpy().astype(np.int64)
    params = {"discrete": [n for n in config["discrete"] if available(n)],
              "continuous": [n for n in config["continuous"] if available(n)],
              "temp_normal": None}
    if "syd_temp_anomaly" in params["continuous"]:
        # 'Normal' temperature = least-squares fit on season x time-of-day terms.
        params["temp_normal"] = np.linalg.lstsq(
            seasonal_basis(features), features["syd_temp"].to_numpy(np.float64), rcond=None)[0]
    counts = np.bincount(y, minlength=2)
    params["log_prior"] = np.log(counts / counts.sum())
    x = naive_bayes_continuous(params, features)
    zero = x == 0
    # Features that are exactly 0 much of the time (heating/cooling, night-time solar) get a
    # separate probability of being 0, and a bell curve fitted to the non-zero values only.
    params["zero_inflated"] = zero.mean(axis=0) >= NB_ZERO_SHARE
    mean, variance, p_zero = (np.zeros((2, x.shape[1])) for _ in range(3))
    for k in (0, 1):
        xk, zk = x[y == k], zero[y == k]
        for j in range(x.shape[1]):
            values = xk[~zk[:, j], j] if params["zero_inflated"][j] else xk[:, j]
            if len(values) >= 2:
                mean[k, j], variance[k, j] = values.mean(), values.var()
            else:
                mean[k, j], variance[k, j] = 0.0, 1.0
            p_zero[k, j] = (zk[:, j].sum() + 1) / (len(zk) + 2)   # Laplace-smoothed
    params["mean"], params["p_zero"] = mean, p_zero
    params["var"] = variance + 1e-9 * (variance.max() if variance.size else 1.0)
    params["log_table"] = []
    for name in params["discrete"]:
        codes = features[name].to_numpy().astype(np.int64)
        table = np.array([np.bincount(codes[y == k], minlength=NB_CATEGORIES[name])
                          for k in (0, 1)]) + config["alpha"]
        params["log_table"].append(np.log(table / table.sum(axis=1, keepdims=True)))
    return params


def naive_bayes_terms(params, features):
    """Per-feature log-likelihoods log P(x_j | class), shape (rows, features, 2 classes)."""
    x = naive_bayes_continuous(params, features)[:, :, None]
    mean, var = params["mean"].T[None], params["var"].T[None]          # (1, features, 2)
    gaussian = -0.5 * np.log(2 * np.pi * var) - (x - mean) ** 2 / (2 * var)
    inflated = params.get("zero_inflated")
    if inflated is not None and inflated.any():
        # Hurdle likelihood: P(x=0|c) = pi_c, otherwise (1 - pi_c) * Normal(x | non-zero fit).
        p_zero = params["p_zero"].T[None]
        hurdle = np.where(x == 0, np.log(p_zero), np.log1p(-p_zero) + gaussian)
        gaussian = np.where(inflated[None, :, None], hurdle, gaussian)
    categorical = [table[:, features[name].to_numpy().astype(np.int64)].T
                   for name, table in zip(params["discrete"], params["log_table"])]
    stacked = np.stack(categorical, axis=1) if categorical else np.empty((len(features), 0, 2))
    return np.concatenate([stacked, gaussian], axis=1)


def naive_bayes_probability(params, features):
    """P(up | x) = prior x product of per-feature likelihoods, normalised over the classes."""
    log_joint = params["log_prior"] + naive_bayes_terms(params, features).sum(axis=1)
    log_joint -= log_joint.max(axis=1, keepdims=True)                  # numerical stability
    joint = np.exp(log_joint)
    return joint[:, 1] / joint.sum(axis=1)


# ---- Support vector machine (price direction) --------------------------------------
# Kernel (Gaussian radial basis function), on standardised svm_inputs x in R^218:
#
#     K(x, x') = exp(-gamma * ||x - x'||^2),   gamma = 1.0 / 218 ~ 0.00459
#
# Decision value for a half-hour x (sign = side of the boundary; |f| = distance):
#
#     f(x) = sum_{i in support vectors} alpha_i * y_i * K(x, x_i) + b,   y_i in {-1, +1}
#
# with 0 <= alpha_i <= C_{y_i}, C = 0.3 rescaled per class by class_weight="balanced"
# (C_c = C * n / (2 * n_c)). alpha and b are learned; gamma and C are not.
# Prediction: nsw_price_up = 1 if 1 / (1 + exp(-f(x))) > tuned threshold, else 0.
#
# Exact RBF-kernel SVM on all 218 svm_columns. gamma (in units of 1/#features) and C are
# hyperparameters chosen on validation data, not learned by training.
#
# SVM_FEATURES is NOT used: it is the 76-feature set from code/svm_selection.py (fit
# 2010-2012, permutation-importance backward elimination on 2013), kept for the report.
# It tied all features on a clean 2014 test (80.40 vs 80.33) but lost in the production
# setup (fit 2010-2013 on 20k rows, tuned on 2014: 81.01 vs 81.45), so all features are used.
SVM_FEATURES = [
    "month", "weekend", "holiday", "dow_0_cos", "dow_2", "dow_2_sin", "dow_2_cos",
    "dow_3", "dow_4", "syd_temp_change_7d", "syd_rh_change_1d", "syd_rh_change_7d",
    "syd_cloud_change_1d", "syd_cloud_today_max", "syd_solar_change_1d",
    "syd_solar_change_7d", "syd_solar_today_mean", "mel_temp", "mel_temp_change_1d",
    "mel_temp_today_min", "mel_rh_change_1d", "mel_rh_change_7d", "mel_cloud_change_7d",
    "mel_solar_change_1d", "mel_solar_change_7d", "mel_solar_today_min",
    "bne_temp_today_mean", "bne_temp_today_min", "bne_rh", "bne_rh_today_max",
    "bne_cloud_change_1d", "bne_cloud_today_mean", "bne_cloud_today_min", "bne_solar",
    "bne_solar_today_mean", "syd_cool", "syd_cool_sin", "mel_heat_sin", "mel_cool",
    "mel_cool_sin", "mel_cool_cos", "bne_cool_sin", "nsw_demand_lag_3",
    "nsw_demand_lag_21", "nsw_demand_previous_min", "nsw_demand_previous_max",
    "nsw_demand_mean_7d", "nsw_demand_weekly_mean", "nsw_demand_recent_change",
    "nsw_price_previous_max", "nsw_price_mean_7d", "nsw_price_mean_28d",
    "vic_demand_lag_2", "vic_demand_lag_28", "vic_demand_previous_std",
    "vic_demand_mean_7d", "vic_price_previous_min", "vic_price_previous_max",
    "vic_price_previous_std", "qld_demand_lag_2", "qld_demand_lag_3",
    "qld_demand_lag_7", "qld_demand_lag_28", "qld_demand_mean_7d",
    "qld_demand_recent_change", "qld_price_loglag_28", "qld_price_previous_std",
    "qld_price_mean_28d", "qld_price_weekly_mean", "price_margin_1", "price_margin_2",
    "price_margin_3", "price_margin_7", "price_margin_14", "price_margin_21",
    "price_margin_28",
]
SVM = {"family": "svm", "kernel": "rbf", "gamma": 1.0, "C": 0.3, "max_rows": 20000,
       "features": None}   # None = every column from svm_columns (218)


def svm_columns(features):
    """All features except those that become exact duplicates after the signed log
    (raw price lags = their loglag; price_threshold = nsw_price_previous_mean)."""
    return [col for col in features.columns
            if "_price_lag_" not in col and col != "price_threshold"]


def svm_inputs(columns, features):
    """SVM input matrix: raw $/MWh features are signed-logged so that a single price spike
    cannot dominate the kernel's distances. Scaling happens inside the fitted pipeline."""
    x = features.loc[:, columns].to_numpy(np.float64, copy=True)
    for j, name in enumerate(columns):
        if "price" in name and "loglag" not in name:
            x[:, j] = np.sign(x[:, j]) * np.log1p(np.abs(x[:, j]))
    return x


def price_probability(bundle, features):
    if bundle["config"]["family"] == "naive_bayes":
        return naive_bayes_probability(bundle["estimator"], features)
    if bundle["config"]["family"] == "svm":
        # SVMs output a signed distance to the boundary, not a probability. A sigmoid maps
        # it to (0, 1) monotonically so the shared threshold search works unchanged; the
        # result is a score, not a calibrated probability.
        score = bundle["estimator"].decision_function(svm_inputs(bundle["features"], features))
        return 1 / (1 + np.exp(-score))
    return bundle["estimator"].predict_proba(features.loc[:, bundle["features"]])[:, 1]


def predict_task(bundle, features):
    if bundle["task"] == "demand":
        residual = bundle["estimator"].predict(features.loc[:, bundle["features"]])
        return np.clip(features["demand_anchor"].to_numpy() + residual, 0, 20000)
    probability = price_probability(bundle, features)
    return (probability > bundle["threshold"]).astype(np.int64)


class Forecaster:
    def __init__(self, artifact_dir: str):
        self.models = joblib.load(os.path.join(artifact_dir, "models.joblib"))
        for bundle in self.models.values():
            estimator = bundle["estimator"]
            if hasattr(estimator, "get_params") and "n_jobs" in estimator.get_params():
                estimator.set_params(n_jobs=2)

    def predict(self, history: pd.DataFrame, weather_next: pd.DataFrame) -> pd.DataFrame:
        features = forecast_features(history, weather_next)
        return pd.DataFrame({
            "nsw_demand_mw": predict_task(self.models["demand"], features),
            "nsw_price_up": predict_task(self.models["price"], features),
        }, index=weather_next.index)
