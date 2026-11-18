"""Price-only, causal context. No forecast-day market observations are used."""
import numpy as np
import pandas as pd

SETS = ('base', 'geometry', 'recency', 'both')


def price_context_frame(frame, feature_set='both'):
    from model import validate_day_index, market_days, LAGS
    if feature_set not in SETS:
        raise ValueError(f'Unknown price feature set: {feature_set}')
    validate_day_index(frame.index)
    if feature_set == 'base':
        return pd.DataFrame(index=frame.index)
    days = market_days(frame.index).unique()
    p = pd.DataFrame(frame.nsw_price.to_numpy(dtype=float).reshape(-1, 48), index=days)
    prev = p.shift(1)
    mean = p.mean(axis=1)
    tau = mean.shift(1)
    median = prev.median(axis=1)
    scale = (prev.quantile(.75, axis=1) - prev.quantile(.25, axis=1)).clip(lower=1.)
    geometry, recency = {}, {}
    def daily(v):
        return np.repeat(v.to_numpy(), 48)
    for lag in LAGS:
        geometry[f'pc_margin_{lag}'] = p.shift(lag).sub(tau, axis=0).div(scale, axis=0).to_numpy().ravel()
    geometry['pc_skew'] = daily((tau - median) / scale)
    trimmed = pd.Series(np.sort(prev.to_numpy(), axis=1)[:, 4:-4].mean(axis=1), index=days)
    geometry['pc_tail'] = daily((tau - trimmed) / scale)
    geometry['pc_std_ratio'] = daily(prev.std(axis=1) / scale)
    geometry['pc_spike_fraction'] = daily(prev.gt(median + 3 * scale, axis=0).mean(axis=1).where(prev.notna().all(axis=1)))
    geometry['pc_negative_fraction'] = daily((prev < 0).mean(axis=1).where(prev.notna().all(axis=1)))
    lagged = np.stack([p.shift(j).to_numpy() for j in range(1, 29)])
    exceed = lagged > tau.to_numpy()[None, :, None]
    for k in (7, 28):
        value = exceed[:k].mean(axis=0)
        value[~np.isfinite(lagged[:k]).all(axis=(0, 2))] = np.nan
        recency[f'pc_exceed_{k}'] = value.ravel()
    all_rate = exceed.mean(axis=(0, 2)).astype(float)
    all_rate[~np.isfinite(lagged).all(axis=(0, 2))] = np.nan
    recency['pc_exceed_all_28'] = np.repeat(all_rate, 48)
    local = np.column_stack([prev.iloc[:, max(0, t-1):min(48, t+2)].mean(axis=1) for t in range(48)])
    recency['pc_local_margin'] = ((local - tau.to_numpy()[:, None]) / scale.to_numpy()[:, None]).ravel()
    recency['pc_late_change'] = daily((prev.iloc[:, -8:].mean(axis=1) - prev.iloc[:, :8].mean(axis=1)) / scale)
    recency['pc_mean_momentum'] = daily((tau - mean.shift(2)) / scale)
    recency['pc_weekly_anomaly'] = daily((tau - mean.shift(1).rolling(7).median()) / scale)
    values = geometry if feature_set == 'geometry' else recency if feature_set == 'recency' else {**geometry, **recency}
    return pd.DataFrame(values, index=frame.index, dtype=np.float32)


def extend_price_features(frame, base, feature_set='base'):
    return pd.concat([base, price_context_frame(frame, feature_set).loc[base.index]], axis=1)


def forecast_price_features(history, weather_next, base, feature_set='base'):
    if feature_set == 'base':
        return base
    from model import MARKET, WEATHER
    frame = pd.concat([history.iloc[-35*48:][MARKET + WEATHER],
                       weather_next.reindex(columns=MARKET + WEATHER)])
    result = extend_price_features(frame, base, feature_set)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('Nonfinite price context')
    return result
