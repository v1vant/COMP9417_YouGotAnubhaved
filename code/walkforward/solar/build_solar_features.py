"""Pure test (not in the repo): solar-aware inputs for the demand forecaster (fix b).
Rows = the harness cache (market days 2010-01-01 .. 2024-12-31); the 2025-26 holdout is never read.

The demand model predicts demand - anchor, anchor = 0.5 * (same slot yesterday + same slot a week ago). With rooftop PV
growing every year, the demand response to a change in sunshine grows too, which a model fitted on older years cannot
know. These features give it a recent, past-only estimate of that response:
  sens_beta_solar / sens_beta_temp : OLS of daily midday (10:00-16:00) mean demand on midday mean Sydney irradiance and
                                     temperature over the previous 56 market days (strictly before the target day; >= 30 days)
  sens_midday_dip_28d              : mean over the previous 28 days of midday demand / evening (16:00-21:00) demand
  sens_x_dsolar_1d / _7d / _anchor : sens_beta_solar x (target-slot irradiance minus the same slot 1 day / 7 days ago /
                                     the anchor days' average) -- the expected solar effect on the residual
Day D (the day before the target) is complete at forecast time, exactly as the existing lag-1 demand features assume."""
import json, os, sys
E = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, E)
import numpy as np, pandas as pd

WINDOW, MIN_DAYS, DIP = 56, 30, 28
NAMES = ["sens_beta_solar", "sens_beta_temp", "sens_midday_dip_28d", "sens_x_dsolar_1d", "sens_x_dsolar_7d",
         "sens_x_dsolar_anchor"]


def build(X, demand, days):
    days = np.asarray(days)
    hour = (X.index - pd.Timedelta(minutes=30)).hour            # timestamps mark the END of each half-hour
    mid, eve = np.isin(hour, range(10, 16)), np.isin(hour, range(16, 21))
    d = pd.DataFrame({"day": days, "dem": np.asarray(demand, float), "sol": X["syd_solar"].to_numpy(float),
                      "tmp": X["syd_temp"].to_numpy(float)})
    all_days = pd.Index(np.unique(days))
    md = d[mid].groupby("day")[["dem", "sol", "tmp"]].mean().reindex(all_days)
    ev = d[eve].groupby("day")["dem"].mean().reindex(all_days)
    D, S, T = (md[c].to_numpy() for c in ("dem", "sol", "tmp"))
    n = len(all_days)
    beta_s, beta_t = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        sl = slice(max(0, i - WINDOW), i)                          # market days strictly before day i
        ok = np.isfinite(D[sl]) & np.isfinite(S[sl]) & np.isfinite(T[sl])
        if ok.sum() < MIN_DAYS:
            continue
        A = np.column_stack([np.ones(ok.sum()), S[sl][ok], T[sl][ok]])
        coef = np.linalg.lstsq(A, D[sl][ok], rcond=None)[0]
        beta_s[i], beta_t[i] = coef[1], coef[2]
    dip = (md["dem"] / ev).shift(1).rolling(DIP, min_periods=14).mean().to_numpy()
    pos = pd.Series(np.arange(n), index=all_days).loc[days].to_numpy()
    bs = beta_s[pos]
    c1, c7 = X["syd_solar_change_1d"].to_numpy(float), X["syd_solar_change_7d"].to_numpy(float)
    out = pd.DataFrame({"sens_beta_solar": bs, "sens_beta_temp": beta_t[pos], "sens_midday_dip_28d": dip[pos],
                        "sens_x_dsolar_1d": bs * c1, "sens_x_dsolar_7d": bs * c7,
                        "sens_x_dsolar_anchor": bs * 0.5 * (c1 + c7)}, index=X.index)
    return out[NAMES].astype(np.float32)


if __name__ == "__main__":
    import harness
    X, truth, days = harness.load()
    F = build(X, truth["demand"].to_numpy(), days)
    # look-ahead checks: features of day L must not move when day L's demand changes; they must move when day L-1's does
    L = days.max(); prev = L - pd.Timedelta(days=1)
    mL, mP = np.asarray(days == L), np.asarray(days == prev)
    dem = truth["demand"].to_numpy(float).copy(); dem[mL] *= 1.5
    assert np.allclose(F.to_numpy()[mL], build(X, dem, days).to_numpy()[mL], equal_nan=True), "look-ahead (same day)!"
    dem = truth["demand"].to_numpy(float).copy(); dem[mP] *= 1.5
    assert not np.allclose(F.to_numpy()[mL][:, 0], build(X, dem, days).to_numpy()[mL][:, 0]), "beta ignores day L-1"
    np.save(f"{harness.RUNS}/solar_sens.npy", F.to_numpy(np.float32))
    json.dump(NAMES, open(f"{harness.RUNS}/solar_sens_names.json", "w"))
    yr = np.asarray(days.year)
    print("look-ahead checks passed; NaN rows:", int(np.isnan(F.to_numpy()).any(axis=1).sum()))
    print("yearly mean of sens_beta_solar (MW per unit of midday irradiance) and midday dip:")
    for y in range(2010, 2025):
        m = yr == y
        print(f"  {y}: beta_solar {np.nanmean(F['sens_beta_solar'].to_numpy()[m]):+8.3f}   "
              f"beta_temp {np.nanmean(F['sens_beta_temp'].to_numpy()[m]):+7.2f}   dip {np.nanmean(F['sens_midday_dip_28d'].to_numpy()[m]):.3f}")
