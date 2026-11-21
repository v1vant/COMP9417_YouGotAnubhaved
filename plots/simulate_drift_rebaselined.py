"""Re-baselined drift detectors for every original Step 1 model, January 2015 - October 2026.

Same models, detectors and retraining rules as simulate_drift.py (statistic: 28-day rolling mean; threshold: 99th
percentile of the reference's 28-day rolling means; >= 28 days between signals; retrain from D+1 on the 4 years up to the
end of D; budget 6 per model and strategy; no minimum gap between retrains), except that BOTH detectors are re-baselined on
the data seen so far, at the start and at every retrain (refitting the weather scorer is part of that retraining):
  input detector  - weather scorer (weather_drift.py: seasonal climatology, seasonal spread and covariance of the
                    Sydney / Melbourne / Brisbane weather summaries) refitted on the first 3 years of the 4-year window;
                    reference = the weather drift scores of the window's last year.
  error detector  - the model refitted on the first 3 years of the window; reference = its daily errors on the last year.
  The deployed model is fitted on the whole window; after each reset the 28-day window starts empty.
At the start (2015) this is the original setup: weather scorer on 2010-2013, model on 2010-2013, reference 2014.
Optional: INPUT_QUANTILE=0.999 uses the 99.9th percentile for the input detector (output in rebaselined_input99.9/).
Run from the repo root: python plots/simulate_drift_rebaselined.py"""
import json, os, sys, time
from collections import deque
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate_drift as base                                      # noqa: E402

INPUT_Q = float(os.environ.get("INPUT_QUANTILE", "0.99"))      # input-detector threshold quantile (error detector: 0.99)
OUT = os.path.join(base.OUT, "rebaselined" if INPUT_Q == 0.99 else f"rebaselined_input{INPUT_Q * 100:g}")


class Detector:
    """analysis_tools.OnlineDetector, but the rolling window starts empty after each (re)baseline."""
    def __init__(self, reference, window=28, quantile=0.99, cooldown=28):
        rolling = pd.Series(np.asarray(reference, float)).rolling(window).mean().dropna()
        self.threshold = float(rolling.quantile(quantile))
        self.window, self.cooldown = window, cooldown
        self.values = deque(maxlen=window)
        self.last_signal, self.statistic = None, None

    def observe(self, day, value):
        self.values.append(float(value))
        self.statistic = float(np.mean(self.values)) if len(self.values) == self.window else None
        active = self.statistic is not None and self.statistic > self.threshold
        eligible = self.last_signal is None or (day - self.last_signal).days >= self.cooldown
        if active and eligible:
            self.last_signal = day
            return True
        return False


def run_model(key):
    base.step1_package()
    from model import market_days
    from train import fit_model
    task, config = base.MODELS[key]
    from weather_drift import daily_weather, weather_scorer
    x, y = base.data()
    days = market_days(x.index)
    col = "demand" if task == "demand" else "price"
    uday = pd.DatetimeIndex(np.unique(np.asarray(days)))
    weather = daily_weather(base.CSV).reindex(uday)               # one row per market day
    sim_days = pd.date_range(base.START, base.END, freq="D")
    rows = np.asarray((days >= base.START) & (days <= base.END))
    xs, actual = x[rows], y[col].to_numpy()[rows]
    sim_pos = np.searchsorted(uday, sim_days)

    def reset(day_end):
        """Baseline at the end of day_end (window = 4 years up to day_end, or 2010-2014 at the start)."""
        if day_end == pd.Timestamp("2014-12-31"):                   # identical to the fixed-baseline run
            fit_days, ref_days = uday[uday.year <= 2013], uday[uday.year == 2014]
        else:
            win = uday[(uday <= day_end) & (uday > day_end - pd.DateOffset(years=base.WINDOW_YEARS))]
            ref_days = win[win > day_end - pd.DateOffset(years=1)]; fit_days = win[win <= day_end - pd.DateOffset(years=1)]
        scorer = weather_scorer(weather.loc[fit_days])
        scores_all = np.full(len(uday), np.nan)
        later = uday >= ref_days.min()
        scores_all[later] = scorer(weather[later]).to_numpy()
        inp_ref = scores_all[np.isin(uday, ref_days)]
        fm = np.isin(np.asarray(days), np.asarray(fit_days)); rm = np.isin(np.asarray(days), np.asarray(ref_days))
        ref_model = fit_model(task, config, x[fm], y[fm])
        err_ref = base.daily_error(task, y[col].to_numpy()[rm], base.predict(task, ref_model, x[rm]))
        return Detector(inp_ref, quantile=INPUT_Q), Detector(err_ref), scores_all, {
            "baseline_end": str(day_end.date()), "fit": f"{fit_days.min().date()}..{fit_days.max().date()}",
            "reference": f"{ref_days.min().date()}..{ref_days.max().date()}"}

    frozen = fit_model(task, config, x[days.year <= 2014], y[days.year <= 2014])
    frozen_pred = base.predict(task, frozen, xs)
    out = {"task": task, "config": config, "frozen_daily_error": base.daily_error(task, actual, frozen_pred).tolist()}
    for trigger in ("input", "error"):
        pred = frozen_pred.copy()
        det_in, det_err, scores_all, info = reset(pd.Timestamp("2014-12-31"))
        baselines = [dict(info, input_threshold=det_in.threshold, error_threshold=det_err.threshold, effective_day="2015-01-01")]
        retrains, log, pending = [], [], False
        for i, day in enumerate(sim_days):
            if pending and len(retrains) < base.BUDGET:            # retrain + re-baseline, effective from D+1
                t0 = time.time()
                mask = (days < day) & (days >= day - pd.DateOffset(years=base.WINDOW_YEARS))
                model = fit_model(task, config, x[mask], y[mask])
                pred[i * 48:] = base.predict(task, model, xs.iloc[i * 48:])
                det_in, det_err, scores_all, info = reset(day - pd.Timedelta(days=1))
                retrains.append({"effective_day": str(day.date()), "training_start": str(days[mask].min().date()),
                                 "training_end": str(days[mask].max().date()), "seconds": round(time.time() - t0, 1)})
                baselines.append(dict(info, input_threshold=det_in.threshold, error_threshold=det_err.threshold,
                                      effective_day=str(day.date())))
            pending = False
            e = float(base.daily_error(task, actual[i * 48:(i + 1) * 48], pred[i * 48:(i + 1) * 48])[0])
            s_in = det_in.observe(day, float(scores_all[sim_pos[i]])); s_err = det_err.observe(day, e)
            log.append({"day": str(day.date()), "error": e, "input_score": float(scores_all[sim_pos[i]]),
                        "input_statistic": det_in.statistic, "input_threshold": det_in.threshold, "input_signal": s_in,
                        "error_statistic": det_err.statistic, "error_threshold": det_err.threshold, "error_signal": s_err})
            if (s_in if trigger == "input" else s_err) and len(retrains) < base.BUDGET:
                pending = True
        out[trigger] = {"retrains": retrains, "baselines": baselines, "log": log}
        print(f"{key} [{trigger}-triggered, re-baselined]: {len(retrains)} retrains: {[r['effective_day'] for r in retrains]}", flush=True)
    json.dump(out, open(os.path.join(OUT, f"{key}.json"), "w"))
    return key


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    os.makedirs(OUT, exist_ok=True)
    base.step1_package(); base.data()
    with ProcessPoolExecutor(4, mp_context=mp.get_context("spawn")) as pool:
        for k in pool.map(run_model, base.MODELS):
            print("done", k, flush=True)
