"""Weather-based input drift score for the drift-detector simulations.

The input detector watches the forecast inputs the model is given for day D - the 12 weather columns (temperature,
relative humidity, cloud cover and solar radiation in Sydney, Melbourne and Brisbane) - rather than the shape of the
demand curve. Per market day and city it summarises: temperature mean / max / min, humidity mean, cloud mean and solar
mean (18 daily features).

Weather is strongly seasonal, so raw distances would flag every summer. The scorer therefore fits, on the fit days only:
  1. a seasonal climatology per feature: least-squares regression on 3 annual harmonics (sin/cos of 2*pi*h*doy/365.25);
  2. a seasonal spread per feature: the same harmonic regression on the squared residuals (floored at 5% of their mean);
  3. a Ledoit-Wolf covariance of the standardised anomalies (the cities and variables are correlated).
A day's score is the Mahalanobis distance of its standardised anomaly vector, i.e. how unusual D's weather is for the
time of year, relative to the fit period. A sustained run of unusual weather (or a shift in the climate / sensors)
raises the 28-day rolling mean the detector monitors."""
import numpy as np, pandas as pd

CITIES = ("syd", "mel", "bne")
HARMONICS = 3


def daily_weather(csv):
    """One row per market day (00:30 .. 24:00), 18 weather summary features."""
    w = pd.read_csv(csv, index_col="timestamp", parse_dates=["timestamp"],
                    usecols=["timestamp"] + [f"{c}_{k}" for c in CITIES for k in ("temp", "rh", "cloud", "solar")])
    g = w.groupby((w.index - pd.Timedelta(minutes=30)).normalize())
    assert g.size().eq(48).all()
    cols = {}
    for c in CITIES:
        cols[f"{c}_temp_mean"], cols[f"{c}_temp_max"], cols[f"{c}_temp_min"] = (
            g[f"{c}_temp"].mean(), g[f"{c}_temp"].max(), g[f"{c}_temp"].min())
        for k in ("rh", "cloud", "solar"):
            cols[f"{c}_{k}_mean"] = g[f"{c}_{k}"].mean()
    return pd.DataFrame(cols)


def _season(index):
    t = 2 * np.pi * np.asarray(index.dayofyear, float) / 365.25
    return np.column_stack([np.ones(len(t))] + [f(h * t) for h in range(1, HARMONICS + 1) for f in (np.sin, np.cos)])


def weather_scorer(fit):
    """Fit on a daily-weather frame (rows = fit days); return a function scoring any daily-weather frame."""
    from sklearn.covariance import LedoitWolf
    S, X = _season(fit.index), fit.to_numpy(float)
    mean_coef = np.linalg.lstsq(S, X, rcond=None)[0]
    sq = (X - S @ mean_coef) ** 2
    var_coef = np.linalg.lstsq(S, sq, rcond=None)[0]
    floor = 0.05 * sq.mean(axis=0)

    def standardise(frame):
        s = _season(frame.index)
        return (frame.to_numpy(float) - s @ mean_coef) / np.sqrt(np.maximum(s @ var_coef, floor))
    precision = LedoitWolf().fit(standardise(fit)).precision_

    def score(frame):
        z = standardise(frame)
        return pd.Series(np.sqrt(np.einsum("ij,jk,ik->i", z, precision, z)), index=frame.index)
    return score
