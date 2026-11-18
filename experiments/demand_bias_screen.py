"""Exploratory causal residual correction on previously saved biennial forecasts.

Select hyperparameters on 2015-2020; check 2021-2024 and the previously
inspected 2025-2026 holdout. A forecast may use errors through yesterday only.
"""
import io, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = 'origin/feature/demand-mlp'
def blob(path):
    return subprocess.check_output(['git', 'show', f'{REF}:{path}'], cwd=ROOT)
def array(path):
    return np.load(io.BytesIO(blob(path)))
truth = np.load(io.BytesIO(blob('results/demand_mlp_10epoch_2015_2024/truth.npz')))
idx = pd.DatetimeIndex(truth['timestamps'])
y = truth['demand'].ravel()
a = array('results/feature_comparison_2015_2024/current_demand.npy').ravel()
b = array('results/demand_mlp_10epoch_2015_2024/mlp_demand.npy').ravel()
old = pd.DataFrame({'actual': y, 'pred': (a+b)/2}, index=idx)
late = pd.read_csv(io.BytesIO(blob('results/frozen_demand_blend_holdout/predictions.csv')), parse_dates=['timestamp']).set_index('timestamp')
late = late.rename(columns={'frozen_50_50':'pred'})[['actual','pred']]
df = pd.concat([old,late]).sort_index()
if df.index.has_duplicates or not (df.index.to_series().diff().dropna() == pd.Timedelta(minutes=30)).all():
    raise ValueError('predictions must be contiguous and unique')
day = (df.index - pd.Timedelta(minutes=30)).normalize()
slot = (df.index.hour * 2 + df.index.minute // 30 - 1) % 48
resid = (df.actual - df.pred).to_numpy().reshape(-1,48)
assert all(np.bincount(day.factorize()[0]) == 48)
# Each row describes a complete market day; shift(1) excludes target-day errors.
r = pd.DataFrame(resid)
periods = {'selection_2015_2020': (day.year <= 2020), 'check_2021_2024': ((day.year >= 2021)&(day.year <= 2024)), 'reused_2025_2026': (day.year >= 2025)}
rows=[]
for days in (3,7,14,28,56):
    shifted = r.shift(1)
    for mode in ('daily','slot'):
        hist = shifted.mean(axis=1).rolling(days, min_periods=days).mean().to_numpy()[:,None] if mode=='daily' else shifted.rolling(days, min_periods=days).mean().to_numpy()
        correction = np.broadcast_to(hist, resid.shape).ravel()
        for scale in (0.25,0.5,0.75,1.0):
            pred = df.pred.to_numpy() + scale * correction
            row={'days':days,'mode':mode,'scale':scale}
            for name, mask in periods.items():
                valid=mask & np.isfinite(pred)
                row[name] = 100*np.mean(np.abs(df.actual.to_numpy()[valid]-pred[valid])/df.actual.to_numpy()[valid])
                row[name+'_baseline'] = 100*np.mean(np.abs(df.actual.to_numpy()[valid]-df.pred.to_numpy()[valid])/df.actual.to_numpy()[valid])
            rows.append(row)
out=ROOT/'results'/'codex_demand_bias_screen'
out.mkdir(parents=True,exist_ok=True)
res=pd.DataFrame(rows)
res['selection_delta'] = res['selection_2015_2020'] - res['selection_2015_2020_baseline']
res=res.sort_values('selection_delta')
res.to_csv(out/'metrics.csv',index=False)
print(res.head(10).to_string(index=False))
print('\nBest selected:',res.iloc[0].to_json())

# Stateless proxy: for each historical day, its same-slot yesterday / last-week
# anchor would have been known before forecasting it. No saved forecasts needed.
actual = pd.DataFrame(df.actual.to_numpy().reshape(-1,48))
anchor_error = actual - (actual.shift(1) + actual.shift(7)) / 2
proxy_rows=[]
for days in (3,7,14,28,56):
    for mode in ('daily','slot'):
        past=anchor_error.shift(1)
        hist = past.mean(axis=1).rolling(days,min_periods=days).mean().to_numpy()[:,None] if mode=='daily' else past.rolling(days,min_periods=days).mean().to_numpy()
        correction = np.broadcast_to(hist,resid.shape).ravel()
        for scale in (-0.5,-0.25,0.0,0.25,0.5,0.75,1.0):
            pred=df.pred.to_numpy()+scale*correction
            row={'days':days,'mode':mode,'scale':scale}
            for name,mask in periods.items():
                valid=mask & np.isfinite(pred)
                row[name]=100*np.mean(np.abs(df.actual.to_numpy()[valid]-pred[valid])/df.actual.to_numpy()[valid])
                row[name+'_baseline']=100*np.mean(np.abs(df.actual.to_numpy()[valid]-df.pred.to_numpy()[valid])/df.actual.to_numpy()[valid])
            proxy_rows.append(row)
proxy=pd.DataFrame(proxy_rows)
proxy['selection_delta'] = proxy['selection_2015_2020'] - proxy['selection_2015_2020_baseline']
proxy=proxy.sort_values('selection_delta')
proxy.to_csv(out/'stateless_proxy_metrics.csv',index=False)
print('\nStateless proxy top settings:')
print(proxy.head(8).to_string(index=False))

# A stateless implementation could recompute recent forecasts with the currently
# loaded model. Saved forecasts use the same model except just after refits, so
# exclude the first 28 days following every historical refit. This is a proxy
# for that implementation; it is not a runtime or inference-parity test.
from datetime import timedelta
refits=pd.to_datetime(['2015-01-01','2016-01-01','2018-01-01','2020-01-01','2022-01-01','2024-01-01','2025-01-01','2026-01-01'])
eligible=np.ones(len(df), dtype=bool)
for start in refits:
    eligible &= ~((day>=start)&(day<start+pd.Timedelta(days=28)))
correction=r.shift(1).rolling(28,min_periods=28).mean().to_numpy().ravel()
new=df.pred.to_numpy()+0.5*correction
err0=100*np.abs(df.actual.to_numpy()-df.pred.to_numpy())/df.actual.to_numpy()
err1=100*np.abs(df.actual.to_numpy()-new)/df.actual.to_numpy()
summary=[]
for name,mask in periods.items():
    valid=eligible & mask & np.isfinite(new)
    daily=pd.DataFrame({'day':day[valid], 'gain':err0[valid]-err1[valid]}).groupby('day').gain.mean()
    # Moving-block bootstrap, seven days, fixed seed; reflects serial dependence.
    vals=daily.to_numpy(); rng=np.random.default_rng(9417); n=len(vals); starts=np.arange(max(1,n-6))
    means=[]
    for _ in range(2000):
        ix=rng.choice(starts,size=int(np.ceil(n/7)),replace=True)[:,None]+np.arange(7)
        means.append(vals[ix.ravel()[:n]].mean())
    summary.append({'period':name,'days':n,'baseline_MAPE':float(err0[valid].mean()),'corrected_MAPE':float(err1[valid].mean()),'gain_pp':float((err0[valid]-err1[valid]).mean()),'bootstrap_low':float(np.quantile(means,.025)),'bootstrap_high':float(np.quantile(means,.975))})
summary=pd.DataFrame(summary)
summary.to_csv(out/'same_model_proxy.csv',index=False)
print('\nExcluding first 28 days after refits:')
print(summary.to_string(index=False))
