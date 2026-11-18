"""Prototype stateless recent-forecast reconstruction, with feature parity check.

The model artifacts are obtained from the existing demand-mlp branch. This is a
runtime/causality check, not an out-of-sample performance evaluation.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
import subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'/'price_v2'))
import reference_model as m
from demand_bias import corrected_demand

with tempfile.TemporaryDirectory() as temp:
    for name in ('models.joblib','manifest.json','demand_mlp.npz'):
        blob=subprocess.check_output(['git','show',f'origin/feature/demand-mlp:code/forecast_package/artifacts/{name}'],cwd=ROOT)
        (Path(temp)/name).write_bytes(blob)
    model=m.Forecaster(temp)
    data=m.load_data(ROOT/'data'/'nem_weather_halfhourly_2010_2026.csv')
    history, weather=data.iloc[:-48], data.iloc[-48:][m.WEATHER]
    start=time.perf_counter()
    # 35 days of feature history plus 28 complete prediction days.
    past=history.iloc[-63*48:][m.MARKET+m.WEATHER]
    combined=pd.concat([past,weather.reindex(columns=m.MARKET+m.WEATHER)])
    batch=m.feature_frame(combined)
    recent=batch.loc[history.index[-28*48:]]
    today=batch.loc[weather.index]
    demand=.5*m.predict_task(model.models['demand'],recent)+.5*m.predict_numpy_mlp(model.demand_mlp,recent)
    correction=.5*(history.nsw_demand.iloc[-28*48:].to_numpy()-demand).reshape(28,48).mean(axis=0)
    result=.5*m.predict_task(model.models['demand'],today)+.5*m.predict_numpy_mlp(model.demand_mlp,today)+correction
    runtime=time.perf_counter()-start
    start=time.perf_counter()
    prototype=corrected_demand(m, model, history, weather, '2026-01-01')
    prototype_runtime=time.perf_counter()-start
    np.testing.assert_allclose(prototype, np.clip(result, 0, 20000), rtol=1e-6, atol=1e-3)
    uncorrected = model.predict(history, weather).nsw_demand_mw.to_numpy()
    fallback = corrected_demand(m, model, history, weather, '2026-10-31')
    np.testing.assert_allclose(fallback, uncorrected)
    # Compare batch features against the official one-day path at several positions.
    for offset in (28,14,1,0):
        split=len(history)-offset*48
        h=data.iloc[:split]; w=data.iloc[split:split+48][m.WEATHER]
        single=m.forecast_features(h,w)
        pd.testing.assert_frame_equal(batch.loc[w.index],single,check_exact=False,rtol=1e-6,atol=1e-5)
    print(f'Batch/single-day feature parity: passed (4 days); prototype: {prototype_runtime:.3f}s; combined batch: {runtime:.3f}s')
    print('Correction parity and training-cutoff fallback: passed')
    print(f'Corrected forecast finite: {np.isfinite(result).all()}; range: {result.min():.1f}–{result.max():.1f} MW')
