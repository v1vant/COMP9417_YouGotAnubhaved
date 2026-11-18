"""Chronological demand-blend features; disposable auxiliary fits, never live updates."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from research import fit_model, market_days


def demand_features(frame, blend, disagreement):
    # All forecast-day values here are MODEL OUTPUTS, never market observations.
    q=pd.DataFrame(blend,index=frame.index).iloc[:,0]
    days=market_days(q.index); values=q.to_numpy().reshape(-1,48)
    daily_days=days.unique();lookup=pd.Series(days,index=q.index)
    group=q.groupby(days)
    raw=frame.nsw_demand
    rawdays=market_days(frame.index)
    source=pd.DataFrame(raw.to_numpy().reshape(-1,48),index=rawdays.unique())
    out=pd.DataFrame(index=q.index)
    out['pd_slot']=q;out['pd_delta1']=q-raw.shift(48);out['pd_delta7']=q-raw.shift(336)
    out['pd_mean']=lookup.map(group.mean());out['pd_max']=lookup.map(group.max())
    out['pd_mean_change']=out.pd_mean-lookup.map(raw.groupby(rawdays).mean().shift(1))
    evening=values[:,34:40].mean(axis=1)-values[:,20:30].mean(axis=1)
    out['pd_evening_midday']=lookup.map(pd.Series(evening,index=daily_days))
    out['pd_ramp']= (values-values[:,np.maximum(np.arange(48)-4,0)]).ravel()
    out['pd_disagreement']=disagreement
    median=source.shift(1).rolling(28).median().to_numpy().ravel()
    spread=(source.shift(1).rolling(28).quantile(.75)-source.shift(1).rolling(28).quantile(.25)).clip(lower=100).to_numpy().ravel()
    out['pd_stress']=(q-median)/spread
    return out.astype(np.float32)


def generate(frame,x,y,output):
    output.mkdir(exist_ok=True)
    columns=json.loads((Path(__file__).parent/'demand_feature_columns.json').read_text())
    dx=x.loc[:,columns];days=market_days(x.index)
    predictions=pd.Series(np.nan,index=x.index);disagreement=predictions.copy();audit=[]
    boundaries=pd.to_datetime([f'{v}-01-01' for v in [2011,2012,2013,2014,2015,2016,2018,2020,2022,2024,2026,2027]])
    for begin,end in zip(boundaries[:-1],boundaries[1:]):
        train=days<begin;score=(days>=begin)&(days<end)
        if not score.any():continue
        # Demand has the same 405 inputs and residual loss as the preserved blend.
        m=fit_model('demand',dict(family='boosted',trees=900,leaves=63,learning_rate=.025),dx.loc[train],y.loc[train])
        pred=np.clip(dx.loc[score,'demand_anchor'].to_numpy()+m['estimator'].predict(dx.loc[score]),0,20000)
        data=dx.loc[train|score].copy();data['actual']=y.loc[data.index,'demand']
        path=output/f'fold_{begin.year}.parquet';data.to_parquet(path)
        dest=output/f'mlp_{begin.year}.npy'
        with (output/f'mlp_{begin.year}.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).parent/'stack_worker.py'),str(path),str(begin.date()),str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True)
        mlp=np.load(dest,allow_pickle=False)
        predictions.loc[score]=.5*(pred+mlp);disagreement.loc[score]=abs(pred-mlp)
        audit.append(dict(training_end=str(days[train].max().date()),prediction_start=str(begin.date()),prediction_end=str(days[score].max().date())))
        pd.DataFrame(audit).to_csv(output/'audit.csv',index=False)
    # Align raw complete days to x; training starts after its warm-up day.
    aligned=frame.loc[x.index]
    features=demand_features(aligned,predictions.to_numpy(),disagreement.to_numpy())
    features=features.loc[days.year>=2011]
    if not np.isfinite(features.to_numpy()).all():raise ValueError('Incomplete out-of-fold demand features')
    features.to_parquet(output/'features.parquet')
    return features
