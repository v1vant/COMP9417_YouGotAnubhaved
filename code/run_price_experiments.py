"""Price-only research. Never scores post-2024 outcomes or replaces demand artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/forecast_package'))
from model import load_data, market_days, training_table, predict_price_probability
from price_context import extend_price_features
from train import fit_model, SEED

BASE = dict(family='boosted', leaves=31, trees=600, learning_rate=.04, threshold=.57)


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return dict(recall_0=float(100*np.mean(p[y==0]==0)) if np.any(y==0) else None,
                recall_1=float(100*np.mean(p[y==1]==1)) if np.any(y==1) else None,
                accuracy=float(100*np.mean(y == p)), balanced_accuracy=float(100*np.mean(
        [np.mean(p[y == k] == k) for k in np.unique(y)])))


def threshold(y, p, previous=.57):
    grid = np.round(np.arange(.05, .951, .01), 2)
    scores = np.array([metrics(y, p > t)['balanced_accuracy'] for t in grid])
    ties = grid[np.isclose(scores, scores.max(), rtol=0, atol=1e-10)]
    return float(ties[np.argmin(abs(ties-previous))])


def fit(task_config, x, y, cutoff, window=0):
    days = market_days(x.index)
    mask = days < cutoff
    if window:
        mask &= days >= cutoff - pd.DateOffset(years=window)
    return fit_model('price', task_config, x.loc[mask], y.loc[mask])


def refresh(config, x, y, cutoff, window, previous):
    start = cutoff - pd.Timedelta(days=365)
    middle = start + pd.Timedelta(days=182)
    days = market_days(x.index)
    probs, labels = [], []
    for a, b in [(start, middle), (middle, cutoff)]:
        m = fit(config, x, y, a, window)
        mask = (days >= a) & (days < b)
        probs.extend(predict_price_probability(m, x.loc[mask]))
        labels.extend(y.loc[mask, 'price'])
    return threshold(labels, np.array(probs), previous)


def backtest(config, x, y, end, window=0, adaptive=False):
    days = market_days(x.index)
    boundaries = list(pd.to_datetime(['2015-01-01','2016-01-01','2018-01-01',
                                      '2020-01-01','2022-01-01','2024-01-01']))
    rows, fits = [], []
    frozen = None
    t = config.get('threshold', .57)
    for start, stop in zip(boundaries, boundaries[1:] + [end + pd.Timedelta(days=1)]):
        if start > end:
            break
        if adaptive:
            t = refresh(config, x, y, start, window, t)
        began = time.monotonic()
        m = fit(config, x, y, start, window)
        if frozen is None:
            frozen = m
        mask = (days >= start) & (days < min(stop, end + pd.Timedelta(days=1)))
        p = predict_price_probability(m, x.loc[mask])
        fp = predict_price_probability(frozen, x.loc[mask])
        rows.append(pd.DataFrame(dict(actual=y.loc[mask, 'price'], probability=p,
                                     frozen_prediction=(fp > config.get('threshold', .57)).astype(int),
                                     prediction=(p > t).astype(int), threshold=t)))
        fits.append(dict(effective_day=str(start.date()), training_start=m['training_start'],
                         training_end=m['training_end'], seconds=time.monotonic()-began))
    return pd.concat(rows), fits


def paired(reference, candidate, draws=2000):
    """Paired seven-day moving blocks; BA recomputed from confusion counts."""
    if not reference.index.equals(candidate.index):
        raise ValueError('Paired predictions must have identical timestamps')
    y = reference.actual.to_numpy().reshape(-1, 48)
    def counts(frame):
        p = frame.prediction.to_numpy().reshape(-1, 48)
        return np.stack([((y == k) & (p == j)).sum(axis=1)
                         for k,j in [(0,0),(0,1),(1,0),(1,1)]], axis=1)
    a,b = counts(reference),counts(candidate)
    def score(c):
        den = np.array([c[0]+c[1], c[2]+c[3]])
        return 100*np.mean(np.array([c[0],c[3]])[den>0]/den[den>0])
    rng = np.random.default_rng(SEED)
    delta=[]
    n=len(y); block=min(7,n)
    for _ in range(draws):
        starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
        ix=(starts[:,None]+np.arange(block)).ravel()[:n]
        delta.append(score(b[ix].sum(axis=0))-score(a[ix].sum(axis=0)))
    return dict(delta_BA_pp=score(b.sum(axis=0))-score(a.sum(axis=0)),
                low=float(np.percentile(delta,2.5)),high=float(np.percentile(delta,97.5)))


def reports(name, p):
    days=market_days(p.index)
    groups=[('development',days.year<=2022),('confirmation',days.year>=2023)]
    groups += [(str(v),days.year==v) for v in sorted(set(days.year))]
    groups += [(str(v),days.to_period('M')==v) for v in days.to_period('M').unique()]
    return [dict(candidate=name,period=label,rows=int(mask.sum()),
                 **metrics(p.loc[mask,'actual'],p.loc[mask,'prediction']))
            for label,mask in groups if mask.any()]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=ROOT/'data/nem_weather_halfhourly_2010_2026.csv')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--phase',choices=['features','policies','tuning','complementary'],default='features')
    ap.add_argument('--feature-set',choices=['base','geometry','recency','both'],default='base')
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--seed',type=int,default=SEED)
    ap.add_argument('--hours',type=float,default=24)
    args=ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()): ap.error('Use an empty output directory')
    if args.hours<=0: ap.error('--hours must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    began=time.monotonic()
    sources=[Path(__file__),*sorted((ROOT/'code/forecast_package').glob('*.py'))]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (args.output/'started.json').write_text(json.dumps(dict(sources=hashes,arguments=vars(args)),default=str,indent=2))
    frame=load_data(args.data)
    # Truncate BEFORE any feature engineering. Later outcomes never enter research.
    frame=frame.loc[market_days(frame.index)<=pd.Timestamp('2024-12-31')]
    x,y=training_table(frame); days=market_days(x.index)
    if not set(range(2010,2025)).issubset(set(days.year)) or days.max()!=pd.Timestamp('2024-12-31'):
        ap.error('Research requires complete history through market day 2024-12-31')
    config=dict(BASE,trees=120 if args.quick else 600,seed=args.seed)
    candidates=[]
    if args.phase=='features':
        candidates=[(s,dict(config,price_feature_set=s),0,False) for s in ['base','geometry','recency','both']]
    elif args.phase=='policies':
        for name,extra,window,adapt in [('control',{},0,False),('refresh',{},0,True),
                  ('unweighted',{'class_weight':None},0,True),('window2',{},2,False),
                  ('window4',{},4,False),('decay2',{'half_life_years':2},0,False),
                  ('decay4',{'half_life_years':4},0,False)]:
            candidates.append((name,dict(config,price_feature_set=args.feature_set,**extra),window,adapt))
    elif args.phase=='complementary':
        candidates=[('control',dict(config,price_feature_set=args.feature_set),0,False)]
        candidates += [(f'logistic_{c}',dict(config,family='linear',regularization=c,price_feature_set=args.feature_set),0,False)
                       for c in [.001,.01,.1,1.]]
        candidates += [(f'xgb_{depth}_{rate}',dict(config,family='xgboost',max_depth=depth,
                        learning_rate=rate,min_child_weight=50,reg_lambda=20,price_feature_set=args.feature_set),0,False)
                       for depth in [4,6,8] for rate in [.03,.06]]
    else:
        space=dict(leaves=[15,31,63],max_depth=[5,7,-1],min_child_samples=[80,160,320,640],
                   learning_rate=[.02,.04,.07],reg_lambda=[1,8,25,60],reg_alpha=[0,1,5],
                   colsample_bytree=[.65,.85,1.],subsample=[.75,.85,1.])
        configs=[dict(config,price_feature_set=args.feature_set)]
        configs += [dict(config,price_feature_set=args.feature_set,early_stopping=True,**p) for p in ParameterSampler(space,n_iter=16,random_state=SEED)
                    if p['max_depth']<0 or p['leaves']<=2**p['max_depth']]
        sx=extend_price_features(frame,x,args.feature_set)
        screening=[]
        for i,c in enumerate(configs):
            if time.monotonic()-began>args.hours*3600: break
            scores=[]
            for year in [2012,2013]:
                m=fit(c,sx,y,pd.Timestamp(year=year,month=1,day=1))
                mask=days.year==year
                scores.append(metrics(y.loc[mask,'price'],predict_price_probability(m,sx.loc[mask])>.57)['balanced_accuracy'])
            screening.append(dict(name=f'tune{i}',config=c,score=float(np.mean(scores))))
        (args.output/'screening.json').write_text(json.dumps(screening,indent=2))
        candidates=[('control',dict(config,price_feature_set=args.feature_set),0,False)]
        candidates += [(r['name'],r['config'],0,False) for r in sorted(screening,key=lambda z:-z['score'])[:4] if r['name']!='tune0']
    records=[]; validation=[]; comparisons=[]; reference=None; configs={}; all_fits=[]
    for name,c,window,adapt in candidates:
        if time.monotonic()-began>args.hours*3600: break
        sx=extend_price_features(frame,x,c['price_feature_set'])
        m=fit(c,sx,y,pd.Timestamp('2014-01-01'))
        valid=days.year==2014
        vp=predict_price_probability(m,sx.loc[valid])
        pd.DataFrame(dict(actual=y.loc[valid,'price'],probability=vp)).to_parquet(args.output/f'{name}_validation.parquet')
        validation.append(dict(candidate=name,**metrics(y.loc[valid,'price'],vp>.57)))
        p,fits=backtest(c,sx,y,pd.Timestamp('2024-12-31'),window,adapt)
        # Retain knowable context separately from retrospective error cohorts.
        prior = frame.nsw_price.groupby(market_days(frame.index)).mean().shift(1)
        tau = prior.reindex(market_days(p.index)).to_numpy()
        p['actual_margin'] = frame.loc[p.index, 'nsw_price'].to_numpy() - tau
        for lag in (1,7):
            p[f'baseline_{lag}d'] = (frame.nsw_price.shift(48*lag).loc[p.index].to_numpy()>tau).astype(int)
        p.to_parquet(args.output/f'{name}_predictions.parquet')
        records.extend(reports(name,p))
        frozen_p=p.copy(); frozen_p['prediction']=frozen_p['frozen_prediction']
        records.extend(reports('frozen_'+name,frozen_p)); all_fits.extend(dict(candidate=name,**f) for f in fits)
        configs[name]=dict(config=c,window_years=window,refresh_threshold=adapt)
        if reference is None: reference=p
        mask=market_days(p.index).year<=2022
        comparisons.append(dict(candidate=name,**paired(reference.loc[mask],p.loc[mask])))
        pd.DataFrame(records).to_csv(args.output/'metrics.csv',index=False)
        pd.DataFrame(validation).to_csv(args.output/'validation.csv',index=False)
        pd.DataFrame(all_fits).to_csv(args.output/'fits.csv',index=False)
        pd.DataFrame(comparisons).to_csv(args.output/'paired_deltas.csv',index=False)
        print(name,records[-1],comparisons[-1],flush=True)
    if records:
        table=pd.DataFrame(records)
        dev=table[(table.period=='development') & table.candidate.isin(configs)].sort_values('balanced_accuracy',ascending=False)
        winner=str(dev.iloc[0].candidate)
        v=pd.DataFrame(validation).sort_values('balanced_accuracy',ascending=False)
        selection=dict(stage1_candidate=str(v.iloc[0].candidate),development_candidate=winner,
                       configuration=configs[winner],status='Exploratory; promotion requires paired and confirmation review')
        (args.output/'selection.json').write_text(json.dumps(selection,indent=2))
        diagnostics=[]
        for name in configs:
            p=pd.read_parquet(args.output/f'{name}_predictions.parquet')
            for label,mask in [('margin_0_5',p.actual_margin.abs()<=5),
                               ('margin_5_20',(p.actual_margin.abs()>5)&(p.actual_margin.abs()<=20)),
                               ('margin_20_plus',p.actual_margin.abs()>20)]:
                if mask.any(): diagnostics.append(dict(candidate=name,cohort=label,rows=int(mask.sum()),**metrics(p.loc[mask,'actual'],p.loc[mask,'prediction'])))
        pd.DataFrame(diagnostics).to_csv(args.output/'error_cohorts.csv',index=False)
        for lag in (1,7):
            bp=reference.copy(); bp['prediction']=bp[f'baseline_{lag}d']
            records.extend(reports(f'baseline_{lag}d',bp))
        pd.DataFrame(records).to_csv(args.output/'metrics.csv',index=False)
    sources=[Path(__file__),*sorted((ROOT/'code/forecast_package').glob('*.py'))]
    manifest=dict(phase=args.phase,quick=args.quick,seed=args.seed,configurations=configs,
        seconds=time.monotonic()-began,cutoff='2024-12-31',confirmation='Previously inspected; exploratory',
        data_hash=hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest(),
        sources=hashes,
        versions={p:__import__('importlib.metadata',fromlist=['version']).version(p) for p in ['numpy','pandas','lightgbm','scikit-learn','joblib']},python=platform.python_version())
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=2): main()
