"""Frozen schedule, cache validation and historical weekly prediction helpers."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DEFAULT_DATA=ROOT/'data/nem_weather_halfhourly_2010_2026.csv'


def read_json(path):return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def frozen_model():return load_module('live_week_frozen_model',HERE/'frozen_package/model.py')


def harness():return load_module('live_week_frozen_harness',HERE/'frozen_harness.py')


def verify_sources():
    for relative,expected in read_json(HERE/'frozen_sources.json').items():
        if digest(HERE/relative)!=expected:raise ValueError(f'Frozen source changed: {relative}')


def experiment_key(data):
    verify_sources()
    payload={'plan':read_json(HERE/'plan.json'),'recipe':read_json(HERE/'recipe.json'),
             'sources':read_json(HERE/'frozen_sources.json'),'data_sha256':digest(data)}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def build_windows():
    plan=read_json(HERE/'plan.json');rows=[]
    for year in plan['seasonal_years']:
        rows.append(dict(window=f'seasonal_{year}',group='seasonal',training_cutoff=f'{year}-10-31',
                         week_start=f'{year}-12-07',week_end=f'{year}-12-13'))
    for cutoff in pd.date_range(plan['recent_cutoffs']['first'],plan['recent_cutoffs']['last'],freq='ME'):
        start=cutoff+pd.Timedelta(days=plan['forecast_start_lag_days'])
        rows.append(dict(window=f'recent_{cutoff:%Y_%m}',group='recent',training_cutoff=str(cutoff.date()),
            week_start=str(start.date()),week_end=str((start+pd.Timedelta(days=6)).date())))
    table=pd.DataFrame(rows)
    for row in table.itertuples():
        assert (pd.Timestamp(row.week_start)-pd.Timestamp(row.training_cutoff)).days==37
        assert (pd.Timestamp(row.week_end)-pd.Timestamp(row.week_start)).days==6
    ordered=table.sort_values('week_start')
    assert all(pd.Timestamp(b.week_start)>pd.Timestamp(a.week_end)
               for a,b in zip(list(ordered.itertuples())[:-1],list(ordered.itertuples())[1:]))
    return table


def cache_location(window,key,cache_root=None,quick=False):
    root=Path(cache_root) if cache_root else HERE/'cache'
    mode='smoke' if quick else 'full'
    return root/f'{key[:16]}_{mode}_{window["training_cutoff"]}'


def validate_cache(path,key,quick=False):
    marker=Path(path)/'complete.json'
    if not marker.exists():return False
    metadata=read_json(marker)
    if metadata.get('experiment_key')!=key or metadata.get('quick')!=quick:return False
    for relative,expected in metadata['artifact_sha256'].items():
        target=Path(path)/relative
        if not target.is_file() or digest(target)!=expected:return False
    return True


def fit_snapshot(frame,window,key,cache_root=None,log_root=None,quick=False):
    """Only this function fits; all writes are inside the evaluation cache."""
    verify_sources()
    path=cache_location(window,key,cache_root,quick)
    if validate_cache(path,key,quick):return path
    cutoff=pd.Timestamp(window['training_cutoff']);da=harness()
    prefix=frame.loc[da.market_day(frame.index)<=cutoff]
    if prefix.empty or da.market_day(prefix.index).max()!=cutoff:raise ValueError('Training cutoff absent')
    assert prefix.index.max()==cutoff+pd.Timedelta(days=1)
    path.mkdir(parents=True,exist_ok=True);package=path/'package'
    shutil.copytree(HERE/'frozen_package',package,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','artifacts'))
    logs=Path(log_root) if log_root else HERE/'logs';logs.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='live_week_training_') as directory:
        csv=Path(directory)/'training_prefix.csv'
        prefix.to_csv(csv,index_label='timestamp')
        command=[sys.executable,'-u',str(package/'train.py'),str(csv),'--selection',str(HERE/'recipe.json')]
        if quick:command.append('--quick')
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
        with (logs/f'{path.name}.log').open('a') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        if result.returncode:raise RuntimeError(f'Training failed; see {logs/path.name}.log')
    manifest=read_json(package/'artifacts/manifest.json')
    assert pd.Timestamp(manifest['data_end']).normalize()==cutoff
    assert manifest['quick']==quick
    assert manifest['demand_mlp_seeds']==list(range(9417,9422))
    assert manifest['task_feature_counts']=={'demand':405,'price':597}
    hashes={str(p.relative_to(path)):digest(p) for p in (package/'artifacts').iterdir() if p.is_file()}
    (path/'complete.json').write_text(json.dumps(dict(experiment_key=key,quick=quick,
        training_cutoff=str(cutoff.date()),training_raw_rows=len(prefix),seconds=time.monotonic()-started,
        artifact_sha256=hashes),indent=2))
    return path


def predict_week(frame,window,path,key,quick=False):
    if not validate_cache(path,key,quick):raise ValueError('Incomplete, modified or incompatible snapshot')
    marker=read_json(Path(path)/'complete.json')
    assert marker['training_cutoff']==window['training_cutoff']
    da=harness();module=frozen_model()
    forecaster=module.Forecaster(str(Path(path)/'package/artifacts'))
    assert len(forecaster.demand_mlp_members)==5
    assert len(forecaster.models['price']['members'])==(1 if quick else 5)
    rows=[]
    for day in pd.date_range(window['week_start'],window['week_end'],freq='D'):
        history,weather=da.make_inputs(frame,day)
        assert da.market_day(history.index).max()==day-pd.Timedelta(days=1)
        assert len(weather)==48 and (da.market_day(weather.index)==day).all()
        # The fitted object stays fixed; newly observed history is supplied daily.
        with threadpool_limits(limits=2):prediction=forecaster.predict(history,weather)
        actual=da.targets(frame,day);baseline=da.baseline(history,weather,1)
        assert prediction.index.equals(actual.index) and prediction.shape==(48,2)
        assert np.isfinite(prediction.to_numpy()).all()
        assert prediction.nsw_price_up.isin([0,1]).all()
        row=actual.add_prefix('actual_').join(prediction.add_prefix('current_')).join(baseline.add_prefix('baseline_'))
        rows.append(row)
    result=pd.concat(rows)
    assert len(result)==336 and validate_cache(path,key,quick)
    return result


def score_week(predictions):
    da=harness();truth=predictions.filter(like='actual_').rename(columns=lambda x:x.removeprefix('actual_'))
    rows=[]
    for model in ['current','baseline']:
        prediction=predictions.filter(like=model+'_').rename(columns=lambda x:x.removeprefix(model+'_'))
        scores=da.score(truth,prediction)
        rows.append(dict(model=model,demand_MAPE=scores['MAPE'],demand_MAE=scores['MAE'],
            price_accuracy=scores['Acc'],price_balanced_accuracy=scores['BalAcc'],
            positive_price_intervals=int(truth.nsw_price_up.sum()),negative_price_intervals=int((truth.nsw_price_up==0).sum())))
    return rows
