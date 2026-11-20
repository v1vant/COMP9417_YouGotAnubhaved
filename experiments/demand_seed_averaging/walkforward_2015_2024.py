"""Frozen five-seed demand comparison, team biennial schedule, no holdout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/'code'
sys.path.insert(0,str(CODE));sys.path.insert(0,str(CODE/'forecast_package'))
OUT=ROOT/'results/demand_seed_averaging_5seeds_2015_2024'
SEEDS=list(range(9417,9422))
PERIODS=[(2015,2015),(2016,2017),(2018,2019),(2020,2021),(2022,2023),(2024,2024)]


def score(actual,predicted):
    return dict(MAPE=float(np.mean(np.abs(actual-predicted)/actual)*100),MAE=float(np.mean(np.abs(actual-predicted))))


def worker(kind,seed,data):
    import joblib
    from model import market_days
    x,y,config=pd.read_pickle(data)
    days=market_days(x.index)
    if kind=='mlp':
        import torch
        from demand_mlp import fit_demand_mlp,predict_demand_mlp
    else:
        import train as training
        from model import predict_task
        training.SEED=seed
    audit=[]
    for start,end in PERIODS:
        train=days<pd.Timestamp(f'{start}-01-01')
        test=(days.year>=start)&(days.year<=end)
        assert days[train].max()<days[test].min()
        path=OUT/f'{kind}_{seed}_{start}_{end}.csv'
        metadata=OUT/f'{kind}_{seed}_{start}_fit.json'
        if path.exists() and metadata.exists():
            saved=json.loads(metadata.read_text())
            assert saved['seed']==seed and saved['training_end']==str(days[train].max().date())
            audit.append(saved);continue
        started=time.monotonic()
        print(f'Fitting {kind} seed {seed} for {start}–{end}, {int(train.sum())} training rows',flush=True)
        if kind=='mlp':
            bundle=fit_demand_mlp(x.loc[train],y.loc[train,'demand'],epochs=10,seed=seed)
            values=predict_demand_mlp(bundle,x.loc[test])
            torch.save(bundle['model'].state_dict(),OUT/f'mlp_{seed}_{start}.pt')
            joblib.dump({'scaler':bundle['scaler'],'features':bundle['features']},OUT/f'mlp_{seed}_{start}_preprocessing.joblib')
        else:
            bundle=training.fit_model('demand',config,x.loc[train],y.loc[train])
            values=predict_task(bundle,x.loc[test])
            joblib.dump(bundle,OUT/f'lgbm_{seed}_{start}.joblib',compress=3)
        assert np.isfinite(values).all()
        pd.Series(values,index=x.index[test],name='prediction').to_csv(path,index_label='timestamp')
        row=dict(kind=kind,seed=seed,effective_year=start,evaluation_end_year=end,
                 training_start=str(days[train].min().date()),training_end=str(days[train].max().date()),
                 training_rows=int(train.sum()),seconds=time.monotonic()-started)
        metadata.write_text(json.dumps(row,indent=2));audit.append(row)
        pd.DataFrame(audit).to_csv(OUT/f'{kind}_{seed}_audit.csv',index=False)
        print(f'Completed {kind} {seed} {start}–{end}: {row["seconds"]:.1f}s',flush=True)


def run_worker(kind,seed,data):
    log=OUT/f'{kind}_{seed}.log'
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    with log.open('a') as handle:
        result=subprocess.run([sys.executable,'-u',__file__,'--worker',kind,'--seed',str(seed),'--data',str(data)],
            stdout=handle,stderr=subprocess.STDOUT,env=env)
    if result.returncode:
        raise RuntimeError(f'{kind} seed {seed} failed: see {log}')
    print(f'Completed all folds: {kind} seed {seed}',flush=True)


def main():
    from model import load_data,market_days,training_table
    OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    if (OUT/'completion.json').exists():
        print((OUT/'metrics.csv').read_text(),flush=True);return
    config=json.loads((ROOT/'results/final_config.json').read_text())['demand']['config']
    frozen=dict(seeds=SEEDS,MLP_epochs=10,demand_features=405,blend_weights=[.5,.5],
        LightGBM=config,periods=PERIODS,threshold_tuning=False,holdout_evaluated=False,
        source_seed_9417='Saved chronological forecasts from current demand recipe',
        maximum_LightGBM_workers=2,threads_per_worker=2,
        MLP_source_sha256=hashlib.sha256((CODE/'demand_mlp.py').read_bytes()).hexdigest())
    plan=OUT/'frozen_plan.json'
    if plan.exists():assert json.loads(plan.read_text())==json.loads(json.dumps(frozen))
    else:plan.write_text(json.dumps(frozen,indent=2))
    frame=load_data(ROOT/'data/nem_weather_halfhourly_2010_2026.csv').loc[:'2025-01-01 00:00:00']
    x,y=training_table(frame)
    columns=json.loads((CODE/'demand_feature_columns.json').read_text());x=x.loc[:,columns]
    assert x.shape[1]==405
    days=market_days(x.index);evaluation=days.year>=2015
    index=x.index[evaluation];actual=y.loc[index,'demand'].to_numpy()
    assert market_days(index).min()==pd.Timestamp('2015-01-01') and market_days(index).max()==pd.Timestamp('2024-12-31')
    source=ROOT/'results/demand_mlp_10epoch_2015_2024'
    truth=np.load(source/'truth.npz')
    np.testing.assert_array_equal(truth['timestamps'],index.to_numpy());np.testing.assert_array_equal(truth['demand'],actual)
    source_audit=pd.read_csv(source/'retraining_audit.csv')
    assert list(source_audit.effective_day)==[f'{start}-01-01' for start,_ in PERIODS]
    assert all(pd.Timestamp(r.training_end)<pd.Timestamp(r.effective_day) for r in source_audit.itertuples())
    source_config=json.loads((ROOT/'results/feature_comparison_2015_2024/config.json').read_text())
    assert source_config['demand']['config']==config
    predictions=pd.DataFrame({'actual':actual},index=index)
    predictions['lgbm_9417']=np.load(ROOT/'results/feature_comparison_2015_2024/current_demand.npy').ravel()
    mlp=pd.concat([pd.read_csv(source/f'predictions_{year}.csv',index_col='timestamp',parse_dates=True).predicted for year in range(2015,2025)])
    assert mlp.index.equals(index)
    binary_mlp=np.load(source/'mlp_demand.npy').ravel()
    # Float32 CSV export rounds values; retain the original binary predictions.
    np.testing.assert_allclose(mlp.to_numpy(),binary_mlp,rtol=0,atol=.001)
    predictions['mlp_9417']=binary_mlp
    predictions.to_csv(OUT/'baseline_predictions.csv',index_label='timestamp')
    with tempfile.TemporaryDirectory(prefix='demand_seed_walkforward_',dir='/private/tmp') as directory:
        data=Path(directory)/'training.pkl';pd.to_pickle((x,y,config),data)
        # Two independent LightGBM seed jobs, four threads total. Torch is isolated.
        with ThreadPoolExecutor(max_workers=2) as pool:
            for _ in pool.map(lambda seed:run_worker('lgbm',seed,data),SEEDS[1:]):pass
        for seed in SEEDS[1:]:run_worker('mlp',seed,data)
    for kind in ['lgbm','mlp']:
        for seed in SEEDS[1:]:
            pieces=[pd.read_csv(OUT/f'{kind}_{seed}_{start}_{end}.csv',index_col='timestamp',parse_dates=True).prediction for start,end in PERIODS]
            values=pd.concat(pieces);assert values.index.equals(index)
            assert np.isfinite(values.to_numpy()).all();predictions[f'{kind}_{seed}']=values
        for n in [3,5]:predictions[f'{kind}_mean{n}']=predictions[[f'{kind}_{seed}' for seed in SEEDS[:n]]].mean(axis=1)
    combos={'current':('lgbm_9417','mlp_9417')}
    for n in [3,5]:
        combos.update({f'bagged_LGBM_only_{n}':(f'lgbm_mean{n}','mlp_9417'),
                       f'bagged_MLP_only_{n}':('lgbm_9417',f'mlp_mean{n}'),
                       f'both_bagged_{n}':(f'lgbm_mean{n}',f'mlp_mean{n}')})
    rows=[];yearly=[];daily={};eval_days=market_days(index)
    for name,(a,b) in combos.items():
        predicted=.5*(predictions[a]+predictions[b]);predictions[name]=predicted
        rows.append(dict(model=name,**score(actual,predicted.to_numpy())))
        for year in range(2015,2025):
            mask=eval_days.year==year
            yearly.append(dict(model=name,year=year,**score(actual[mask],predicted.to_numpy()[mask])))
        daily[name]=(np.abs(actual-predicted.to_numpy())/actual).reshape(-1,48).mean(axis=1)*100
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(yearly).to_csv(OUT/'yearly_metrics.csv',index=False)
    component_rows=[dict(model=name,**score(actual,predictions[name].to_numpy())) for name in predictions if name.startswith(('lgbm_','mlp_'))]
    pd.DataFrame(component_rows).to_csv(OUT/'component_metrics.csv',index=False)
    predictions.to_csv(OUT/'predictions.csv',index_label='timestamp')
    comparisons={name:daily['current']-values for name,values in daily.items() if name!='current'}
    comparisons['both5_vs_MLP5']=daily['bagged_MLP_only_5']-daily['both_bagged_5']
    rng=np.random.default_rng(9417);n=len(daily['current']);samples={name:[] for name in comparisons}
    for _ in range(20):
        starts=rng.integers(0,n,size=(250,int(np.ceil(n/7))))
        indices=((starts[:,:,None]+np.arange(7))%n).reshape(250,-1)[:,:n]
        for name,delta in comparisons.items():samples[name].extend(delta[indices].mean(axis=1))
    intervals={name:dict(MAPE_reduction_pp=float(delta.mean()),
        paired_7day_CI95_pp=np.percentile(samples[name],[2.5,97.5]).tolist()) for name,delta in comparisons.items()}
    (OUT/'significance.json').write_text(json.dumps(intervals,indent=2))
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete',seconds=time.monotonic()-started,
        evaluation_start='2015-01-01',evaluation_end='2024-12-31',seeds=SEEDS,holdout_evaluated=False,
        production_artifacts_modified=False),indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=['lgbm','mlp']);parser.add_argument('--seed',type=int);parser.add_argument('--data',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        if args.worker:worker(args.worker,args.seed,args.data)
        else:main()
