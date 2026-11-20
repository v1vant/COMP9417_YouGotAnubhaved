"""Frozen single-LightGBM/five-MLP demand blend on the reused 2025–Oct 2026 holdout."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'code/forecast_package'))
OUT=ROOT/'results/demand_single_lgbm_5mlp_holdout'
PERIODS=[(2025,'2023-12-31'),(2026,'2025-12-31')]
SEEDS=list(range(9417,9422))


def score(actual,predicted):
    return dict(MAPE=float(np.mean(np.abs(actual-predicted)/actual)*100),MAE=float(np.mean(np.abs(actual-predicted))))


def worker(seed,data):
    import joblib
    import torch
    from model import market_days
    from demand_mlp import fit_demand_mlp,predict_demand_mlp
    x,y=pd.read_pickle(data);days=market_days(x.index)
    for year,end in PERIODS:
        destination=OUT/f'mlp_{seed}_{year}.npy';metadata=OUT/f'mlp_{seed}_{year}_fit.json'
        train=days<=pd.Timestamp(end);test=days.year==year
        assert days[train].max()<days[test].min()
        if destination.exists() and metadata.exists():
            record=json.loads(metadata.read_text());assert record['training_end']==end and record['seed']==seed
            continue
        started=time.monotonic();print(f'MLP seed {seed} for {year}, training through {end}',flush=True)
        bundle=fit_demand_mlp(x.loc[train],y.loc[train],epochs=10,seed=seed)
        prediction=predict_demand_mlp(bundle,x.loc[test]);assert np.isfinite(prediction).all()
        np.save(destination,prediction)
        torch.save(bundle['model'].state_dict(),OUT/f'mlp_{seed}_{year}.pt')
        joblib.dump({'features':bundle['features'],'scaler':bundle['scaler']},OUT/f'mlp_{seed}_{year}_preprocessing.joblib')
        metadata.write_text(json.dumps(dict(seed=seed,forecast_year=year,training_end=end,
            training_rows=int(train.sum()),epochs=10,seconds=time.monotonic()-started),indent=2))


def main():
    from model import load_data,market_days,training_table
    OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    if (OUT/'completion.json').exists():print((OUT/'metrics.csv').read_text(),flush=True);return
    plan=dict(seeds=SEEDS,MLP_epochs=10,demand_features=405,blend_weights={'single_LightGBM':.5,'mean_5_MLP':.5},
        periods=PERIODS,evaluation_start='2025-01-01',evaluation_end='2026-10-31',
        tuning_on_holdout=False,holdout_previously_inspected=True,price='C2 unchanged',
        source_sha256=hashlib.sha256((ROOT/'code/demand_mlp.py').read_bytes()).hexdigest(),
        comparison='Frozen current single-LightGBM/single-MLP vs single-LightGBM/five-MLP only')
    path=OUT/'frozen_plan.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(plan))
    else:path.write_text(json.dumps(plan,indent=2))
    artifacts=ROOT/'code/forecast_package/artifacts'
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts.iterdir() if p.is_file()}
    frame=load_data(ROOT/'data/nem_weather_halfhourly_2010_2026.csv').loc[:'2026-11-01 00:00:00']
    x,y=training_table(frame);columns=json.loads((ROOT/'code/demand_feature_columns.json').read_text());x=x.loc[:,columns]
    assert x.shape[1]==405
    days=market_days(x.index);evaluation=days.year>=2025;index=x.index[evaluation];actual=y.loc[index,'demand'].to_numpy()
    assert days[evaluation].max()==pd.Timestamp('2026-10-31')
    source=ROOT/'results/frozen_demand_blend_holdout'
    cached=pd.read_csv(source/'predictions.csv',index_col='timestamp',parse_dates=True)
    assert cached.index.equals(index);np.testing.assert_array_equal(cached.actual,actual)
    reference=ROOT/'results/holdout_original_vs_current'
    truth=np.load(reference/'truth.npz');np.testing.assert_array_equal(truth['demand'].ravel(),actual)
    np.testing.assert_array_equal(truth['days'],days[evaluation].unique().to_numpy())
    lgbm=np.load(reference/'current_demand.npy').ravel()
    current=np.load(source/'blend_demand.npy').ravel()
    np.testing.assert_allclose(lgbm,cached.current_LightGBM,rtol=0,atol=1e-10)
    np.testing.assert_allclose(current,.5*(lgbm+cached.MLP_10_epochs.to_numpy()),rtol=0,atol=1e-10)
    predictions=pd.DataFrame({'actual':actual,'lgbm':lgbm,'mlp_9417':cached.MLP_10_epochs,'current':current},index=index)
    audit=pd.read_csv(source/'retraining_audit.csv')
    assert list(audit.training_end)==[end for _,end in PERIODS]
    with tempfile.TemporaryDirectory(prefix='demand_holdout_',dir='/private/tmp') as directory:
        data=Path(directory)/'data.pkl';pd.to_pickle((x,y.demand),data)
        for seed in SEEDS[1:]:
            print(f'Starting MLP seed {seed}',flush=True)
            subprocess.run([sys.executable,'-u',__file__,'--worker',str(seed),str(data)],check=True)
            values=np.concatenate([np.load(OUT/f'mlp_{seed}_{year}.npy') for year,_ in PERIODS])
            assert values.shape==actual.shape;predictions[f'mlp_{seed}']=values
    predictions['mlp_mean5']=predictions[[f'mlp_{seed}' for seed in SEEDS]].mean(axis=1)
    predictions['candidate']=.5*(predictions.lgbm+predictions.mlp_mean5)
    totals=[];periods=[]
    for name in ['current','candidate','lgbm','mlp_9417','mlp_mean5']:
        values=predictions[name].to_numpy();totals.append(dict(model=name,**score(actual,values)))
        for frequency in ['Y','M']:
            period=days[evaluation].to_period(frequency)
            for label in period.unique():
                mask=period==label
                periods.append(dict(model=name,frequency=frequency,period=str(label),**score(actual[mask],values[mask])))
    pd.DataFrame(totals).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(periods).to_csv(OUT/'period_metrics.csv',index=False)
    predictions.to_csv(OUT/'predictions.csv',index_label='timestamp')
    # Paired circular seven-day block bootstrap over 669 complete market days.
    y=actual.reshape(-1,48);old=current.reshape(-1,48);new=predictions.candidate.to_numpy().reshape(-1,48)
    delta=(np.abs(y-old)/y-np.abs(y-new)/y).mean(axis=1)*100
    mae_delta=(np.abs(y-old)-np.abs(y-new)).mean(axis=1)
    rng=np.random.default_rng(9417);n=len(delta)
    starts=rng.integers(0,n,size=(5000,int(np.ceil(n/7))))
    indices=((starts[:,:,None]+np.arange(7))%n).reshape(5000,-1)[:,:n]
    result=dict(MAPE_reduction_pp=float(delta.mean()),relative_MAPE_reduction_percent=float(delta.mean()/totals[0]['MAPE']*100),
        paired_7day_MAPE_CI95_pp=np.percentile(delta[indices].mean(axis=1),[2.5,97.5]).tolist(),
        MAE_reduction_MW=float(mae_delta.mean()),paired_7day_MAE_CI95_MW=np.percentile(mae_delta[indices].mean(axis=1),[2.5,97.5]).tolist(),
        days=n,replicates=5000,seed=9417)
    (OUT/'significance.json').write_text(json.dumps(result,indent=2))
    after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts.iterdir() if p.is_file()}
    assert before==after
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete',seconds=time.monotonic()-started,
        holdout_evaluated=True,tuning_on_holdout=False,production_artifacts_unchanged=True),indent=2))
    print(pd.DataFrame(totals).to_string(index=False),flush=True);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(int(sys.argv[2]),Path(sys.argv[3]))
        else:main()
