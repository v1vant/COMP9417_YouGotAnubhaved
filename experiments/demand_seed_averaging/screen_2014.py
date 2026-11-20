"""Fixed three-seed demand screen: train <=2013, evaluate 2014 only."""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/'code'
sys.path.insert(0,str(CODE))
sys.path.insert(0,str(CODE/'forecast_package'))
OUT=ROOT/'results/demand_seed_averaging_2014'
SEEDS=[9417,9418,9419]


def worker(seed):
    import torch
    from demand_mlp import fit_demand_mlp,predict_demand_mlp
    train_x,train_y,test_x=pd.read_pickle(OUT/'mlp_inputs.pkl')
    bundle=fit_demand_mlp(train_x,train_y,epochs=10,seed=seed)
    pd.Series(predict_demand_mlp(bundle,test_x),index=test_x.index,name='prediction').to_csv(
        OUT/f'mlp_{seed}.csv',index_label='timestamp')
    torch.save(bundle['model'].state_dict(),OUT/f'mlp_{seed}.pt')
    (OUT/f'mlp_{seed}_training.json').write_text(json.dumps({'seed':seed,'epochs':10,
        'training_rows':len(train_x),'training_end':str(train_x.index.max()),
        'epoch_seconds':bundle['epoch_seconds']},indent=2))


def score(actual,predicted):
    return {'MAPE':float(np.mean(np.abs(actual-predicted)/actual)*100),
            'MAE':float(np.mean(np.abs(actual-predicted)))}


def main():
    import joblib
    import train as training
    from model import load_data,market_days,training_table,predict_task
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completion.json').exists():
        print((OUT/'metrics.csv').read_text(),flush=True);return
    plan=json.loads((Path(__file__).parent/'plan.json').read_text())
    plan['status']='running_2014';plan['protocol']['no_training_authorized_yet']=False
    (Path(__file__).parent/'plan.json').write_text(json.dumps(plan,indent=2))
    (OUT/'frozen_plan.json').write_text(json.dumps(plan,indent=2))
    frame=load_data(ROOT/'data/nem_weather_halfhourly_2010_2026.csv').loc[:'2015-01-01 00:00:00']
    x,y=training_table(frame)
    columns=json.loads((CODE/'demand_feature_columns.json').read_text())
    assert len(columns)==405
    days=market_days(x.index);train=days.year<=2013;test=days.year==2014
    assert days[train].max()<days[test].min() and days.max()==pd.Timestamp('2014-12-31')
    config=json.loads((ROOT/'results/final_config.json').read_text())['demand']['config']
    pd.to_pickle((x.loc[train,columns],y.loc[train,'demand'],x.loc[test,columns]),OUT/'mlp_inputs.pkl')
    output=pd.DataFrame({'actual':y.loc[test,'demand']},index=x.index[test]);audit=[]
    for seed in SEEDS:
        destination=OUT/f'lgbm_{seed}.csv'
        if not destination.exists():
            print(f'Fitting LightGBM seed {seed}',flush=True)
            training.SEED=seed
            model=training.fit_model('demand',config,x.loc[train],y.loc[train])
            assert model['features']==columns
            pd.Series(predict_task(model,x.loc[test]),index=x.index[test],name='prediction').to_csv(destination,index_label='timestamp')
            joblib.dump(model,OUT/f'lgbm_{seed}.joblib',compress=3)
        print(f'Preparing MLP seed {seed}',flush=True)
        if not (OUT/f'mlp_{seed}.csv').exists():
            subprocess.run([sys.executable,'-u',__file__,'--mlp-worker',str(seed)],check=True)
        for kind in ['lgbm','mlp']:
            predicted=pd.read_csv(OUT/f'{kind}_{seed}.csv',index_col='timestamp',parse_dates=True).prediction
            assert predicted.index.equals(output.index) and np.isfinite(predicted.to_numpy()).all()
            output[f'{kind}_{seed}']=predicted
        audit.append({'seed':seed,'training_start':str(days[train].min().date()),
            'training_end':str(days[train].max().date()),'training_rows':int(train.sum()),'evaluation_year':2014})
        pd.DataFrame(audit).to_csv(OUT/'chronology_audit.csv',index=False)
        print(f'Completed seed {seed}',flush=True)
    output['lgbm_mean3']=output[[f'lgbm_{seed}' for seed in SEEDS]].mean(axis=1)
    output['mlp_mean3']=output[[f'mlp_{seed}' for seed in SEEDS]].mean(axis=1)
    combinations={'current':('lgbm_9417','mlp_9417'),
        'bagged_LGBM_only':('lgbm_mean3','mlp_9417'),
        'bagged_MLP_only':('lgbm_9417','mlp_mean3'),
        'both_bagged':('lgbm_mean3','mlp_mean3')}
    rows=[]
    for name,(a,b) in combinations.items():
        output[name]=.5*(output[a]+output[b])
        rows.append(dict(model=name,**score(output.actual,output[name])))
    components=[dict(model=name,**score(output.actual,output[name])) for name in
        [*[f'{kind}_{seed}' for kind in ['lgbm','mlp'] for seed in SEEDS],'lgbm_mean3','mlp_mean3']]
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(components).to_csv(OUT/'component_metrics.csv',index=False)
    output.to_csv(OUT/'predictions.csv',index_label='timestamp')
    # Paired weekly blocks, maintaining temporal dependence and identical sampled days.
    actual=output.actual.to_numpy().reshape(-1,48)
    baseline=np.mean(np.abs(actual-output.current.to_numpy().reshape(-1,48))/actual,axis=1)*100
    rng=np.random.default_rng(9417);n=len(actual)
    starts=rng.integers(0,n,size=(5000,int(np.ceil(n/7))))
    indices=((starts[:,:,None]+np.arange(7))%n).reshape(5000,-1)[:,:n]
    intervals={}
    for name in list(combinations)[1:]:
        loss=np.mean(np.abs(actual-output[name].to_numpy().reshape(-1,48))/actual,axis=1)*100
        difference=baseline-loss
        intervals[name]={'MAPE_reduction_pp':float(difference.mean()),
            'paired_7day_CI95_pp':np.percentile(difference[indices].mean(axis=1),[2.5,97.5]).tolist()}
    (OUT/'significance.json').write_text(json.dumps(intervals,indent=2))
    (OUT/'mlp_inputs.pkl').unlink()
    (OUT/'completion.json').write_text(json.dumps({'status':'complete','seconds':time.monotonic()-started,
        'seeds':SEEDS,'training_end':'2013-12-31','evaluation_year':2014,'holdout_evaluated':False,
        'production_artifacts_modified':False},indent=2))
    plan['status']='completed_2014';(Path(__file__).parent/'plan.json').write_text(json.dumps(plan,indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        if len(sys.argv)>1 and sys.argv[1]=='--mlp-worker':worker(int(sys.argv[2]))
        else:main()
