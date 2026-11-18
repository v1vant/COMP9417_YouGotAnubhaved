"""Frozen forecast-demand price candidate: chronological 2025–October 2026 holdout comparison."""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_experiments import load_data, market_days, training_table, fit_model
from test_forecast_demand_price import added_features
from test_price_recency_weights import metrics

ROOT=Path(__file__).resolve().parents[1]


def main():
    started=time.monotonic()
    output=ROOT/'results/forecast_demand_price_holdout'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'completion.json').exists():
        print((output/'metrics.csv').read_text(),flush=True)
        return
    config=json.loads((ROOT/'results/final_config.json').read_text())['price']['config']
    frame=load_data(ROOT/'data/nem_weather_halfhourly_2010_2026.csv').loc[:'2026-11-01 00:00:00']
    x,y=training_table(frame)
    days=market_days(x.index)
    early=pd.concat([pd.read_csv(ROOT/f'results/forecast_demand_price_2014/demand_forecasts_{year}.csv',
        index_col='timestamp',parse_dates=True).forecast for year in range(2011,2015)])
    source=ROOT/'results/demand_mlp_10epoch_2015_2024'
    dev_index=x.index[(days.year>=2015)&(days.year<=2024)]
    mlp=pd.concat([pd.read_csv(source/f'predictions_{year}.csv',index_col='timestamp',parse_dates=True).predicted
        for year in range(2015,2025)])
    assert mlp.index.equals(dev_index)
    truth=np.load(source/'truth.npz')
    np.testing.assert_array_equal(truth['timestamps'],dev_index.to_numpy())
    np.testing.assert_array_equal(truth['demand'],y.loc[dev_index,'demand'])
    lgbm=np.load(ROOT/'results/feature_comparison_2015_2024/current_demand.npy').reshape(-1)
    holdout=pd.read_csv(ROOT/'results/frozen_demand_blend_holdout/predictions.csv',index_col='timestamp',parse_dates=True)
    eval_index=x.index[days.year>=2025]
    assert holdout.index.equals(eval_index)
    np.testing.assert_array_equal(holdout.actual,y.loc[eval_index,'demand'])
    np.testing.assert_allclose(holdout.frozen_50_50,.5*(holdout.current_LightGBM+holdout.MLP_10_epochs))
    prediction=pd.concat([early,pd.Series(.5*(lgbm+mlp.to_numpy()),index=dev_index),holdout.frozen_50_50])
    extra=added_features(x.loc[prediction.index],prediction)
    features=x.loc[extra.index].join(extra)
    assert features.shape[1]==576 and np.isfinite(features.to_numpy()).all()
    feature_days=market_days(features.index)
    reference=ROOT/'results/holdout_original_vs_current'
    truth=np.load(reference/'truth.npz')
    np.testing.assert_array_equal(truth['days'],market_days(eval_index).unique().to_numpy())
    np.testing.assert_array_equal(truth['price'].reshape(-1),y.loc[eval_index,'price'])
    current=np.load(reference/'current_price.npy').reshape(-1)
    assert len(current)==len(eval_index) and set(np.unique(current)).issubset({0,1})
    saved=pd.DataFrame({'actual_up':y.loc[eval_index,'price'], 'current_up':current},index=eval_index)
    (output/'frozen_config.json').write_text(json.dumps({'price_config':config,'extra_features':list(extra),
        'initial_training_end':'2023-12-31','retrain':'2026-01-01','retrain_training_end':'2025-12-31',
        'ramp_features':False,'tuning_on_holdout':False,'bootstrap':'paired circular 7-day moving blocks, 5000 replicates, seed 9417'},indent=2))
    probability=pd.Series(index=eval_index,dtype=float)
    baseline=pd.Series(index=eval_index,dtype=float)
    retrains=[]
    for start,end in ((2025,2025),(2026,2026)):
        training_year=2024 if start==2025 else start
        train=feature_days<pd.Timestamp(f'{training_year}-01-01')
        test=(feature_days.year>=start)&(feature_days.year<=end)
        assert feature_days[train].max()<feature_days[test].min()
        for name,columns,target in [('matched_reference',list(x),baseline),
                                     ('forecast_demand_augmented',list(features),probability)]:
            destination=output/f'{name}_{start}_{end}.csv'
            if destination.exists():
                values=pd.read_csv(destination,index_col='timestamp',parse_dates=True).probability
                assert values.index.equals(features.index[test])
                target.loc[values.index]=values
                continue
            if start==2025:
                print(f'Reusing frozen 2024 {name} model for 2025',flush=True)
                model=joblib.load(ROOT/f'results/forecast_demand_price_2021_2024/{name}_2024.joblib')
                assert model['features']==columns and model['config']==config
            else:
                print(f'Fitting {name} for {start}–{end}',flush=True)
                model=fit_model('price',config,features.loc[train,columns],y.loc[features.index[train]])
            values=model['estimator'].predict_proba(features.loc[test,model['features']])[:,1]
            target.loc[features.index[test]]=values
            pd.Series(values,index=features.index[test],name='probability').to_csv(destination,index_label='timestamp')
            joblib.dump(model,output/f'{name}_{start}.joblib')
        retrains.append({'effective_year':start,'training_end':str(feature_days[train].max().date()),'rows':int(train.sum())})
        pd.DataFrame(retrains).to_csv(output/'chronology_audit.csv',index=False)
    assert probability.notna().all() and baseline.notna().all()
    forecasts={'current':saved.current_up,'matched_reference':baseline,'forecast_demand_augmented':probability}
    rows=[]; yearly=[]
    for name,values in forecasts.items():
        pred=values.to_numpy()>config['threshold']
        rows.append(dict(model=name,**metrics(saved.actual_up,pred)))
        for year in range(2025,2027):
            mask=market_days(eval_index).year==year
            yearly.append(dict(model=name,year=year,**metrics(saved.actual_up.to_numpy()[mask],pred[mask])))
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(yearly).to_csv(output/'yearly_metrics.csv',index=False)
    pd.DataFrame({'actual_up':saved.actual_up,**forecasts}).to_csv(output/'predictions.csv',index_label='timestamp')
    # Aggregate sufficient statistics by day, then resample paired contiguous
    # weekly blocks so both models see exactly the same sampled observations.
    actual=saved.actual_up.to_numpy().astype(int)
    current_pred=saved.current_up.to_numpy().astype(int)
    candidate_pred=(probability.to_numpy()>config['threshold']).astype(int)
    def daily_counts(pred):
        return np.column_stack([actual==0,actual==1,(actual==0)&(pred==0),(actual==1)&(pred==1),pred==actual]).reshape(-1,48,5).sum(axis=1)
    c=daily_counts(current_pred); n=daily_counts(candidate_pred)
    def scores(counts):
        accuracy=100*counts[...,4]/(counts[...,0]+counts[...,1])
        ba=50*(counts[...,2]/counts[...,0]+counts[...,3]/counts[...,1])
        return np.stack([accuracy,ba],axis=-1)
    rng=np.random.default_rng(9417); size=len(c)
    starts=rng.integers(0,size,size=(5000,int(np.ceil(size/7))))
    indices=((starts[:,:,None]+np.arange(7))%size).reshape(5000,-1)[:,:size]
    differences=scores(n[indices].sum(axis=1))-scores(c[indices].sum(axis=1))
    observed=scores(n.sum(axis=0))-scores(c.sum(axis=0))
    intervals=np.percentile(differences,[2.5,97.5],axis=0)
    significance={name:{'improvement_pp':float(observed[k]),'paired_7day_bootstrap_CI95_pp':intervals[:,k].tolist()}
                  for k,name in enumerate(['accuracy','balanced_accuracy'])}
    significance.update(days=size,replicates=5000,seed=9417)
    (output/'significance.json').write_text(json.dumps(significance,indent=2))
    (output/'completion.json').write_text(json.dumps({'seconds':time.monotonic()-started,'holdout_evaluated':True,
        'features_added':list(extra),'config':config,'evaluation_start':'2025-01-01','evaluation_end':'2026-10-31',
        'current_column_contains':'hard labels from cached current policy, not probabilities'},indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
