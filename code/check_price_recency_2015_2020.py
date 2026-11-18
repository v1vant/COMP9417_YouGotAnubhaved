"""Frozen two-year recency weighting versus saved current price forecasts."""
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from test_price_recency_weights import fit_recency_classifier,metrics
from run_experiments import load_data,market_days,training_table

ROOT=Path(__file__).resolve().parents[1]


def main():
    started=time.monotonic()
    output=ROOT/'results/price_recency_2015_2020'
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'results/final_config.json').read_text())['price']['config']
    frozen={'half_life_years':2,'config':config,'features':571,
            'evaluation_start':'2015-01-01','evaluation_end':'2020-12-31',
            'fit_effective_years':[2015,2016,2018,2020],
            'class_weighting':'Equal total sample weight per class after recency weighting',
            'spike_features_enabled':False,'holdout_evaluated':False}
    (output/'frozen_config.json').write_text(json.dumps(frozen,indent=2))
    frame=load_data(ROOT/'data/nem_weather_halfhourly_2010_2026.csv').loc[:'2021-01-01 00:00:00']
    features,labels=training_table(frame)
    assert features.shape[1]==571
    days=market_days(features.index)
    evaluation=days.year>=2015
    index=features.index[evaluation]
    actual=labels.loc[index,'price'].to_numpy()
    reference=np.load(ROOT/'results/feature_comparison_2015_2024/current_price.npy').ravel()[:len(actual)]
    assert len(reference)==len(actual)
    probability=pd.Series(index=index,dtype=float)
    audit=[]
    for start,end in ((2015,2015),(2016,2017),(2018,2019),(2020,2020)):
        fit_start=time.monotonic()
        train=days<pd.Timestamp(f'{start}-01-01')
        test=(days.year>=start)&(days.year<=end)
        assert days[train].max()<days[test].min()
        print(f'Fitting recency model for {start}–{end}; data through {days[train].max().date()}',flush=True)
        model=fit_recency_classifier(features.loc[train],labels.loc[train,'price'],config,2.)
        predicted=model.predict_proba(features.loc[test])[:,model.classes_.tolist().index(1)]
        probability.loc[features.index[test]]=predicted
        bundle={'estimator':model,'features':list(features.columns),'config':config,
                'half_life_years':2,'training_start':str(days[train].min().date()),
                'training_end':str(days[train].max().date())}
        joblib.dump(bundle,output/f'model_{start}.joblib')
        pd.DataFrame({'probability':predicted},index=features.index[test]).to_csv(
            output/f'probabilities_{start}_{end}.csv',index_label='timestamp')
        row={'effective_year':start,'training_end':bundle['training_end'],
             'training_rows':int(train.sum()),'seconds':time.monotonic()-fit_start}
        audit.append(row)
        pd.DataFrame(audit).to_csv(output/'retraining_audit.csv',index=False)
        print(f'Completed {start}–{end}: {metrics(labels.loc[test,"price"].to_numpy(),predicted>config["threshold"])}',flush=True)
    assert probability.notna().all() and probability.between(0,1).all()
    candidate=(probability.to_numpy()>config['threshold']).astype(int)
    rows=[];yearly=[]
    for name,predicted in (('current_LightGBM',reference),('recency_2years',candidate)):
        rows.append({'model':name,**metrics(actual,predicted)})
        for year in range(2015,2021):
            mask=days[evaluation].year==year
            yearly.append({'model':name,'year':year,**metrics(actual[mask],predicted[mask])})
    pd.DataFrame(rows).to_csv(output/'aggregate_metrics.csv',index=False)
    pd.DataFrame(yearly).to_csv(output/'yearly_metrics.csv',index=False)
    pd.DataFrame({'actual_up':actual,'reference_up':reference,'recency_probability':probability,
                  'recency_up':candidate},index=index).to_csv(output/'predictions.csv',index_label='timestamp')
    (output/'completion.json').write_text(json.dumps({'status':'passed','seconds':time.monotonic()-started,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(pd.DataFrame(yearly).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
