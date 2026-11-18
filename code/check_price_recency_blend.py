"""Frozen 50/50 current/recency price blend on 2015–2020 only."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from run_experiments import load_data,market_days,training_table,fit_model
from test_price_recency_weights import metrics

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    started=time.monotonic()
    output=ROOT/'results/price_recency_blend_2015_2020'
    output.mkdir(parents=True,exist_ok=args.resume)
    source=ROOT/'results/price_recency_2015_2020'
    fixed=json.loads((source/'frozen_config.json').read_text())
    config=fixed['config']
    assert fixed['half_life_years']==2 and fixed['features']==571
    data_path=ROOT/'data/nem_weather_halfhourly_2010_2026.csv'
    frozen={'weights':{'current':.5,'recency_2years':.5},'threshold':config['threshold'],
            'config':config,'evaluation_start':'2015-01-01','evaluation_end':'2020-12-31',
            'data_sha256':digest(data_path),'recency_predictions_sha256':digest(source/'predictions.csv'),
            'features_source_sha256':digest(ROOT/'code/forecast_package/model.py'),
            'trainer_source_sha256':digest(ROOT/'code/forecast_package/train.py'),
            'holdout_evaluated':False,'tuning':False}
    frozen_path=output/'frozen_config.json'
    if frozen_path.exists():
        assert json.loads(frozen_path.read_text())==frozen,'Resume inputs changed'
    else:
        frozen_path.write_text(json.dumps(frozen,indent=2))
    saved=pd.read_csv(source/'predictions.csv',index_col='timestamp',parse_dates=True)
    frame=load_data(data_path).loc[:'2021-01-01 00:00:00']
    x,y=training_table(frame)
    assert x.shape[1]==571
    days=market_days(x.index)
    eval_mask=days.year>=2015
    index=x.index[eval_mask]
    assert index.equals(saved.index)
    np.testing.assert_array_equal(y.loc[index,'price'],saved.actual_up)
    probability=pd.Series(index=index,dtype=float)
    audit=[]
    for start,end in ((2015,2015),(2016,2017),(2018,2019),(2020,2020)):
        train=days<pd.Timestamp(f'{start}-01-01')
        test=(days.year>=start)&(days.year<=end)
        assert days[train].max()<days[test].min()
        path=output/f'current_probabilities_{start}_{end}.csv'
        t=time.monotonic()
        if args.resume and path.exists():
            chunk=pd.read_csv(path,index_col='timestamp',parse_dates=True)
            assert chunk.index.equals(x.index[test])
            values=chunk.probability.to_numpy()
        else:
            print(f'Rebuilding current classifier for {start}–{end}; training ends {days[train].max().date()}',flush=True)
            model=fit_model('price',config,x.loc[train],y.loc[train])
            values=model['estimator'].predict_proba(x.loc[test,model['features']])[:,1]
            model['training_end']=str(days[train].max().date())
            joblib.dump(model,output/f'current_model_{start}.joblib')
            pd.DataFrame({'probability':values},index=x.index[test]).to_csv(path,index_label='timestamp')
        np.testing.assert_array_equal((values>config['threshold']).astype(int),saved.loc[x.index[test],'reference_up'])
        probability.loc[x.index[test]]=values
        audit.append({'effective_year':start,'training_end':str(days[train].max().date()),
                      'training_rows':int(train.sum()),'seconds':time.monotonic()-t})
        pd.DataFrame(audit).to_csv(output/'retraining_audit.csv',index=False)
        print(f'Completed {start}–{end}; reproduced saved current-model labels exactly',flush=True)
    assert probability.notna().all()
    actual=saved.actual_up.to_numpy()
    blend=.5*probability.to_numpy()+.5*saved.recency_probability.to_numpy()
    forecasts={'current':probability.to_numpy()>config['threshold'],
               'recency_2years':saved.recency_up.to_numpy(),
               'blend_50_50':blend>config['threshold']}
    rows=[];yearly=[]
    eval_days=market_days(index)
    for name,pred in forecasts.items():
        rows.append({'model':name,**metrics(actual,pred)})
        for year in range(2015,2021):
            mask=eval_days.year==year
            yearly.append({'model':name,'year':year,**metrics(actual[mask],pred[mask])})
    pd.DataFrame(rows).to_csv(output/'aggregate_metrics.csv',index=False)
    pd.DataFrame(yearly).to_csv(output/'yearly_metrics.csv',index=False)
    pd.DataFrame({'actual_up':actual,'current_probability':probability,
                  'recency_probability':saved.recency_probability,'blend_probability':blend,
                  'blend_up':forecasts['blend_50_50'].astype(int)},index=index).to_csv(
        output/'predictions.csv',index_label='timestamp')
    (output/'completion.json').write_text(json.dumps({'status':'passed',
        'seconds':time.monotonic()-started,'saved_current_labels_reproduced_exactly':True},indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(pd.DataFrame(yearly).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
