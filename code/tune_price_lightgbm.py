"""Small fixed price-model screen on 2014, then a 2015–2020 winner check."""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits

from run_experiments import load_data,market_days,training_table
from test_price_recency_weights import metrics

ROOT=Path(__file__).resolve().parents[1]
CANDIDATES={
    'stronger_regularisation':{'leaves':31,'min_child_samples':320,'reg_lambda':20.,'colsample_bytree':.85},
    'fewer_leaves':{'leaves':15,'min_child_samples':160,'reg_lambda':8.,'colsample_bytree':.85},
    'lower_feature_sampling':{'leaves':31,'min_child_samples':160,'reg_lambda':8.,'colsample_bytree':.60},
}


def fit_candidate(x,y,config,variant):
    model=LGBMClassifier(objective='binary',n_estimators=config['trees'],
        learning_rate=config['learning_rate'],num_leaves=variant['leaves'],
        min_child_samples=variant['min_child_samples'],reg_lambda=variant['reg_lambda'],
        colsample_bytree=variant['colsample_bytree'],subsample=.85,subsample_freq=1,
        max_bin=127,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True,
        random_state=9417,class_weight='balanced')
    model.fit(x,y)
    return model


def main():
    started=time.monotonic()
    output=ROOT/'results/price_lightgbm_tuning_2014_2020'
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'results/final_config.json').read_text())['price']['config']
    (output/'frozen_search.json').write_text(json.dumps({'candidates':CANDIDATES,
        'base_config':config,'selection_year':2014,'historical_check':[2015,2020],
        'winner_rule':'Highest 2014 balanced accuracy; historical check only if above current baseline.',
        'spike_features':False,'recency_weighting':False,'holdout_evaluated':False},indent=2))
    frame=load_data(ROOT/'data/nem_weather_halfhourly_2010_2026.csv').loc[:'2021-01-01 00:00:00']
    x,y=training_table(frame)
    assert x.shape[1]==571
    days=market_days(x.index)
    train,valid=days.year<=2013,days.year==2014
    cached=pd.read_csv(ROOT/'results/price_recency_2014/predictions.csv',index_col='timestamp',parse_dates=True)
    assert cached.index.equals(x.index[valid])
    actual=y.loc[valid,'price'].to_numpy()
    np.testing.assert_array_equal(actual,cached.actual_up)
    baseline=metrics(actual,cached.current_probability.to_numpy()>config['threshold'])
    screen=[dict(candidate='current',**baseline)]
    for name,variant in CANDIDATES.items():
        print(f'2014 screen: {name}',flush=True)
        model=fit_candidate(x.loc[train],y.loc[train,'price'],config,variant)
        probability=model.predict_proba(x.loc[valid])[:,1]
        screen.append(dict(candidate=name,**metrics(actual,probability>config['threshold'])))
        pd.DataFrame(screen).to_csv(output/'screen_2014.csv',index=False)
        pd.DataFrame({'actual_up':actual,'probability':probability},index=x.index[valid]).to_csv(
            output/f'{name}_2014.csv',index_label='timestamp')
        joblib.dump({'estimator':model,'features':list(x),'variant':variant,
                     'config':config,'training_end':'2013-12-31'},output/f'{name}_2013.joblib')
        print(screen[-1],flush=True)
    winner=max(screen[1:],key=lambda row:row['balanced_accuracy'])
    selected=winner['candidate']
    promising=winner['balanced_accuracy']>baseline['balanced_accuracy']
    (output/'selection.json').write_text(json.dumps({'selected':selected,'promising':promising,
        'selected_score':winner,'reference_score':baseline},indent=2))
    if not promising:
        print('No candidate improved 2014: stopping before historical check.',flush=True)
        (output/'completion.json').write_text(json.dumps({'status':'screen_complete_no_improvement',
            'seconds':time.monotonic()-started},indent=2))
        return
    variant=CANDIDATES[selected]
    print(f'Frozen winner: {selected}; starting 2015–2020',flush=True)
    evaluation=days.year>=2015
    index=x.index[evaluation]
    actual=y.loc[index,'price'].to_numpy()
    saved=pd.read_csv(ROOT/'results/price_recency_blend_2015_2020/predictions.csv',
                      index_col='timestamp',parse_dates=True)
    assert index.equals(saved.index)
    np.testing.assert_array_equal(actual,saved.actual_up)
    probability=pd.Series(index=index,dtype=float)
    audit=[]
    for start,end in ((2015,2015),(2016,2017),(2018,2019),(2020,2020)):
        train=days<pd.Timestamp(f'{start}-01-01')
        test=(days.year>=start)&(days.year<=end)
        assert days[train].max()<days[test].min()
        print(f'Fitting {selected} for {start}–{end}',flush=True)
        model=fit_candidate(x.loc[train],y.loc[train,'price'],config,variant)
        values=model.predict_proba(x.loc[test])[:,1]
        probability.loc[x.index[test]]=values
        joblib.dump({'estimator':model,'features':list(x),'variant':variant,'config':config,
                     'training_end':str(days[train].max().date())},output/f'{selected}_{start}.joblib')
        pd.DataFrame({'probability':values},index=x.index[test]).to_csv(
            output/f'probabilities_{start}_{end}.csv',index_label='timestamp')
        audit.append({'effective_year':start,'training_end':str(days[train].max().date()),'rows':int(train.sum())})
        pd.DataFrame(audit).to_csv(output/'retraining_audit.csv',index=False)
    assert probability.notna().all()
    rows=[];yearly=[]
    forecasts={'current':saved.current_probability.to_numpy()>config['threshold'],
               selected:probability.to_numpy()>config['threshold']}
    for name,pred in forecasts.items():
        rows.append(dict(candidate=name,**metrics(actual,pred)))
        for year in range(2015,2021):
            mask=days[evaluation].year==year
            yearly.append(dict(candidate=name,year=year,**metrics(actual[mask],pred[mask])))
    pd.DataFrame(rows).to_csv(output/'historical_2015_2020.csv',index=False)
    pd.DataFrame(yearly).to_csv(output/'yearly_metrics.csv',index=False)
    pd.DataFrame({'actual_up':actual,'current_probability':saved.current_probability,
                  'candidate_probability':probability},index=index).to_csv(output/'predictions.csv',index_label='timestamp')
    (output/'completion.json').write_text(json.dumps({'status':'passed','seconds':time.monotonic()-started,
        'holdout_evaluated':False},indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
