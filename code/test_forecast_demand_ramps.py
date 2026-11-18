"""Six demand-forecast ramp features; 2014 screen then frozen 2015–2020 check."""
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
OUT=ROOT/'results/forecast_demand_ramps_2014_2020'


def ramp_features(x,prediction):
    """Clamp backward neighbours to the market-day opening at day boundaries."""
    if not x.index.equals(prediction.index):
        raise ValueError('Forecast and feature indices must align')
    days=market_days(x.index)
    result={}
    for offset in (1,2,4):
        before=prediction.groupby(days).shift(offset)
        before=before.fillna(prediction.groupby(days).transform('first'))
        yesterday=x.nsw_demand_lag_1
        yesterday_before=yesterday.groupby(days).shift(offset)
        yesterday_before=yesterday_before.fillna(yesterday.groupby(days).transform('first'))
        ramp=prediction-before
        result[f'forecast_demand_ramp_{offset}']=ramp
        result[f'forecast_demand_ramp_{offset}_change_1d']=ramp-(yesterday-yesterday_before)
    return pd.DataFrame(result,index=x.index)


def verify_ramps():
    index=pd.date_range('2014-01-01 00:30',periods=96,freq='30min')
    forecast=pd.Series(np.tile(np.arange(48)*10.,2),index=index)
    x=pd.DataFrame({'nsw_demand_lag_1':forecast/2},index=index)
    ramps=ramp_features(x,forecast)
    for offset in (1,2,4):
        expected=np.tile(np.minimum(np.arange(48),offset)*10.,2)
        np.testing.assert_array_equal(ramps[f'forecast_demand_ramp_{offset}'],expected)
        np.testing.assert_array_equal(ramps[f'forecast_demand_ramp_{offset}_change_1d'],expected/2)
    modified=forecast.copy(); modified.iloc[48:]+=999
    pd.testing.assert_frame_equal(ramps.iloc[:48],ramp_features(x,modified).iloc[:48])


def fit_predict(features,y,train,test,config,name):
    destination=OUT/f'{name}_probability.csv'
    if destination.exists():
        result=pd.read_csv(destination,index_col='timestamp',parse_dates=True).probability
        assert result.index.equals(features.index[test])
        return result
    print(f'Fitting {name}',flush=True)
    model=fit_model('price',config,features.loc[train],y.loc[features.index[train]])
    result=pd.Series(model['estimator'].predict_proba(features.loc[test,model['features']])[:,1],
                     index=features.index[test],name='probability')
    result.to_csv(destination,index_label='timestamp')
    joblib.dump(model,OUT/f'{name}.joblib')
    return result


def main():
    started=time.monotonic(); verify_ramps()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completion.json').exists():
        print((OUT/'screen_2014.csv').read_text(),flush=True)
        if (OUT/'metrics_2015_2020.csv').exists(): print((OUT/'metrics_2015_2020.csv').read_text(),flush=True)
        return
    config=json.loads((ROOT/'results/final_config.json').read_text())['price']['config']
    (OUT/'plan.json').write_text(json.dumps({'offsets_halfhours':[1,2,4],
        'boundary_rule':'Clamp to same market-day opening for both forecast and yesterday ramp',
        'selection':'Proceed only if 2014 balanced accuracy exceeds five-feature candidate',
        'threshold':config['threshold'],'holdout_evaluated':False,
        'follow_up_idea':'Feed chronological frozen base price probabilities into a separate demand candidate; avoid circular dependencies.'},indent=2))
    frame=load_data(ROOT/'data/nem_weather_halfhourly_2010_2026.csv').loc[:'2021-01-01 00:00:00']
    x,y=training_table(frame)
    early=pd.concat([pd.read_csv(ROOT/f'results/forecast_demand_price_2014/demand_forecasts_{year}.csv',
        index_col='timestamp',parse_dates=True).forecast for year in range(2011,2015)])
    def features_for(prediction):
        base=x.loc[prediction.index]
        result=base.join(added_features(base,prediction)).join(ramp_features(base,prediction))
        assert result.shape[1]==582 and np.isfinite(result.to_numpy()).all()
        return result
    features=features_for(early); days=market_days(features.index)
    probability=fit_predict(features,y,days.year<2014,days.year==2014,config,'ramps_2014')
    saved=pd.read_csv(ROOT/'results/forecast_demand_price_2014/predictions.csv',index_col='timestamp',parse_dates=True)
    assert saved.index.equals(probability.index)
    np.testing.assert_array_equal(saved.actual_up,y.loc[saved.index,'price'])
    forecasts={'current':saved.current_probability,'five_features':saved.forecast_demand_augmented_probability,'eleven_features':probability}
    screen=[dict(model=name,**metrics(saved.actual_up,values>config['threshold'])) for name,values in forecasts.items()]
    pd.DataFrame(screen).to_csv(OUT/'screen_2014.csv',index=False)
    print(pd.DataFrame(screen).to_string(index=False),flush=True)
    if screen[-1]['balanced_accuracy']<=screen[-2]['balanced_accuracy']:
        (OUT/'completion.json').write_text(json.dumps({'status':'screen_no_improvement','seconds':time.monotonic()-started,'holdout_evaluated':False},indent=2))
        return
    print('2014 screen improved: freezing six ramp additions for 2015–2020',flush=True)
    late_index=x.index[market_days(x.index).year>=2015]
    source=ROOT/'results/demand_mlp_10epoch_2015_2024'
    mlp=pd.concat([pd.read_csv(source/f'predictions_{year}.csv',index_col='timestamp',parse_dates=True).predicted for year in range(2015,2021)])
    assert mlp.index.equals(late_index)
    truth=np.load(source/'truth.npz')
    np.testing.assert_array_equal(truth['timestamps'][:len(late_index)],late_index.to_numpy())
    np.testing.assert_array_equal(truth['demand'][:len(late_index)],y.loc[late_index,'demand'])
    lgbm=np.load(ROOT/'results/feature_comparison_2015_2024/current_demand.npy',mmap_mode='r').reshape(-1)[:len(late_index)]
    features=features_for(pd.concat([early,pd.Series(.5*(lgbm+mlp.to_numpy()),index=late_index)]))
    days=market_days(features.index); chunks=[]; audit=[]
    for start,end in ((2015,2015),(2016,2017),(2018,2019),(2020,2020)):
        train=days<pd.Timestamp(f'{start}-01-01'); test=(days.year>=start)&(days.year<=end)
        assert days[train].max()<days[test].min()
        chunks.append(fit_predict(features,y,train,test,config,f'ramps_{start}_{end}'))
        audit.append({'effective_year':start,'training_end':str(days[train].max().date()),'rows':int(train.sum())})
        pd.DataFrame(audit).to_csv(OUT/'chronology_audit.csv',index=False)
    probability=pd.concat(chunks)
    saved=pd.read_csv(ROOT/'results/forecast_demand_price_2015_2020/predictions.csv',index_col='timestamp',parse_dates=True)
    assert saved.index.equals(probability.index)
    np.testing.assert_array_equal(saved.actual_up,y.loc[saved.index,'price'])
    forecasts={'current':saved.current,'five_features':saved.forecast_demand_augmented,'eleven_features':probability}
    totals=[]; yearly=[]; subsets=[]
    raw_price=frame.nsw_price.loc[saved.index]
    for name,values in forecasts.items():
        pred=values>config['threshold']
        totals.append(dict(model=name,**metrics(saved.actual_up,pred)))
        for year in range(2015,2021):
            mask=market_days(saved.index).year==year
            yearly.append(dict(model=name,year=year,**metrics(saved.actual_up.loc[mask],pred.loc[mask])))
        for label,mask in [('spike_gt300',raw_price>300),('nonspike_le300',raw_price<=300)]:
            subsets.append(dict(model=name,subset=label,rows=int(mask.sum()),**metrics(saved.actual_up.loc[mask],pred.loc[mask])))
    pd.DataFrame(totals).to_csv(OUT/'metrics_2015_2020.csv',index=False)
    pd.DataFrame(yearly).to_csv(OUT/'yearly_metrics.csv',index=False)
    pd.DataFrame(subsets).to_csv(OUT/'spike_metrics.csv',index=False)
    pd.DataFrame({'actual_up':saved.actual_up,**forecasts}).to_csv(OUT/'predictions.csv',index_label='timestamp')
    (OUT/'completion.json').write_text(json.dumps({'status':'complete','seconds':time.monotonic()-started,'holdout_evaluated':False},indent=2))
    print(pd.DataFrame(totals).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
