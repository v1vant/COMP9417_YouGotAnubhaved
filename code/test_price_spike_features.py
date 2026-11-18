"""Screen fixed spike groups on 2014; check the best on 2015–2020 only."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from run_experiments import load_data,market_days,training_table,fit_model,predict_task,simulate_policy
from price_spike_features import spike_features

ROOT=Path(__file__).resolve().parents[1]


def metrics(y,p):
    return {"accuracy":float(np.mean(y==p)*100),
            "balanced_accuracy":float(balanced_accuracy_score(y,p)*100)}


def main():
    output=ROOT/'results/price_spike_candidates_2014_2020'
    output.mkdir(parents=True,exist_ok=False)
    frame=load_data(ROOT/'data/nem_weather_halfhourly_2010_2026.csv').loc[:'2021-01-01 00:00:00']
    x,y=training_table(frame)
    extra,groups=spike_features(frame)
    extra=extra.loc[x.index]
    keep=extra.notna().all(axis=1)
    x,y,extra=x.loc[keep],y.loc[keep],extra.loc[keep]
    assert x.shape[1]==571 and extra.shape[1]==23
    days=market_days(x.index)
    candidates={'current_matched':[], 'trimmed':groups['trimmed'], 'relative':groups['relative'],
                'combined':groups['trimmed']+groups['relative'],
                'combined_absolute':sum(groups.values(),[])}
    config=json.loads((ROOT/'results/final_config.json').read_text())['price']['config']
    (output/'frozen_config.json').write_text(json.dumps({'config':config,'groups':groups,
        'candidates':candidates,'training_start':str(x.index.min()),'data_end':str(days.max()),
        'note':'All candidates use identical rows after the 30-day warmup. Fixed hyperparameters and 0.57 threshold. No post-2020 data.'},indent=2))
    train,valid=days.year<=2013,days.year==2014
    screen=[]
    for name,columns in candidates.items():
        table=x.join(extra[columns])
        print(f'2014 screening: {name}, {len(table.columns)} features',flush=True)
        model=fit_model('price',config,table.loc[train],y.loc[train])
        prediction=predict_task(model,table.loc[valid])
        row={'candidate':name,'features':len(table.columns),**metrics(y.loc[valid,'price'].to_numpy(),prediction)}
        screen.append(row)
        pd.DataFrame(screen).to_csv(output/'screen_2014.csv',index=False)
        np.save(output/f'{name}_2014.npy',prediction)
        joblib.dump(model,output/f'{name}_2013_model.joblib')
        print(row,flush=True)
    winner=max((r for r in screen if r['candidate']!='current_matched'),
               key=lambda r:(r['balanced_accuracy'],-r['features']))['candidate']
    (output/'selected_candidate.json').write_text(json.dumps({'candidate':winner,
        'selection':'Highest 2014 balanced accuracy among spike candidates; same settings retained for historical check.'},indent=2))
    print(f'Starting 2015–2020: matched reference versus {winner}',flush=True)
    initial=days.year<=2014
    evaluate=days.year>=2015
    simulation_days=days[evaluate].unique()
    assert simulation_days[-1]==pd.Timestamp('2020-12-31')
    truth=y.loc[evaluate,'price'].to_numpy().reshape(-1,48)
    np.savez(output/'truth.npz',price=truth,days=simulation_days.to_numpy())
    totals,periods=[] , []
    for name in ('current_matched',winner):
        table=x.join(extra[candidates[name]])
        model=fit_model('price',config,table.loc[initial],y.loc[initial])
        prediction,updates,_=simulate_policy('price','periodic_expanding',model,config,table,y,
            simulation_days,pd.Series(0.,index=simulation_days),np.zeros(365),np.zeros(365))
        assert len(updates)==3
        np.save(output/f'{name}_2015_2020.npy',prediction)
        pd.DataFrame(updates).to_csv(output/f'{name}_retrains.csv',index=False)
        row={'candidate':name,**metrics(truth.ravel(),prediction.ravel())}
        totals.append(row)
        pd.DataFrame(totals).to_csv(output/'historical_2015_2020.csv',index=False)
        for year in range(2015,2021):
            mask=simulation_days.year==year
            periods.append({'candidate':name,'year':year,**metrics(truth[mask].ravel(),prediction[mask].ravel())})
        pd.DataFrame(periods).to_csv(output/'yearly_metrics.csv',index=False)
        print(row,flush=True)
    print(pd.DataFrame(totals).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
