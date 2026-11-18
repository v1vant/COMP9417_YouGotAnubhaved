"""2014 screen: current features/settings with fixed recency half-lives."""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from run_experiments import load_data,market_days,training_table

ROOT=Path(__file__).resolve().parents[1]


def metrics(y,p):
    return {"accuracy":float(np.mean(y==p)*100),
            "balanced_accuracy":float(balanced_accuracy_score(y,p)*100),
            "recall_0":float(np.mean(p[y==0]==0)*100),
            "recall_1":float(np.mean(p[y==1]==1)*100)}


def fit_recency_classifier(features, labels, config, half_life):
    days=market_days(features.index)
    target=labels.to_numpy()
    age=(days.max()-days).days.to_numpy()/365.25
    weights=np.power(.5,age/half_life)
    for label in (0,1):
        mask=target==label
        weights[mask]*=len(weights)/(2*weights[mask].sum())
    np.testing.assert_allclose(weights[target==0].sum(),weights[target==1].sum())
    model=LGBMClassifier(objective='binary',n_estimators=config['trees'],
        learning_rate=config['learning_rate'],num_leaves=config['leaves'],
        min_child_samples=160,colsample_bytree=.85,subsample=.85,subsample_freq=1,
        reg_lambda=8.,max_bin=127,n_jobs=2,verbosity=-1,deterministic=True,
        force_col_wise=True,random_state=9417,class_weight=None)
    model.fit(features,target,sample_weight=weights)
    return model


def main():
    started=time.monotonic()
    output=ROOT/'results/price_recency_2014'
    output.mkdir(parents=True,exist_ok=False)
    frame=load_data(ROOT/'data/nem_weather_halfhourly_2010_2026.csv').loc[:'2015-01-01 00:00:00']
    x,y=training_table(frame)
    assert x.shape[1]==571
    days=market_days(x.index)
    train,valid=days.year<=2013,days.year==2014
    config=json.loads((ROOT/'results/final_config.json').read_text())['price']['config']
    cached=pd.read_csv(ROOT.parent/'COMP9417_price_regression/results/price_regression_2014/predictions.csv',
                       index_col='timestamp',parse_dates=True)
    assert cached.index.equals(x.index[valid])
    target=y.loc[valid,'price'].to_numpy()
    np.testing.assert_array_equal(target,cached.actual_up)
    results=[dict(model='current_no_decay',half_life_years=None,
                  **metrics(target,(cached.classifier_probability.to_numpy()>config['threshold']).astype(int)))]
    probabilities=pd.DataFrame({'actual_up':target,'current_probability':cached.classifier_probability},index=cached.index)
    for half_life in (2.,4.):
        print(f'Fitting recency half-life {half_life:g} years',flush=True)
        model=fit_recency_classifier(x.loc[train],y.loc[train,'price'],config,half_life)
        probability=model.predict_proba(x.loc[valid])[:,1]
        results.append(dict(model=f'half_life_{half_life:g}years',half_life_years=half_life,
                            **metrics(target,(probability>config['threshold']).astype(int))))
        probabilities[f'half_life_{half_life:g}years_probability']=probability
        pd.DataFrame(results).to_csv(output/'metrics.csv',index=False)
        joblib.dump({'estimator':model,'features':list(x),'config':config,'half_life_years':half_life,
                     'training_end':'2013-12-31'},output/f'model_half_life_{half_life:g}.joblib')
        print(results[-1],flush=True)
    probabilities.to_csv(output/'predictions.csv',index_label='timestamp')
    (output/'metadata.json').write_text(json.dumps({'training_years':[2010,2011,2012,2013],
        'evaluation_year':2014,'features':571,'config':config,'half_lives':[2,4],
        'effective_class_weights':'Normalised within class to equal total weight',
        'spike_features_enabled':False,'holdout_evaluated':False,
        'seconds':time.monotonic()-started},indent=2))
    print(pd.DataFrame(results).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
