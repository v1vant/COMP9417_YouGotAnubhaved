"""Validate built C2 package, staged parity and exact preservation of demand."""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/forecast_package'))
from model import Forecaster,load_data,market_days,WEATHER,forecast_features,predict_task,predict_numpy_mlp


def main():
    package=ROOT/'code/forecast_package'; staged=ROOT/'code/submission_package'
    spec=importlib.util.spec_from_file_location('staged_c2_model',staged/'model.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    started=time.monotonic(); current=Forecaster(str(package/'artifacts')); init=time.monotonic()-started
    old=Forecaster(str(ROOT/'results/previous_current_f92a38e'))
    other=module.Forecaster(str(staged/'artifacts'))
    bundle=current.models['price']
    assert len(bundle['members'])==5 and len(bundle['features'])==597 and bundle['threshold']==.5
    for seed,member in zip(range(9417,9422),bundle['members']):
        params=member.get_params()
        assert params['random_state']==seed and params['n_estimators']==1200 and params['num_leaves']==63
        assert params['learning_rate']==.02 and params['colsample_bytree']==.5 and params['min_child_samples']==300
    frame=load_data(ROOT/'data/nem_weather_halfhourly_2010_2026.csv'); days=market_days(frame.index)
    timings=[]
    for day in days.unique()[-7:]:
        history=frame.loc[days<day]; weather=frame.loc[days==day,WEATHER]
        started=time.monotonic(); result=current.predict(history,weather); timings.append(time.monotonic()-started)
        previous=old.predict(history,weather); mirrored=other.predict(history,weather)
        assert result.shape==(48,2) and result.index.equals(weather.index)
        assert np.isfinite(result.to_numpy()).all() and result.nsw_price_up.isin([0,1]).all()
        assert result.nsw_demand_mw.between(0,20000).all()
        np.testing.assert_array_equal(result.nsw_demand_mw,previous.nsw_demand_mw)
        pd.testing.assert_frame_equal(result,mirrored)
        features=forecast_features(history,weather,weather_shape=True).loc[:,bundle['features']]
        probability=np.mean([member.predict_proba(features)[:,1] for member in bundle['members']],axis=0)
        np.testing.assert_array_equal(result.nsw_price_up,(probability>.5).astype(int))
    files=['model.py','train.py','mlp_train.py','demand_feature_columns.json','c2_price.py','c2_recipe.py','c2_candidate.json']
    for name in files: assert (package/name).read_bytes()==(staged/name).read_bytes()
    for path in (package/'artifacts').iterdir():
        if path.is_file(): assert path.read_bytes()==(staged/'artifacts'/path.name).read_bytes()
    size=sum(p.stat().st_size for p in (package/'artifacts').rglob('*') if p.is_file())
    assert size<100_000_000 and max(timings)<60 and 'torch' not in sys.modules
    result=dict(status='passed',days=7,init_seconds=init,max_predict_seconds=max(timings),
        demand_predictions_exactly_unchanged=True,price_mean_probability_verified=True,
        staged_matches=True,torch_imported=False,artifact_bytes=size,
        note='Output validity and parity checks, not accuracy: these dates were used in final fitting.')
    (ROOT/'results/current_c2_build/package_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
