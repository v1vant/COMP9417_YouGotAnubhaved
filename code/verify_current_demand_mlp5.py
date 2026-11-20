"""Verify five-MLP package outputs, unchanged C2, and staged package parity."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/forecast_package'))
from model import Forecaster,load_data,market_days,WEATHER,forecast_features,predict_task,predict_numpy_mlp


def main():
    package=ROOT/'code/forecast_package';staged=ROOT/'code/submission_package'
    backup=ROOT/'results/previous_current_30d9c60'
    assert (package/'artifacts/models.joblib').read_bytes()==(backup/'models.joblib').read_bytes()
    started=time.monotonic();current=Forecaster(str(package/'artifacts'));init=time.monotonic()-started
    previous=Forecaster(str(backup))
    spec=importlib.util.spec_from_file_location('staged_mlp5',staged/'model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    mirrored=module.Forecaster(str(staged/'artifacts'))
    assert len(current.demand_mlp_members)==5 and len(previous.demand_mlp_members)==1
    assert len(current.models['price']['members'])==5 and current.models['price']['threshold']==.5
    # Rebuilding the original seed must preserve its fitted weights exactly.
    with np.load(backup/'demand_mlp.npz') as old:
        for key in old.files:np.testing.assert_array_equal(current.demand_mlp[key],old[key])
    frame=load_data(ROOT/'data/nem_weather_halfhourly_2010_2026.csv');days=market_days(frame.index)
    timings=[]
    for day in days.unique()[-7:]:
        history=frame.loc[days<day];weather=frame.loc[days==day,WEATHER]
        started=time.monotonic();result=current.predict(history,weather);timings.append(time.monotonic()-started)
        assert result.shape==(48,2) and result.index.equals(weather.index)
        assert np.isfinite(result.to_numpy()).all() and result.nsw_price_up.isin([0,1]).all()
        assert result.nsw_demand_mw.between(0,20000).all()
        old=previous.predict(history,weather)
        np.testing.assert_array_equal(result.nsw_price_up,old.nsw_price_up)
        pd.testing.assert_frame_equal(result,mirrored.predict(history,weather))
        features=forecast_features(history,weather)
        lgbm=predict_task(current.models['demand'],features)
        mlp_mean=np.mean([predict_numpy_mlp(member,features) for member in current.demand_mlp_members],axis=0,dtype=np.float64)
        np.testing.assert_array_equal(result.nsw_demand_mw,.5*lgbm+.5*mlp_mean)
    for path in package.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            other=staged/path.relative_to(package)
            assert other.exists() and path.read_bytes()==other.read_bytes(),str(path)
    size=sum(p.stat().st_size for p in (package/'artifacts').iterdir() if p.is_file())
    assert size<100_000_000 and max(timings)<60 and 'torch' not in sys.modules
    result=dict(status='passed',days=7,init_seconds=init,max_predict_seconds=max(timings),artifact_bytes=size,
        MLP_members=5,price_C2_and_demand_LightGBM_byte_identical=True,original_seed_9417_arrays_identical=True,
        price_outputs_unchanged=True,demand_blend_verified=True,staged_matches=True,torch_imported=False,
        note='Output validity and parity, not an accuracy evaluation: final-fit dates were used.')
    (ROOT/'results/current_demand_mlp5_build/package_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
