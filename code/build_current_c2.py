"""Build adopted C2 price artifacts while preserving validated demand exactly."""
import hashlib
import json
import shutil
import sys
import time
import tempfile
from pathlib import Path
import joblib
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/forecast_package'))
from model import load_data
from c2_price import price_table
from c2_recipe import fit_price


def main():
    started=time.monotonic()
    package=ROOT/'code/forecast_package'; artifacts=package/'artifacts'
    backup=ROOT/'results/previous_current_f92a38e'
    if not backup.exists(): shutil.copytree(artifacts,backup)
    out=ROOT/'results/current_c2_build'; out.mkdir(exist_ok=True)
    config=json.loads((package/'c2_candidate.json').read_text())
    x,y=price_table(load_data(ROOT/'data/nem_weather_halfhourly_2010_2026.csv'))
    members=[]; audit=[]
    previous_timings={row['seed']:row['seconds'] for row in json.loads((out/'member_timings.json').read_text())} if (out/'member_timings.json').exists() else {}
    for seed in config['seeds']:
        path=out/f'price_seed_{seed}.joblib'; start=time.monotonic()
        if path.exists():
            member=joblib.load(path)
            assert member.feature_name_==list(x)
        else:
            print(f'Fitting full-data C2 seed {seed}, {len(x)} rows',flush=True)
            member=fit_price(x,y.price,seed,config['params'])
            joblib.dump(member,path,compress=3)
        members.append(member)
        audit.append(dict(seed=seed,seconds=previous_timings.get(seed,time.monotonic()-start)))
        (out/'member_timings.json').write_text(json.dumps(audit,indent=2))
        print(f'Saved C2 seed {seed}: {audit[-1]["seconds"]:.1f}s',flush=True)
    models=joblib.load(backup/'models.joblib')
    original_demand_hash=joblib.hash(models['demand'])
    models['price']=dict(task='price',config=config,features=list(x),estimator=members[0],members=members,
        threshold=.5,training_start='2010-01-30',training_end='2026-10-31')
    with tempfile.TemporaryDirectory(prefix='c2_artifact_',dir='/private/tmp') as directory:
        local=Path(directory)/'models.joblib'
        joblib.dump(models,local,compress=3)
        pending=artifacts/'models.joblib.pending'
        shutil.copyfile(local,pending)
        pending.replace(artifacts/'models.joblib')
    assert joblib.hash(joblib.load(artifacts/'models.joblib')['demand'])==original_demand_hash
    assert (artifacts/'demand_mlp.npz').read_bytes()==(backup/'demand_mlp.npz').read_bytes()
    manifest=json.loads((backup/'manifest.json').read_text())
    selection=json.loads((ROOT/'results/final_config.json').read_text())
    manifest.update(selection=selection,task_feature_counts={'demand':405,'price':597},
        price_c2={'seeds':config['seeds'],'threshold':.5,'training_rows':len(x),'training_end':'2026-10-31'},
        previous_current='f92a38e',price_build_seconds=time.monotonic()-started)
    (artifacts/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (artifacts/'blend_training_selection.json').write_text(json.dumps(selection,indent=2))
    result=dict(status='built',seconds=time.monotonic()-started,member_training_seconds=sum(row['seconds'] for row in audit),demand_artifacts_unchanged=True,
        artifact_bytes=sum(p.stat().st_size for p in artifacts.iterdir() if p.is_file()),
        price_members=5,price_features=597,threshold=.5)
    (out/'completion.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
