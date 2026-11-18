"""Standalone candidate rebuild. Demand uses the pinned reference training recipe."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from pipeline import table, calibration
from research import fit, threshold, load_data, market_days


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('csv',type=Path)
    ap.add_argument('--selection',type=Path,default=Path(__file__).parent/'candidate.json');a=ap.parse_args()
    policy=json.loads(a.selection.read_text())
    subprocess.run([sys.executable,str(Path(__file__).parent/'reference_train.py'),str(a.csv.resolve())],check=True)
    frame=load_data(a.csv)
    def build(component):
        c=component['config']
        x,y=table(frame,compact=c.get('compact',False),context=c.get('price_feature_set')=='both')
        if component.get('stacked'):
            from stack import generate
            with tempfile.TemporaryDirectory(prefix='price_demand_oof_') as cache:
                extra=generate(frame,x,y,Path(cache))
            x=pd.concat([x.loc[extra.index],extra],axis=1);y=y.loc[x.index]
        end=market_days(x.index).max()+pd.Timedelta(days=1)
        scores=calibration(c,x,y,end) if component['refresh'] else None
        cutoff=threshold(y.loc[scores.index,'price'],scores.to_numpy(),c.get('threshold',.57)) if scores is not None else c.get('threshold',.57)
        bundle=fit(c,x,y,end);bundle['threshold']=cutoff;bundle['stacked']=component.get('stacked',False)
        return bundle,scores,y
    if 'components' in policy:
        base=dict(family='boosted',leaves=31,trees=600,learning_rate=.04,threshold=.57)
        a,pa,y=build(dict(config=dict(base,price_feature_set='base'),refresh=True))
        b,pb,_=build(dict(config=dict(base,price_feature_set='both',compact=True),refresh=True))
        w=policy['weight'];cutoff=threshold(y.loc[pa.index,'price'],(w*pa+(1-w)*pb).to_numpy())
        bundle=dict(a,components=[a,b],weight=w,threshold=cutoff,config=dict(base,price_feature_set='both'))
    else:
        bundle,_,_=build(policy);cutoff=bundle['threshold']
    path=Path(__file__).parent/'artifacts';models=joblib.load(path/'models.joblib')
    models['price']=bundle;joblib.dump(models,path/'models.joblib',compress=3)
    manifest=json.loads((path/'manifest.json').read_text());manifest['price_v2']=dict(policy=policy,threshold=cutoff,training_end=bundle['training_end'])
    (path/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
