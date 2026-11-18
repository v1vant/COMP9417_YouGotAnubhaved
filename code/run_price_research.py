"""Bounded research queue; never promotes or changes the incumbent package."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import pandas as pd

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features-results',type=Path,required=True,help='Completed or currently running full feature experiment')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hours',type=float,default=24)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()): p.error('Use an empty output directory')
    if args.hours<=0: p.error('--hours must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+args.hours*3600
    status=dict(state='waiting_for_features',features=str(args.features_results),stages=[])
    def save(): (args.output/'queue.json').write_text(json.dumps(status,indent=2))
    save()
    while not (args.features_results/'manifest.json').exists():
        if time.monotonic()>=deadline:
            status['state']='budget_exhausted_waiting_for_features'; save();return
        time.sleep(10)
    manifest=json.loads((args.features_results/'manifest.json').read_text())
    if manifest['quick']: raise ValueError('Queue requires a full feature experiment')
    validation=pd.read_csv(args.features_results/'validation.csv')
    feature=str(validation.sort_values('balanced_accuracy',ascending=False).iloc[0].candidate)
    if feature not in ['base','geometry','recency','both']: raise ValueError('Expected a feature ablation')
    status['stage1_feature_set']=feature
    # These jobs retain the Stage-1 feature decision; no later holdout chooses it.
    for phase in ['policies','tuning','complementary']:
        remaining=(deadline-time.monotonic())/3600
        if remaining<=0: break
        status['state']='running_'+phase; save()
        dest=args.output/phase
        with (args.output/f'{phase}.log').open('w') as log:
            command=[sys.executable,str(HERE/'run_price_experiments.py'),'--phase',phase,
                     '--feature-set',feature,'--output',str(dest),'--hours',str(min(remaining,4.))]
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        status['stages'].append(dict(phase=phase,returncode=result.returncode))
        save()
        if result.returncode:
            status['state']='failed_'+phase;save();return
        subprocess.run([sys.executable,str(HERE/'analyse_price_experiments.py'),str(dest)],check=True)
    status['state']='completed' if len(status['stages'])==3 else 'budget_exhausted'
    save()

if __name__=='__main__': main()
