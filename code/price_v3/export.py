"""WP4: assemble a self-contained submission package for ONE frozen price candidate.

Never overwrites code/submission_package. Copies the reference demand recipe
unchanged (reference_model.py -> base_model.py, reference_train.py, mlp_train.py,
demand_feature_columns.json) plus the v3 package model/train and candidate.json.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from common import ROOT, V2

HERE = Path(__file__).resolve().parent


def assemble(candidate, output):
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Use a new empty output directory')
    allowed = {'name', 'seeds', 'half_life_years', 'threshold', 'source', 'weather_shape', 'params', 'near_threshold_scale'}
    if set(candidate) - allowed or not candidate.get('seeds'):
        raise SystemExit(f'Unsupported candidate fields: {candidate}')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(V2/'reference_model.py', output/'base_model.py')
    for name in ['reference_train.py', 'mlp_train.py', 'demand_feature_columns.json']:
        shutil.copyfile(V2/name, output/name)
    for name in ['model.py', 'train.py']:
        shutil.copyfile(HERE/'package'/name, output/name)
    shutil.copyfile(HERE/'recipe.py', output/'recipe.py')
    (output/'candidate.json').write_text(json.dumps(candidate, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate', type=Path, required=True,
                    help='JSON: seeds, threshold, optional half_life_years/weather_shape/params/near_threshold_scale')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--data', type=Path, help='Also run the bare rebuild with this CSV')
    args = ap.parse_args()
    assemble(json.loads(args.candidate.read_text()), args.output)
    if args.data:
        subprocess.run([sys.executable, str(args.output/'train.py'), str(args.data.resolve())], check=True)
    print('Package assembled:', args.output)


if __name__ == '__main__':
    main()
