"""Export the adopted five-MLP demand ensemble without refitting either LightGBM."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started=time.monotonic();package=ROOT/'code/forecast_package';artifacts=package/'artifacts'
    out=ROOT/'results/current_demand_mlp5_build';out.mkdir(exist_ok=True)
    backup=ROOT/'results/previous_current_30d9c60'
    if not backup.exists():
        shutil.copytree(artifacts,backup)
        for source,name in [(ROOT/'README.md','README.md'),(ROOT/'results/current_model.json','current_model.json')]:shutil.copy2(source,backup/name)
    original_models_hash=digest(backup/'models.joblib')
    with tempfile.TemporaryDirectory(prefix='full_demand_mlp5_',dir='/private/tmp') as directory:
        output=Path(directory)
        subprocess.run([sys.executable,'-u',str(package/'mlp_train.py'),
            str(ROOT/'data/nem_weather_halfhourly_2010_2026.csv'),'--output',str(output),
            '--selection',str(ROOT/'results/final_config.json')],check=True)
        for path in output.iterdir():
            pending=artifacts/(path.name+'.pending');shutil.copyfile(path,pending);pending.replace(artifacts/path.name)
    assert digest(artifacts/'models.joblib')==original_models_hash
    metadata=json.loads((artifacts/'demand_mlp_manifest.json').read_text())
    assert metadata['seeds']==list(range(9417,9422))
    manifest=json.loads((backup/'manifest.json').read_text())
    selection=json.loads((ROOT/'results/final_config.json').read_text())
    manifest.update(selection=selection,demand_mlp_members=[member['file'] for member in metadata['members']],
        demand_mlp_seeds=metadata['seeds'],mlp_training=metadata,demand_blend={'LightGBM':.5,'MLP':.5},
        previous_current='30d9c60',demand_mlp5_build_seconds=time.monotonic()-started)
    (artifacts/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (artifacts/'blend_training_selection.json').write_text(json.dumps(selection,indent=2))
    result=dict(status='built',seconds=time.monotonic()-started,seeds=metadata['seeds'],
        price_C2_and_demand_LightGBM_artifacts_byte_identical=True,
        artifact_bytes=sum(p.stat().st_size for p in artifacts.iterdir() if p.is_file()),
        numpy_torch_max_difference_MW=metadata['numpy_torch_max_difference_MW'])
    assert result['artifact_bytes']<100_000_000
    (out/'completion.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
