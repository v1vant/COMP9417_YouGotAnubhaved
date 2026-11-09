"""Run the forecasting tests, experiments, and model training."""
import argparse
import importlib.metadata
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def check_runtime():
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Use a Python 3.11 environment with the kit's pinned requirements")
    required = {"numpy", "pandas", "scipy", "scikit-learn", "lightgbm",
                "joblib", "holidays", "pyarrow"}
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as handle:
        for line in handle:
            if "==" not in line or line.startswith("#"):
                continue
            package, expected = line.strip().split("==")
            if package in required:
                actual = importlib.metadata.version(package)
                if actual != expected:
                    raise RuntimeError(f"{package}: expected {expected}, installed {actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    check_runtime()
    environment = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", PYTHONHASHSEED="9417")
    commands = [
        ["-m", "unittest", "discover", "-s", "tests", "-v"],
        ["run_experiments.py"] + (["--resume"] if args.resume else []),
        [os.path.join("forecast_package", "train.py"),
         os.path.join(ROOT, "data", "nem_weather_halfhourly_2010_2026.csv"),
         "--selection", os.path.join(ROOT, "results", "final_config.json")],
    ]
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "execution.log"), "w", encoding="utf-8") as log:
        for arguments in commands:
            command = [sys.executable] + arguments
            print("RUN:", " ".join(command), flush=True)
            process = subprocess.Popen(command, cwd=HERE, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait() != 0:
                raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
