#!/bin/bash
#SBATCH --job-name=glioma-clean-env
#SBATCH --partition=markov_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/markov3/users/arj94/brain-glioma/logs/clean-env-%j.out

set -euo pipefail
module load Python/3.11.5-GCCcore-13.2.0
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PIP_NO_CACHE_DIR=1

project_root=/scratch/markov3/users/arj94/brain-glioma
run_root="$project_root/reproducibility/v1-clean-$SLURM_JOB_ID"
export CLEAN_RUN_ROOT="$run_root"
export CLEAN_SOURCE_COMMIT=34378fbd77674310f75dc26ecd5c57cb5ebb14ab
mkdir -p "$project_root/reproducibility"
mkdir "$run_root"
mkdir "$run_root/tmp"
export TMPDIR="$run_root/tmp"
python -m venv "$run_root/venv"
source "$run_root/venv/bin/activate"

python - <<'PY'
import os
from pathlib import Path
import urllib.request

root = Path(os.environ["CLEAN_RUN_ROOT"])
commit = os.environ["CLEAN_SOURCE_COMMIT"]
base = f"https://raw.githubusercontent.com/ankithjr/brain-mri-tumor-segmentation-and-grade-prediction/{commit}/"
for name in (
    "requirements-nnunet-lock.txt",
    "scripts/predict_case.py",
    "scripts/train_classifier_oracle_crop.py",
    "scripts/prepare_predicted_val_crops.py",
    "tests/test_predict_case.py",
):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(base + name, path)
print(f"Independent source copy: {root}", flush=True)
PY

python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "$run_root/requirements-nnunet-lock.txt" --extra-index-url https://download.pytorch.org/whl/cu126
python -m pip check
python -m pip freeze > "$run_root/installed-requirements.txt"
python -m unittest discover -s "$run_root/tests" -v
python "$run_root/scripts/predict_case.py" --help

python - <<'PY'
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

root = Path(os.environ["CLEAN_RUN_ROOT"])
if Path(sys.prefix).resolve() != (root / "venv").resolve():
    raise RuntimeError("Wrong virtual environment")
packages = {}
for module_name, distribution in (
    ("torch", "torch"), ("numpy", "numpy"), ("scipy", "scipy"),
    ("pandas", "pandas"), ("nibabel", "nibabel"),
    ("nnunetv2", "nnunetv2"), ("sklearn", "scikit-learn"),
):
    module = importlib.import_module(module_name)
    location = Path(module.__file__).resolve()
    if not location.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Package outside clean environment: {location}")
    packages[distribution] = version(distribution)
assert packages["torch"] == "2.7.1+cu126"
assert packages["nnunetv2"] == "2.8.1"
report = {
    "status": "cpu_environment_checks_passed",
    "source_commit": os.environ["CLEAN_SOURCE_COMMIT"],
    "run_directory": str(root), "python": sys.version,
    "packages": packages,
    "checks": ["pip check", "three CPU tests", "CLI help", "isolated package imports"],
    "gpu_inference": "pending; CPU allocation only",
    "scope": "Fresh venv on Markov using existing system Python module, not a new machine",
}
(root / "environment-check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
print("CLEAN ENVIRONMENT CPU CHECKS PASSED", flush=True)
PY
