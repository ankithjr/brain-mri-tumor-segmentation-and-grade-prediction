#!/bin/bash

#SBATCH --job-name=glioma-preprocess

#SBATCH --partition=markov_cpu

#SBATCH --cpus-per-task=4

#SBATCH --mem=32G

#SBATCH --time=06:00:00

#SBATCH --output=/scratch/markov3/users/arj94/brain-glioma/logs/preprocess-%j.out



set -euo pipefail



module load Python/3.11.5-GCCcore-13.2.0

source "$HOME/projects/brain-glioma/.venv/bin/activate"



export nnUNet_raw="/scratch/markov3/users/arj94/brain-glioma/nnUNet_raw"

export nnUNet_preprocessed="/scratch/markov3/users/arj94/brain-glioma/nnUNet_preprocessed"

export nnUNet_results="/scratch/markov3/users/arj94/brain-glioma/nnUNet_results"

export OMP_NUM_THREADS=1

export MKL_NUM_THREADS=1

export OPENBLAS_NUM_THREADS=1



nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity -c 3d_fullres -npfp 2 -np 2 -gpu_memory_target 10



python - <<'PY'

import json

import os

from pathlib import Path



raw = Path(os.environ["nnUNet_raw"]) / "Dataset501_UCSFPDGM"

processed = Path(os.environ["nnUNet_preprocessed"]) / "Dataset501_UCSFPDGM"

split = json.loads((raw / "fixed_split.json").read_text())

assert len(split[0]["train"]) == 346

assert len(split[0]["val"]) == 74

assert not set(split[0]["train"]) & set(split[0]["val"])

target = processed / "splits_final.json"

if target.exists():

    assert json.loads(target.read_text()) == split, "Existing split differs"

else:

    target.write_text(json.dumps(split, indent=2) + "\n")

print("PREPROCESSING COMPLETE; fixed 346/74 split installed.", flush=True)

PY

