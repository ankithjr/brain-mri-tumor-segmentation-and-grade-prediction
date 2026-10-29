#!/bin/bash

#SBATCH --job-name=glioma-test-crops

#SBATCH --partition=markov_cpu

#SBATCH --cpus-per-task=1

#SBATCH --mem=8G

#SBATCH --time=04:00:00

#SBATCH --output=/scratch/markov3/users/arj94/brain-glioma/logs/test-crops-%j.out



set -euo pipefail

module load Python/3.11.5-GCCcore-13.2.0

source "$HOME/projects/brain-glioma/.venv/bin/activate"

export OMP_NUM_THREADS=1

export OPENBLAS_NUM_THREADS=1

export PYTHONUNBUFFERED=1

python "$HOME/projects/brain-glioma/scripts/prepare_predicted_test_crops.py"

