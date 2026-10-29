#!/bin/bash

#SBATCH --job-name=glioma-grade-eval

#SBATCH --partition=markov_gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4

#SBATCH --mem=16G

#SBATCH --time=00:30:00

#SBATCH --output=/scratch/markov3/users/arj94/brain-glioma/logs/grade-auto-eval-%j.out



set -euo pipefail

module load Python/3.11.5-GCCcore-13.2.0

source "$HOME/projects/brain-glioma/.venv/bin/activate"

export OMP_NUM_THREADS=1

export MKL_NUM_THREADS=1

export OPENBLAS_NUM_THREADS=1

export PYTHONUNBUFFERED=1

python "$HOME/projects/brain-glioma/scripts/evaluate_predicted_crops.py"

