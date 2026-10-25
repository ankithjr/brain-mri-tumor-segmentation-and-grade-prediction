#!/bin/bash
#SBATCH --job-name=glioma-baseline
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/markov3/users/arj94/brain-glioma/logs/train-%j.out

set -euo pipefail
module load Python/3.11.5-GCCcore-13.2.0
source "$HOME/projects/brain-glioma/.venv/bin/activate"
export nnUNet_raw="/scratch/markov3/users/arj94/brain-glioma/nnUNet_raw"
export nnUNet_preprocessed="/scratch/markov3/users/arj94/brain-glioma/nnUNet_preprocessed"
export nnUNet_results="/scratch/markov3/users/arj94/brain-glioma/nnUNet_results"
export nnUNet_compile=false
export nnUNet_n_proc_DA=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
nvidia-smi
nnUNetv2_train 501 3d_fullres 0 --npz
