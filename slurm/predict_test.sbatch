#!/bin/bash

#SBATCH --job-name=glioma-test

#SBATCH --partition=markov_gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4

#SBATCH --mem=32G

#SBATCH --time=02:00:00

#SBATCH --output=/scratch/markov3/users/arj94/brain-glioma/logs/test-%j.out



set -euo pipefail

module load Python/3.11.5-GCCcore-13.2.0

source "$HOME/projects/brain-glioma/.venv/bin/activate"



export nnUNet_raw="/scratch/markov3/users/arj94/brain-glioma/nnUNet_raw"

export nnUNet_preprocessed="/scratch/markov3/users/arj94/brain-glioma/nnUNet_preprocessed"

export nnUNet_results="/scratch/markov3/users/arj94/brain-glioma/nnUNet_results"

export nnUNet_compile=false

export OMP_NUM_THREADS=1

export MKL_NUM_THREADS=1

export OPENBLAS_NUM_THREADS=1

export PYTHONUNBUFFERED=1



nvidia-smi

nnUNetv2_predict -d 501 -c 3d_fullres -f 0 -chk checkpoint_final.pth -i /scratch/markov3/users/arj94/brain-glioma/results/test-evaluation/images -o /scratch/markov3/users/arj94/brain-glioma/results/test-evaluation/predictions -npp 2 -nps 2

