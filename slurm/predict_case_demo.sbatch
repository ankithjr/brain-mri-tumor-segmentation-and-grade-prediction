#!/bin/bash
#SBATCH --job-name=glioma-demo
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/markov3/users/arj94/brain-glioma/logs/demo-%j.out

set -euo pipefail
module load Python/3.11.5-GCCcore-13.2.0
source "$HOME/projects/brain-glioma/.venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export nnUNet_raw="/scratch/markov3/users/arj94/brain-glioma/nnUNet_raw"
export nnUNet_preprocessed="/scratch/markov3/users/arj94/brain-glioma/nnUNet_preprocessed"
export nnUNet_results="/scratch/markov3/users/arj94/brain-glioma/nnUNet_results"

project_root=/scratch/markov3/users/arj94/brain-glioma
exam_root="$project_root/data/UCSF-PDGM-core/UCSF-PDGM-0005_nifti"
python "$HOME/projects/brain-glioma/scripts/predict_case.py" \
    --t1 "$exam_root/UCSF-PDGM-0005_T1.nii.gz" \
    --t1c "$exam_root/UCSF-PDGM-0005_T1c.nii.gz" \
    --t2 "$exam_root/UCSF-PDGM-0005_T2.nii.gz" \
    --flair "$exam_root/UCSF-PDGM-0005_FLAIR.nii.gz" \
    --segmentation-results "$nnUNet_results" \
    --classifier-checkpoint "$project_root/checkpoints/grade_classifier_oracle_crop/checkpoint_best.pt" \
    --output "$project_root/results/inference-demo-$SLURM_JOB_ID"
