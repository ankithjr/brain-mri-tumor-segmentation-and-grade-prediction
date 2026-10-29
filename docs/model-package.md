# Frozen model package v1

The uploaded archive was independently checked against its recorded SHA-256 and internal manifest. All four model/configuration file hashes matched. No MRI files are included.

- Asset: `brain-mri-models-v1.tar.gz`
- Size: 233529574 bytes (about 222.71 MiB)
- SHA-256: `b20aae39903ce75767cdaec89dae41be9b68b54f79ebf79e8049ea782e56f0a5`
- File manifest: [model-package-checksums.json](../configs/model-package-checksums.json)
- Publication status: verified package prepared; GitHub Release asset upload pending.

## Environment

Use Linux, Python 3.11, and an NVIDIA GPU compatible with the recorded PyTorch CUDA 12.6 build. The verified Markov environment used Python 3.11.5, PyTorch 2.7.1+cu126, nnU-Net 2.8.1, and an RTX 2080 Ti or H100 MIG slice. The tested inference demo ran on Markov. A fresh installation on another machine has not yet been tested.

From a checkout of this repository:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-nnunet-lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
python -m pip check
python -c "import torch, nibabel, nnunetv2; print('CUDA:', torch.cuda.is_available())"
```

The package snapshot preserves the observed training environment; it is not a dependency hash lock or a guarantee of compatibility on other platforms. On Markov, load `Python/3.11.5-GCCcore-13.2.0` before activating the existing environment.

## Verify and unpack

Download the asset from the repository's Releases page once published, then verify it before extracting:

```bash
echo "b20aae39903ce75767cdaec89dae41be9b68b54f79ebf79e8049ea782e56f0a5  brain-mri-models-v1.tar.gz" | sha256sum --check
mkdir brain-mri-models-v1
tar -xzf brain-mri-models-v1.tar.gz -C brain-mri-models-v1
```

The archive contains:

```text
model-package-checksums.json
nnUNet_results/Dataset501_UCSFPDGM/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json
nnUNet_results/Dataset501_UCSFPDGM/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json
nnUNet_results/Dataset501_UCSFPDGM/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth
checkpoints/grade_classifier_oracle_crop/checkpoint_best.pt
```

The segmentation checkpoint hash is `fcdce8cf16fa6bdd2ff97418c6cc112b38b0ad1ab6b7922227ef16ece606691d`. The classifier hash is `b56365465a3c883aa1fa22c0bf73ae0685ea30025b4867bfabfcc5d63466b806`. Archive verification establishes byte integrity; it is not a fresh model accuracy evaluation.

## Predict one exam

Activate the environment and run inside an allocated GPU session. Keep `scripts/train_classifier_oracle_crop.py` alongside the inference script because it supplies the trained architecture.

```bash
python scripts/predict_case.py \
  --t1 /absolute/path/to/T1.nii.gz \
  --t1c /absolute/path/to/T1c.nii.gz \
  --t2 /absolute/path/to/T2.nii.gz \
  --flair /absolute/path/to/FLAIR.nii.gz \
  --segmentation-results "$PWD/brain-mri-models-v1/nnUNet_results" \
  --classifier-checkpoint "$PWD/brain-mri-models-v1/checkpoints/grade_classifier_oracle_crop/checkpoint_best.pt" \
  --output "$PWD/new-case-output"
```

Inputs must already match UCSF-PDGM preprocessing: four skull-stripped, co-registered sequences in the same voxel orientation and 1 mm isotropic grid. The command does not perform registration, skull stripping, orientation harmonization, or DICOM conversion. See [input checks and output details](single-case-inference.md).

Expect `prediction.json` with `status: complete` and `segmentation/case.nii.gz`. Grading probabilities are uncalibrated. Use a new output directory for each run.

## Evaluation and limits

On 75 held-out patients, segmentation mean foreground Dice was 0.8438; automatic-crop grading macro-F1 was 0.6086 and balanced accuracy 0.6407. The test cohort contains only 9 grade-2 and 6 grade-3 patients; see [uncertainty analysis](grade-test-analysis.md). Validation examples include both a correct grade-4 prediction and a grade-4-to-3 error. The single-case CLI reproduced a prior validation prediction at displayed probability precision; a clean-machine/package extraction inference run remains to be checked.

The segmentation model was trained on 346 development patients, with 74 validation patients; nnU-Net planning used the pooled development cohort. The grader was trained with reference-mask crops and evaluated with automatic crops. Results are single-institution and do not establish external clinical performance. Research and educational use only.

Models were trained using UCSF-PDGM Version 5; dataset attribution and CC BY 4.0 information are in the repository README. The dataset license does not by itself specify a license for all repository code or model weights; no separate code/weight license is asserted by these instructions.
