# Single-case inference

`scripts/predict_case.py` runs the frozen fold-0 final nnU-Net checkpoint followed by the hash-verified epoch-17 crop classifier. It requires no grade label or reference segmentation. The CLI wrapper is new: CPU preprocessing parity and input rejection checks pass locally; Markov job 58407 subsequently completed full checkpoint/GPU integration with exit code 0:0 on validation exam UCSF-PDGM-0005. It reproduced the earlier grade-3 prediction with probability differences displayed as 0.00000000. The reference grade is 4, so this remains a misclassified case. See [integration record](../results/single-case-inference-integration.json).

## Inputs and models

Supply four distinct `.nii.gz` files in T1, T1c, T2, FLAIR order through named arguments. Inputs must already match the UCSF-PDGM preprocessing: skull-stripped, co-registered MRI in the same voxel orientation and 1 mm isotropic grid. The command checks finite 3D arrays, matching shapes and affines, spacing, and nonconstant nonzero intensities. These checks cannot establish successful registration, correct modality identity, skull stripping, or appropriate anatomical orientation. It does not perform those operations or convert DICOM.

Header units must be millimeters or unknown; unknown units are interpreted as millimeters and reported. Confirm this assumption before use. Do not supply arbitrary raw clinical scans.

Use the existing verified environment and keep `train_classifier_oracle_crop.py` beside `predict_case.py` (the model class is imported without running training). The segmentation results root must contain:

```text
Dataset501_UCSFPDGM/nnUNetTrainer__nnUNetPlans__3d_fullres/
    dataset.json
    plans.json
    fold_0/checkpoint_final.pth
```

The classifier checkpoint must match SHA-256 `b56365465a3c883aa1fa22c0bf73ae0685ea30025b4867bfabfcc5d63466b806`. Checkpoints are excluded from Git history. A verified model archive is prepared with [setup instructions and checksums](model-package.md); Release asset upload is pending. The evaluated final segmentation checkpoint SHA-256 is `fcdce8cf16fa6bdd2ff97418c6cc112b38b0ad1ab6b7922227ef16ece606691d`. The CLI records this hash, but does not yet enforce its reference value; verify the archive before use.

## Run inside a GPU allocation

```bash
python scripts/predict_case.py \
  --t1 /path/to/T1.nii.gz \
  --t1c /path/to/T1c.nii.gz \
  --t2 /path/to/T2.nii.gz \
  --flair /path/to/FLAIR.nii.gz \
  --segmentation-results /path/to/nnUNet_results \
  --classifier-checkpoint /path/to/checkpoint_best.pt \
  --output /path/to/new-output-folder
```

Activate the environment in the same shell before running. On Markov, `slurm/predict_case_demo.sbatch` does this and runs an already evaluated validation exam, UCSF-PDGM-0005, with a new job-specific output directory. Submit with:

```bash
sbatch "$HOME/projects/brain-glioma/slurm/predict_case_demo.sbatch"
```

## Outputs and behavior

- `prediction.json`: completion status, predicted grade, uncalibrated class probabilities, crop bounds, empty-mask fallback flag, geometry, and input/checkpoint hashes.
- `segmentation/case.nii.gz`: automatic tumor mask; labels 0, 1, 2, 3 correspond to background and source labels 1, 2, 4. Numeric clinical label meanings are not asserted here.
- `segmentation.log`: nnU-Net subprocess output.
- `nnunet-input/`: symlinks to the supplied images, not copied data. Keep the original inputs accessible.

The segmentation subprocess releases GPU memory before classifier inference. Crop processing uses all nonzero predicted labels, a 16-voxel margin, full-volume nonzero-voxel Z-score normalization clipped to [-5,5], and linear resizing to 96 x 96 x 64. Empty predictions use the full image and are explicitly flagged. No inference hyperparameters are selected from test results.

Existing output directories are rejected. A failed run retains diagnostic outputs and a failed status where possible; choose a fresh folder when retrying. Use only reports with `status: complete`. Softmax probabilities are not calibrated clinical confidence estimates. This is a research demonstration, not a clinical diagnostic tool.

## Verification

```bash
python -m unittest discover -s tests -v
```

CPU tests compare the new crop function bit-for-bit against the executed archived preprocessing source on synthetic boundary-touching and empty-mask cases. They also check rejection of mismatched image geometry and invalid mask values. They do not validate checkpoint execution, model accuracy, or clinical generalization. The Markov demo comparison against the existing automatic validation prediction for UCSF-PDGM-0005 passed at the reported precision. The complete HPC log and voxelwise mask comparison have not been independently archived.
