Frozen research baseline for brain MRI tumor segmentation and WHO grade prediction.

Asset: brain-mri-models-v1.tar.gz (233529574 bytes)
SHA256: b20aae39903ce75767cdaec89dae41be9b68b54f79ebf79e8049ea782e56f0a5

Includes the fold-0 final nnU-Net checkpoint, epoch-17 classifier, plans, dataset configuration, and checksum manifest. No MRI data are included.

Held-out evaluation (75 patients): segmentation mean foreground Dice 0.8438; grading macro-F1 0.6086; balanced accuracy 0.6407. Probabilities are uncalibrated; single-institution research results, not a clinical diagnostic tool.

Setup: https://github.com/ankithjr/brain-mri-tumor-segmentation-and-grade-prediction/blob/main/docs/model-package.md
Requires separately obtained, preprocessed T1/T1c/T2/FLAIR MRI and the recorded Linux/Python/PyTorch/nnU-Net environment. Extracted-package inference subsequently passed in the existing Markov environment, reproducing the prior validation demo probabilities. Fresh-machine installation remains unverified. See results/model-package-inference-check.json.
