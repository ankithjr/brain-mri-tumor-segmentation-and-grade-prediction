# Brain MRI Tumor Segmentation and Grade Prediction

**Status: In development.** Environment setup and core MRI transfer are verified; automated image auditing passed; three-exam visual QC is complete and patient-disjoint splits are created. The nnU-Net segmentation baseline completed and full-volume validation is available; held-out test evaluation and grade prediction remain pending.

A personal medical imaging project developing a reproducible deep learning pipeline to segment diffuse gliomas and predict CNS WHO grade from preoperative MRI using UCSF-PDGM. Development and training use Case Western Reserve University's Markov HPC cluster.

## Project Focus

- Multisequence **brain MRI**
- Tumor **segmentation**
- CNS WHO **grade prediction**
- Patient-disjoint evaluation
- Reproducible PyTorch/HPC workflow
- Explicit handling of class imbalance and dataset leakage risk

This repository is intentionally transparent about project status: environment and data validation are in progress, and no model results are reported before they are actually produced.

## Planned pipeline

1. Validate multisequence MRI, tumor masks, and clinical labels.
2. Establish patient-disjoint training, validation, and test splits.
3. Train a tumor segmentation baseline.
4. Train a classifier to predict WHO grades 2, 3, and 4.
5. Evaluate segmentation with Dice and classification with macro-F1, balanced accuracy, per-class recall, and confusion matrices.
6. Publish reproducible experiments, error analysis, and selected visual results.

These are planned capabilities. Tumor grade is distinct from conventional cancer stage. This is a research and educational project, not a clinical diagnostic tool.

## Verified progress

- Python 3.11.5 environment created on Markov.
- PyTorch 2.7.1+cu126 installed; GPU matrix multiplication verified on an NVIDIA GeForce RTX 2080 Ti (11 GB).
- NumPy, SciPy, pandas, NiBabel, scikit-learn, Matplotlib, and tqdm installed.
- Package imports, NumPy-to-GPU interoperability, and dependency checks passed.
- Clinical metadata inspected: 501 exam records containing 56 grade-2, 43 grade-3, and 402 grade-4 labels before patient reconciliation or exclusions.
- Core MRI archive transferred to HPC with matching Windows/HPC SHA-256 checksums.
- Extracted 2,505 NIfTI files; all passed full gzip stream integrity checks (zero failures). See [validation record](results/data-transfer-validation.json) and [reusable validator](scripts/validate_gzip.py). The subsequent automated audit passed for all 501 exams: one file per required modality, matching 3D shapes and affines within each exam, finite voxels, and nonempty integer tumor masks. Observed mask labels are 0, 1, 2, and 4. Exam IDs match metadata and follow-up grouping identifies 495 patients. See [audit script](scripts/audit_dataset.py) and [aggregate audit](results/dataset-audit.json). A visual spot-check of one initial exam per grade found no obvious cross-sequence mask displacement; this is not a full annotation review. Label semantic verification remains pending.
- nnU-Net 2.8.1 installed with the original environment as constraints; imports and `pip check` passed, with PyTorch remaining 2.7.1+cu126. The expanded [nnU-Net package snapshot](requirements-nnunet-lock.txt) is archived. Dataset conversion completed: Dataset501_UCSFPDGM contains 1,680 image symlinks and 420 converted masks for development exams only; all 75 test exams are excluded. Source label 4 is remapped to 3 in new mask copies; originals are preserved. Planning and 3D full-resolution preprocessing completed for all 420 development exams (preprocessing stage: 7 min 55 s). The installed [fixed split](configs/Dataset501_UCSFPDGM/splits_final.json) exactly matches the archived 346/74 CSV assignments. The [plans](configs/Dataset501_UCSFPDGM/nnUNetPlans.json) use PlainConvUNet, batch size 2, patch size 128 × 192 × 128, 1 mm isotropic spacing, and per-image masked Z-score normalization. The executed [SLURM script](slurm/preprocess.sbatch) is archived. Planning used the pooled development cohort; validation is therefore an internal development estimate, while the 75 test exams remain outside planning and preprocessing. A separate [GPU smoke test](scripts/smoke_train.py) completed five training batches and two validation batches on an H100 NVL MIG 1g.12gb slice (10.75 GiB visible), peaking at 7.0 GiB allocated. See the [execution record](results/smoke-test.json). Checkpointing and compilation were disabled. The [full baseline batch job](slurm/train_baseline.sbatch) completed with SLURM exit code 0. On 74 validation exams, nnU-Net reported mean foreground Dice 0.8627; per-class Dice was 0.8682 (source label 1), 0.8612 (source label 2), and 0.8587 (source label 4, remapped to 3). See [aggregate validation results](results/validation-baseline-aggregate.json). Checkpoint identity remains to be confirmed; held-out test evaluation has not yet run.
- Exact package versions from the verified HPC environment are archived in [requirements-lock.txt](requirements-lock.txt). This is a pip-freeze snapshot, not a hash-locked dependency resolution.

## Dataset and evaluation design

[UCSF-PDGM Version 5](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/) includes preoperative MRI, tumor annotations, pathology-derived grade labels, and molecular information. It contains 501 exams from 495 unique patients. Distinct exam IDs can refer to the same patient; documented follow-ups must be reconciled before splitting.

The cohort is strongly imbalanced toward grade 4. Overall accuracy alone will not be used to assess classification. The baseline excludes six follow-up exams and uses 495 initial exams (56 grade 2, 43 grade 3, 396 grade 4). Seed-42 grade-stratified splits contain 346 training, 74 validation, and 75 test patients, with zero patient overlap. See [split script](scripts/create_splits.py) and [summary](results/splits/summary.json). Exact [training](results/splits/train.csv), [validation](results/splits/validation.csv), and [test](results/splits/test.csv) CSVs are archived in this repository; uploaded copies were independently checked against the summary for counts, unique patients, follow-up exclusion, and zero overlap.

Before splitting, supplied-mask overlays for UCSF-PDGM-0004, UCSF-PDGM-0231, and UCSF-PDGM-0232 were inspected for alignment. This pre-split QC exposure is documented; no model predictions or test performance have been reviewed. Test data must not guide architecture, hyperparameters, or model selection. Preprocessing statistics and model selection will use development data only. Any future pretrained model requires a training-data overlap audit because some UCSF-PDGM cases also appear in BraTS.

## HPC environment

Verified configuration: Python 3.11.5, PyTorch 2.7.1+cu126, NVIDIA driver 570.211.01, RTX 2080 Ti.

Load Python and activate the environment in each new shell or SLURM job:

```bash
module load Python/3.11.5-GCCcore-13.2.0
source "$HOME/projects/brain-glioma/.venv/bin/activate"
```

The snapshot includes `torch==2.7.1+cu126`, obtained from the PyTorch CUDA 12.6 wheel index (`https://download.pytorch.org/whl/cu126`). Preserve that wheel source when recreating the environment. Python module loading is required in each new shell/job to supply runtime libraries; invoking the virtual environment's Python alone previously failed to locate `libbz2.so.1.0`.

Current working locations:

- Code and environment: `$HOME/projects/brain-glioma`
- Large data and working outputs: `/scratch/markov3/users/$USER/brain-glioma`

Run training through SLURM GPU allocations. Training scripts and reproducible commands will be added as they are validated. Observe cluster quota, retention, and backup policies.

## Version control

Commit source code, SLURM scripts, environment specifications, documentation, de-identified split manifests, aggregate metrics, and selected result figures after each verified milestone.

Raw MRI, source clinical tables, processed arrays, virtual environments, transfer credentials, and large checkpoints are excluded by `.gitignore`. Obtain data separately from TCIA. Aggregate metrics and selected figures can be tracked under `results/`.

## Data citation

Calabrese, E., Villanueva-Meyer, J., Rudie, J., Rauschecker, A., Baid, U., Bakas, S., Cha, S., Mongan, J., & Hess, C. (2022). *The University of California San Francisco Preoperative Diffuse Glioma MRI (UCSF-PDGM) (Version 5)* [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.bdgf-8v37

Dataset license: CC BY 4.0. See the collection page for usage and attribution requirements.
