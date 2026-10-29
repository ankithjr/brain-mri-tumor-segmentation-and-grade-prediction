# Executed HPC source archive

The user-provided `portfolio-code.tar.gz` contains 15 Python scripts and 10 Slurm batch scripts from the Markov project. Source files are preserved verbatim in `scripts/` and `slurm/`; Python bytecode caches are excluded. Existing repository-only analysis and validation scripts are retained.

All 15 Python files passed AST parsing and all 10 batch files passed `bash -n` during archival. These are syntax checks, not a rerun of training or inference. A text scan found no embedded credentials. Raw images, clinical source tables, processed arrays, and model checkpoints were not included in this source archive.

## Pipeline entry points

| Stage | Source |
| --- | --- |
| Held-out segmentation inputs and reference conversion | `scripts/prepare_test.py` |
| Frozen final segmentation checkpoint inference | `slurm/predict_test.sbatch` |
| Reference-mask training and validation crops | `scripts/prepare_classifier_oracle_crop.py` |
| Crop classifier training | `scripts/train_classifier_oracle_crop.py` |
| Automatic validation crops | `scripts/prepare_predicted_val_crops.py` |
| Automatic validation grading | `scripts/evaluate_predicted_crops.py` |
| Automatic held-out test crops | `scripts/prepare_predicted_test_crops.py` |
| Frozen held-out grading | `scripts/evaluate_predicted_test_crops.py` |
| Whole-brain balanced-classifier fit diagnostic | `scripts/check_grade_fit.py` |

The test grading evaluator verifies the epoch-17 classifier SHA-256 against `configs/grade_pipeline_frozen.json`. Reference masks are used for classifier training crops; automatic masks are used for the reported automatic validation and test results.

## Execution limits

These are the executed Markov scripts, with site-specific absolute paths and environment modules. They require the separately obtained data, manifests, environment, and trained checkpoints. Several preparation and evaluation scripts deliberately stop when an output directory already exists; do not delete completed results to rerun them. The scripts are not yet a portable inference interface.

The full oracle classifier training history and qualitative MRI prediction figures remain to be archived. Keep the evaluated test cohort fixed and do not use its results for further tuning.
