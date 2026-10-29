"""Render the first correctly graded validation case by exam ID."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd


def dice(reference, prediction):
    denominator = int(reference.sum()) + int(prediction.sum())
    return float(2 * np.count_nonzero(reference & prediction) / denominator) if denominator else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    predictions_csv = args.root / "results/grade-predicted-crop-validation/predictions.csv"
    predictions = pd.read_csv(predictions_csv)
    if predictions["exam_id"].duplicated().any():
        raise ValueError("Duplicate prediction IDs")
    probabilities = predictions[[f"probability_grade_{g}" for g in (2, 3, 4)]].to_numpy()
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Invalid probabilities")
    if not np.array_equal(probabilities.argmax(axis=1) + 2, predictions["predicted_grade"].to_numpy()):
        raise ValueError("Probability argmax differs from predictions")
    correct = predictions.loc[predictions.true_grade.eq(predictions.predicted_grade)].sort_values("exam_id")
    if correct.empty:
        raise ValueError("No correctly graded validation cases")
    selected = correct.iloc[0]
    exam = str(selected["exam_id"])
    manifest = pd.read_csv(args.root / "results/splits/validation.csv")
    if set(predictions.exam_id) != set(manifest.exam_id):
        raise ValueError("Validation cohort mismatch")
    rows = manifest.loc[manifest.exam_id.eq(exam)]
    if len(rows) != 1:
        raise ValueError("Expected exactly one manifest row")
    row = rows.iloc[0]
    if int(row["grade"]) != int(selected["true_grade"]):
        raise ValueError("Reference grade mismatch")
    report = {"predicted_grade": int(selected["predicted_grade"]),
              "probabilities": {str(g): float(selected[f"probability_grade_{g}"]) for g in (2, 3, 4)}}
    modalities = ["T1", "T1c", "T2", "FLAIR"]
    image_paths = [args.root / "data" / row[m] for m in modalities]
    mask_folder = args.root / "nnUNet_results/Dataset501_UCSFPDGM/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation"
    paths = image_paths + [args.root / "data" / row["tumor_segmentation"], mask_folder / f"{exam}.nii.gz"]
    originals = [nib.load(path) for path in paths]
    for image in originals:
        if image.shape != originals[0].shape or not np.allclose(
            image.affine, originals[0].affine, rtol=0, atol=1e-4
        ):
            raise ValueError("Image/mask geometry mismatch")
    images = [nib.as_closest_canonical(image) for image in originals]
    arrays = [np.asarray(image.dataobj) for image in images]
    if any(a.ndim != 3 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Expected finite 3D volumes")
    reference, prediction = arrays[-2:]
    if not set(np.unique(reference)).issubset({0, 1, 2, 4}):
        raise ValueError("Unexpected source mask labels")
    if not set(np.unique(prediction)).issubset({0, 1, 2, 3}):
        raise ValueError("Unexpected predicted mask labels")
    reference = reference.astype(np.uint8).copy()
    reference[reference == 4] = 3
    z = int(np.count_nonzero(reference, axis=(0, 1)).argmax())
    metrics = {str(label): dice(reference == label, prediction == label) for label in (1, 2, 3)}
    whole = dice(reference != 0, prediction != 0)
    record = {
        "exam_id": exam, "cohort": "validation", "reference_grade": int(row["grade"]),
        "predicted_grade": report["predicted_grade"], "probabilities": report["probabilities"],
        "dice_3d_per_remapped_label": metrics, "whole_tumor_dice_3d": whole,
        "display_orientation": "RAS canonical", "slice_axis": 2, "slice_index": z,
        "slice_selection": "Largest reference foreground area; first index on ties",
        "label_mapping": {"1": "source 1", "2": "source 2", "3": "source 4"},
        "selection_note": "First correctly graded validation case by sorted exam ID; illustrative, not a representative sample",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    cmap = ListedColormap(["#00d4ff", "#ffdc42", "#ff5364"])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, axes = plt.subplots(2, 4, figsize=(14, 8), facecolor="#101820")
    for col, (modality, volume) in enumerate(zip(modalities, arrays[:4])):
        nonzero = volume[volume != 0]
        low, high = np.percentile(nonzero, [1, 99])
        for i, mask in enumerate((reference, prediction)):
            ax = axes[i, col]
            ax.imshow(volume[:, :, z].T, cmap="gray", origin="lower", vmin=low, vmax=high)
            ax.imshow(np.ma.masked_equal(mask[:, :, z].T, 0), cmap=cmap, norm=norm,
                      origin="lower", alpha=0.45, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{modality} | {'Reference' if i == 0 else 'Prediction'}", color="white", fontsize=12)
    fig.suptitle(f"{exam}: tumor segmentation and correct grade prediction\n"
                 f"Reference grade {int(row['grade'])} | Predicted grade {report['predicted_grade']}"
                 f" | Whole-tumor 3D Dice {whole:.3f}", color="white", fontsize=16)
    legend = fig.legend(handles=[Patch(color=cmap(i), label=f"Source label {label}")
                                  for i, label in enumerate((1, 2, 4))],
                        loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3, frameon=False)
    for text in legend.get_texts():
        text.set_color("white")
    fig.text(0.5, 0.025, f"Validation example; canonical axial slice {z}, selected by largest reference tumor area. "
             "Overlay colors denote numeric labels.", ha="center", color="#d1d8df", fontsize=10)
    fig.subplots_adjust(top=0.83, bottom=0.13, hspace=0.16, wspace=0.04)
    destination = args.output / f"{exam}-mask-review.png"
    fig.savefig(destination, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    (args.output / "review.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
