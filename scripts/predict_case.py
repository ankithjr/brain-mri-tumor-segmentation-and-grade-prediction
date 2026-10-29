"""Run the frozen segmentation-to-grade pipeline on one preprocessed MRI exam."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

CHANNELS = ("T1", "T1c", "T2", "FLAIR")
TARGET = (96, 96, 64)
CLASSIFIER_SHA256 = "b56365465a3c883aa1fa22c0bf73ae0685ea30025b4867bfabfcc5d63466b806"
MODEL_RELATIVE = Path("Dataset501_UCSFPDGM/nnUNetTrainer__nnUNetPlans__3d_fullres")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_images(paths):
    """Validate common geometry without changing voxel order or orientation."""
    images, volumes = [], []
    for path in paths:
        image = nib.load(path)
        volume = np.asarray(image.dataobj, dtype=np.float32)
        if volume.ndim != 3 or not np.isfinite(volume).all():
            raise ValueError(f"Expected finite 3D MRI: {path}")
        if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
            raise ValueError(f"Invalid image affine: {path}")
        spacing = nib.affines.voxel_sizes(image.affine)
        if image.header.get_xyzt_units()[0] not in ("mm", "unknown"):
            raise ValueError(f"Expected MRI coordinates in millimeters: {path}")
        if not np.allclose(spacing, 1.0, rtol=0, atol=1e-3):
            raise ValueError(f"Expected preprocessed 1 mm isotropic images: {path}; got {spacing}")
        if images and (volume.shape != volumes[0].shape or not np.allclose(
            image.affine, images[0].affine, rtol=0, atol=1e-4
        )):
            raise ValueError(f"MRI geometry mismatch: {path}")
        nonzero = volume[volume != 0]
        if nonzero.size == 0 or float(nonzero.std()) < 1e-8:
            raise ValueError(f"Empty or constant MRI: {path}")
        images.append(image)
        volumes.append(volume)
    return images, volumes


def prepare_crop(volumes, mask):
    """Match the archived automatic-crop preprocessing, including empty fallback."""
    if mask.shape != volumes[0].shape or not np.isfinite(mask).all():
        raise ValueError("Invalid predicted mask shape or values")
    if not set(np.unique(mask)).issubset({0, 1, 2, 3}):
        raise ValueError("Expected nnU-Net mask labels 0, 1, 2, 3")
    positions = np.where(mask != 0)
    empty = len(positions[0]) == 0
    crop = tuple(slice(0, size) for size in mask.shape) if empty else tuple(
        slice(max(0, int(positions[axis].min()) - 16),
              min(mask.shape[axis], int(positions[axis].max()) + 17))
        for axis in range(3)
    )
    channels = []
    for volume in volumes:
        brain = volume != 0
        mean = float(volume[brain].mean())
        std = float(volume[brain].std())
        if not brain.any() or std < 1e-8:
            raise ValueError("Empty or constant MRI")
        normalized = np.zeros_like(volume)
        normalized[brain] = np.clip((volume[brain] - mean) / std, -5, 5)
        cropped = normalized[crop]
        resized = zoom(cropped, tuple(t / n for t, n in zip(TARGET, cropped.shape)), order=1)
        if resized.shape != TARGET or not np.isfinite(resized).all():
            raise ValueError("Invalid resized classifier input")
        channels.append(resized.astype(np.float32))
    return np.stack(channels), [[s.start, s.stop] for s in crop], empty


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for channel in CHANNELS:
        parser.add_argument("--" + channel.lower(), type=Path, required=True)
    parser.add_argument("--segmentation-results", type=Path, required=True,
                        help="nnUNet_results root containing Dataset501_UCSFPDGM")
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = [getattr(args, channel.lower()).resolve(strict=True) for channel in CHANNELS]
    if len(set(paths)) != 4:
        raise ValueError("Supply four distinct modality files")
    if any(not p.name.endswith(".nii.gz") for p in paths):
        raise ValueError("Inputs must be .nii.gz files")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    checkpoint = args.classifier_checkpoint.resolve(strict=True)
    classifier_hash = sha256(checkpoint)
    if classifier_hash != CLASSIFIER_SHA256:
        raise ValueError("Classifier checkpoint differs from the frozen epoch-17 model")
    results = args.segmentation_results.resolve(strict=True)
    model = results / MODEL_RELATIVE
    segmentation_checkpoint = model / "fold_0/checkpoint_final.pth"
    for path in (segmentation_checkpoint, model / "dataset.json", model / "plans.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    executable = shutil.which("nnUNetv2_predict")
    if executable is None:
        raise RuntimeError("Activate the verified nnU-Net environment first")

    import torch
    from train_classifier_oracle_crop import Classifier

    if not torch.cuda.is_available():
        raise RuntimeError("Run inside a Slurm GPU allocation")
    torch.set_num_threads(2)
    images, volumes = load_images(paths)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["epoch"] != 17 or state["grade_mapping"] != {"0": 2, "1": 3, "2": 4}:
        raise ValueError("Unexpected classifier metadata")
    classifier = Classifier()
    classifier.load_state_dict(state["model_state"], strict=True)
    classifier.eval()

    provenance = {
        "status": "running", "channels": list(CHANNELS),
        "input_sha256": {c: sha256(p) for c, p in zip(CHANNELS, paths)},
        "classifier_sha256": classifier_hash, "classifier_epoch": 17,
        "segmentation_sha256": sha256(segmentation_checkpoint),
        "segmentation_checkpoint": "checkpoint_final.pth", "fold": 0,
        "input_shape": list(volumes[0].shape), "affine": images[0].affine.tolist(),
        "spatial_units": [image.header.get_xyzt_units()[0] for image in images],
        "coordinate_assumption": "Coordinates interpreted as millimeters; inputs must match UCSF-PDGM preprocessing",
    }
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "prediction.json"
    report_path.write_text(json.dumps(provenance, indent=2) + "\n")
    try:
        inputs = output / "nnunet-input"
        inputs.mkdir()
        for index, path in enumerate(paths):
            (inputs / f"case_{index:04d}.nii.gz").symlink_to(path)
        predictions = output / "segmentation"
        env = os.environ.copy()
        env["nnUNet_results"] = str(results)
        env["nnUNet_compile"] = "false"
        env["PYTHONUNBUFFERED"] = "1"
        command = [executable, "-d", "501", "-c", "3d_fullres", "-f", "0",
                   "-tr", "nnUNetTrainer", "-p", "nnUNetPlans",
                   "-chk", "checkpoint_final.pth", "-i", str(inputs),
                   "-o", str(predictions), "-npp", "2", "-nps", "2"]
        with (output / "segmentation.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        mask_path = predictions / "case.nii.gz"
        mask_image = nib.load(mask_path)
        if not np.allclose(mask_image.affine, images[0].affine, rtol=0, atol=1e-4):
            raise ValueError("Predicted mask geometry differs from MRI")
        mask = np.asarray(mask_image.dataobj)
        crop, bounds, fallback = prepare_crop(volumes, mask)
        # Segmentation runs in a subprocess, releasing GPU memory before grading.
        classifier = classifier.to("cuda")
        with torch.inference_mode():
            probabilities = torch.softmax(
                classifier(torch.from_numpy(crop[None]).to("cuda")), dim=1
            )[0].cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise ValueError("Non-finite classifier output")
        provenance.update({
            "status": "complete", "predicted_grade": int(probabilities.argmax()) + 2,
            "probabilities": {str(g): float(p) for g, p in zip((2, 3, 4), probabilities)},
            "probability_note": "Uncalibrated softmax outputs; research use only",
            "segmentation": "segmentation/case.nii.gz",
            "mask_labels": {"0": "background", "1": "source label 1",
                            "2": "source label 2", "3": "source label 4"},
            "crop_bounds_start_stop": bounds, "margin_voxels": 16,
            "empty_prediction_fallback": fallback,
            "classifier_input_shape": list(crop.shape),
        })
        report_path.write_text(json.dumps(provenance, indent=2) + "\n")
        print(json.dumps(provenance, indent=2), flush=True)
    except Exception as error:
        provenance.update(status="failed", error=str(error))
        report_path.write_text(json.dumps(provenance, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
