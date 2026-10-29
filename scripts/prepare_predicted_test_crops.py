
from pathlib import Path

import json

import numpy as np

import pandas as pd

import nibabel as nib

from scipy.ndimage import zoom



root = Path("/scratch/markov3/users/arj94/brain-glioma")

predictions = root / "results/test-evaluation/predictions"

destination = root / "preprocessed/classifier_predicted_crop_test_v1/test"

frame = pd.read_csv(root / "results/splits/test.csv")

modalities = ["T1", "T1c", "T2", "FLAIR"]

target_shape = (96, 96, 64)

margin = 16



assert len(frame) == 75

if destination.exists():

    raise SystemExit(f"Destination already exists: {destination}")



for _, row in frame.iterrows():

    path = predictions / f"{row['exam_id']}.nii.gz"

    if not path.is_file():

        raise FileNotFoundError(path)



destination.mkdir(parents=True)

fallbacks = []



for index, (_, row) in enumerate(frame.iterrows(), 1):

    mask_image = nib.load(predictions / f"{row['exam_id']}.nii.gz")

    mask = np.asarray(mask_image.dataobj)

    if mask.ndim != 3 or not np.isfinite(mask).all():

        raise ValueError(f"Invalid prediction: {row['exam_id']}")



    positions = np.where(mask != 0)

    if len(positions[0]):

        crop = tuple(

            slice(

                max(0, int(positions[axis].min()) - margin),

                min(mask.shape[axis], int(positions[axis].max()) + margin + 1),

            )

            for axis in range(3)

        )

    else:

        crop = tuple(slice(0, size) for size in mask.shape)

        fallbacks.append(row["exam_id"])



    channels = []

    for modality in modalities:

        path = root / "data" / row[modality]

        image = nib.load(path)

        volume = np.asarray(image.dataobj, dtype=np.float32)

        if volume.shape != mask.shape or not np.allclose(

            image.affine, mask_image.affine, rtol=0, atol=1e-4

        ):

            raise ValueError(f"Geometry mismatch: {path}")

        if not np.isfinite(volume).all():

            raise ValueError(f"Non-finite image: {path}")



        brain = volume != 0

        if not brain.any():

            raise ValueError(f"Empty image: {path}")

        mean = float(volume[brain].mean())

        std = float(volume[brain].std())

        if std < 1e-8:

            raise ValueError(f"Near-zero intensity variation: {path}")



        normalized = np.zeros_like(volume)

        normalized[brain] = np.clip((volume[brain] - mean) / std, -5, 5)

        cropped = normalized[crop]

        resized = zoom(

            cropped,

            tuple(t / n for t, n in zip(target_shape, cropped.shape)),

            order=1,

        )

        assert resized.shape == target_shape

        assert np.isfinite(resized).all()

        channels.append(resized.astype(np.float32))



    np.savez_compressed(

        destination / f"{row['exam_id']}.npz",

        image=np.stack(channels),

        grade=np.int64(row["grade"]),

    )

    if index % 25 == 0 or index == len(frame):

        print(f"Prepared {index}/{len(frame)}", flush=True)



report = {

    "cases": len(frame),

    "crop_source": "automatic held-out test segmentation predictions",

    "margin_voxels": margin,

    "empty_prediction_fallback": "whole image",

    "fallback_cases": fallbacks,

}

(destination.parent / "preprocessing.json").write_text(

    json.dumps(report, indent=2) + "\n"

)

print(json.dumps(report, indent=2), flush=True)

print("PREDICTED TEST CROPS COMPLETE", flush=True)

