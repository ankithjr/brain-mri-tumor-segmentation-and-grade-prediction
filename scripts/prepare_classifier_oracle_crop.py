
from pathlib import Path

import json

import numpy as np

import pandas as pd

import nibabel as nib

from scipy.ndimage import zoom



root = Path("/scratch/markov3/users/arj94/brain-glioma")

destination = root / "preprocessed/classifier_oracle_crop_v1"

modalities = ["T1", "T1c", "T2", "FLAIR"]

target_shape = (96, 96, 64)

margin = 16



if destination.exists():

    raise SystemExit(f"Destination already exists: {destination}")



splits = {

    name: pd.read_csv(root / f"results/splits/{name}.csv")

    for name in ("train", "validation")

}

assert len(splits["train"]) == 346

assert len(splits["validation"]) == 74

assert not set(splits["train"]["patient_id"]) & set(splits["validation"]["patient_id"])



for frame in splits.values():

    for _, row in frame.iterrows():

        for field in modalities + ["tumor_segmentation"]:

            if not (root / "data" / row[field]).is_file():

                raise FileNotFoundError(row[field])



destination.mkdir(parents=True)



for split_name, frame in splits.items():

    output = destination / split_name

    output.mkdir()



    for index, (_, row) in enumerate(frame.iterrows(), 1):

        mask_path = root / "data" / row["tumor_segmentation"]

        mask_image = nib.load(mask_path)

        mask = np.asarray(mask_image.dataobj)

        if mask.ndim != 3 or not np.isfinite(mask).all():

            raise ValueError(f"Invalid mask: {mask_path}")



        positions = np.where(mask != 0)

        if len(positions[0]) == 0:

            raise ValueError(f"Empty mask: {mask_path}")



        bounds = []

        for axis in range(3):

            start = max(0, int(positions[axis].min()) - margin)

            end = min(mask.shape[axis], int(positions[axis].max()) + margin + 1)

            bounds.append(slice(start, end))

        crop = tuple(bounds)



        channels = []

        for modality in modalities:

            image_path = root / "data" / row[modality]

            image = nib.load(image_path)

            volume = np.asarray(image.dataobj, dtype=np.float32)



            if volume.shape != mask.shape or not np.allclose(

                image.affine, mask_image.affine, rtol=0, atol=1e-4

            ):

                raise ValueError(f"Image/mask geometry mismatch: {image_path}")

            if not np.isfinite(volume).all():

                raise ValueError(f"Non-finite image: {image_path}")



            brain = volume != 0

            if not brain.any():

                raise ValueError(f"Empty image: {image_path}")

            mean = float(volume[brain].mean())

            std = float(volume[brain].std())

            if std < 1e-8:

                raise ValueError(f"Near-zero intensity variation: {image_path}")



            normalized = np.zeros_like(volume)

            normalized[brain] = np.clip((volume[brain] - mean) / std, -5, 5)

            cropped = normalized[crop]

            factors = tuple(

                target / current

                for target, current in zip(target_shape, cropped.shape)

            )

            resized = zoom(cropped, factors, order=1)

            if resized.shape != target_shape or not np.isfinite(resized).all():

                raise ValueError(f"Invalid resized crop: {image_path}")

            channels.append(resized.astype(np.float32))



        np.savez_compressed(

            output / f"{row['exam_id']}.npz",

            image=np.stack(channels),

            grade=np.int64(row["grade"]),

        )



        if index % 25 == 0 or index == len(frame):

            print(f"{split_name}: {index}/{len(frame)}", flush=True)



config = {

    "experiment": "oracle crop using supplied reference tumor masks",

    "modalities": modalities,

    "margin_voxels": margin,

    "target_shape": list(target_shape),

    "normalization": "per-volume nonzero-voxel z-score, clipped to [-5, 5]",

    "interpolation": "linear",

    "splits": {name: len(frame) for name, frame in splits.items()},

}

(destination / "preprocessing.json").write_text(

    json.dumps(config, indent=2) + "\n"

)

print("ORACLE CROP PREPARATION COMPLETE", flush=True)

