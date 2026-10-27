
from pathlib import Path

import json

import numpy as np

import pandas as pd

import nibabel as nib

from scipy.ndimage import zoom



root = Path("/scratch/markov3/users/arj94/brain-glioma")

destination = root / "preprocessed/classifier_mri_v1"

modalities = ["T1", "T1c", "T2", "FLAIR"]

target_shape = (96, 96, 64)



if destination.exists():

    raise SystemExit(f"Destination already exists; stopping: {destination}")



splits = {

    name: pd.read_csv(root / f"results/splits/{name}.csv")

    for name in ("train", "validation")

}

assert len(splits["train"]) == 346

assert len(splits["validation"]) == 74

assert not set(splits["train"]["patient_id"]) & set(splits["validation"]["patient_id"])



for frame in splits.values():

    for _, row in frame.iterrows():

        for modality in modalities:

            path = root / "data" / row[modality]

            if not path.is_file():

                raise FileNotFoundError(path)



destination.mkdir(parents=True)

print(f"Output: {destination}", flush=True)



for split_name, frame in splits.items():

    output = destination / split_name

    output.mkdir()



    for index, (_, row) in enumerate(frame.iterrows(), 1):

        channels = []



        for modality in modalities:

            path = root / "data" / row[modality]

            volume = np.asarray(nib.load(path).dataobj, dtype=np.float32)



            if volume.ndim != 3 or not np.isfinite(volume).all():

                raise ValueError(f"Invalid image: {path}")



            brain = volume != 0

            if not brain.any():

                raise ValueError(f"Empty image: {path}")



            # Normalize each scan using its own nonzero brain voxels.

            mean = float(volume[brain].mean())

            std = float(volume[brain].std())

            if std < 1e-8:

                raise ValueError(f"Near-zero intensity variation: {path}")



            normalized = np.zeros_like(volume)

            normalized[brain] = np.clip(

                (volume[brain] - mean) / std, -5, 5

            )



            factors = tuple(

                target / current

                for target, current in zip(target_shape, volume.shape)

            )

            resized = zoom(normalized, factors, order=1)



            if resized.shape != target_shape or not np.isfinite(resized).all():

                raise ValueError(f"Unexpected resized image: {path}")



            channels.append(resized.astype(np.float32))



        image = np.stack(channels)

        np.savez_compressed(

            output / f"{row['exam_id']}.npz",

            image=image,

            grade=np.int64(row["grade"]),

        )



        if index % 25 == 0 or index == len(frame):

            print(f"{split_name}: {index}/{len(frame)}", flush=True)



config = {

    "modalities": modalities,

    "target_shape": list(target_shape),

    "normalization": "per-volume nonzero-voxel z-score, clipped to [-5, 5]",

    "interpolation": "linear",

    "splits": {name: len(frame) for name, frame in splits.items()},

}

(destination / "preprocessing.json").write_text(

    json.dumps(config, indent=2) + "\n"

)

print("CLASSIFIER PREPARATION COMPLETE", flush=True)

