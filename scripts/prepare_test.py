
from pathlib import Path

import numpy as np

import pandas as pd

import nibabel as nib



root = Path("/scratch/markov3/users/arj94/brain-glioma")

test = pd.read_csv(root / "results/splits/test.csv")

destination = root / "results/test-evaluation"



assert len(test) == 75

assert test["patient_id"].is_unique

assert not test["is_followup"].any()



if destination.exists():

    raise SystemExit(f"Destination already exists: {destination}")



for _, row in test.iterrows():

    for column in ["T1", "T1c", "T2", "FLAIR", "tumor_segmentation"]:

        assert (root / "data" / row[column]).is_file(), row[column]



images = destination / "images"

labels = destination / "labels"

images.mkdir(parents=True)

labels.mkdir()



for index, (_, row) in enumerate(test.iterrows(), 1):

    case = row["exam_id"]

    for channel, modality in enumerate(["T1", "T1c", "T2", "FLAIR"]):

        source = (root / "data" / row[modality]).resolve()

        (images / f"{case}_{channel:04d}.nii.gz").symlink_to(source)



    source = nib.load(root / "data" / row["tumor_segmentation"])

    original = np.asarray(source.dataobj)

    assert set(np.unique(original)).issubset({0, 1, 2, 4})

    converted = original.astype(np.uint8)

    converted[converted == 4] = 3

    header = source.header.copy()

    header.set_data_dtype(np.uint8)

    nib.save(

        nib.Nifti1Image(converted, source.affine, header),

        labels / f"{case}.nii.gz",

    )



    if index % 25 == 0 or index == len(test):

        print(f"Prepared {index}/{len(test)}", flush=True)



print("Image links:", len(list(images.glob("*.nii.gz"))))

print("Converted reference masks:", len(list(labels.glob("*.nii.gz"))))

