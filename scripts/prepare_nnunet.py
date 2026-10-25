from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib

root = Path("/scratch/markov3/users/arj94/brain-glioma")
destination = root / "nnUNet_raw/Dataset501_UCSFPDGM"
splits = {
    name: pd.read_csv(root / f"results/splits/{name}.csv")
    for name in ("train", "validation", "test")
}
assert [len(splits[n]) for n in splits] == [346, 74, 75]

patient_sets = [set(d["patient_id"]) for d in splits.values()]
assert sum(map(len, patient_sets)) == len(set.union(*patient_sets)) == 495

if destination.exists():
    raise SystemExit(f"Destination already exists; stopping: {destination}")

development = pd.concat([splits["train"], splits["validation"]])
modalities = ["T1", "T1c", "T2", "FLAIR"]

for _, row in development.iterrows():
    for column in modalities + ["tumor_segmentation"]:
        assert (root / "data" / row[column]).is_file(), row[column]

(destination / "imagesTr").mkdir(parents=True)
(destination / "labelsTr").mkdir()

for index, (_, row) in enumerate(development.iterrows(), 1):
    case = row["exam_id"]
    for channel, modality in enumerate(modalities):
        source = (root / "data" / row[modality]).resolve()
        target = destination / "imagesTr" / f"{case}_{channel:04d}.nii.gz"
        target.symlink_to(source)

    source = nib.load(root / "data" / row["tumor_segmentation"])
    original = np.asarray(source.dataobj)
    assert set(np.unique(original)).issubset({0, 1, 2, 4})
    converted = original.astype(np.uint8)
    converted[converted == 4] = 3
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(converted, source.affine, header)
    nib.save(output, destination / "labelsTr" / f"{case}.nii.gz")

    if index % 25 == 0 or index == len(development):
        print(f"Converted {index}/{len(development)}", flush=True)

configuration = {
    "channel_names": {str(i): name for i, name in enumerate(modalities)},
    "labels": {
        "background": 0,
        "source_label_1": 1,
        "source_label_2": 2,
        "source_label_4": 3
    },
    "numTraining": len(development),
    "file_ending": ".nii.gz",
}
(destination / "dataset.json").write_text(
    json.dumps(configuration, indent=2) + "\n"
)

fixed_split = [{
    "train": sorted(splits["train"]["exam_id"].tolist()),
    "val": sorted(splits["validation"]["exam_id"].tolist())
}]
(destination / "fixed_split.json").write_text(
    json.dumps(fixed_split, indent=2) + "\n"
)

print("Image links:", len(list((destination / "imagesTr").glob("*.nii.gz"))))
print("Converted masks:", len(list((destination / "labelsTr").glob("*.nii.gz"))))
print("Fixed split saved: 346 training / 74 validation")
print("Test exams excluded: 75")
