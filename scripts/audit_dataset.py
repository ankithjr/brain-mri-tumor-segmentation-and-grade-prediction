
from pathlib import Path

import json

import re

import numpy as np

import pandas as pd

import nibabel as nib



root = Path("/scratch/markov3/users/arj94/brain-glioma")

data = root / "data"

results = root / "results"

results.mkdir(parents=True, exist_ok=True)



def normalize(value):

    value = str(value).removesuffix("_nifti")

    match = re.fullmatch(r"UCSF-PDGM-(\d+)(_FU\d+d)?", value)

    if not match:

        raise ValueError(f"Unrecognized ID: {value}")

    return f"UCSF-PDGM-{int(match[1]):04d}{match[2] or ''}"



metadata = pd.read_csv(data / "UCSF-PDGM-metadata_v5.csv")

metadata["exam_id"] = metadata["ID"].map(normalize)

assert metadata["exam_id"].is_unique

metadata = metadata.set_index("exam_id")



folders = sorted(p for p in (data / "UCSF-PDGM-core").iterdir() if p.is_dir())

assert {normalize(p.name) for p in folders} == set(metadata.index)



modalities = ("T1", "T1c", "T2", "FLAIR", "tumor_segmentation")

rows, issues, observed_labels = [], [], set()



for index, folder in enumerate(folders, 1):

    exam = normalize(folder.name)

    row = {

        "exam_id": exam,

        "patient_id": exam.split("_FU")[0],

        "is_followup": "_FU" in exam,

        "grade": int(metadata.loc[exam, "WHO CNS Grade"]),

    }

    reference = None

    start_issues = len(issues)



    for modality in modalities:

        try:

            matches = list(folder.glob(f"*_{modality}.nii.gz"))

            if len(matches) != 1:

                raise ValueError(f"Expected one file; found {len(matches)}")

            path = matches[0]

            row[modality] = str(path.relative_to(data))

            image = nib.load(path)

            array = np.asarray(image.dataobj)



            if array.ndim != 3:

                raise ValueError(f"Expected 3D image; found {array.shape}")

            if not np.isfinite(image.affine).all():

                raise ValueError("Non-finite affine")

            if not np.isfinite(array).all():

                raise ValueError("Non-finite voxel values")



            if reference is None:

                reference = (image.shape, image.affine.copy())

                row["shape"] = "x".join(map(str, image.shape))

            elif image.shape != reference[0] or not np.allclose(

                image.affine, reference[1], rtol=0, atol=1e-4

            ):

                raise ValueError("Shape or affine differs from T1")



            if modality == "tumor_segmentation":

                labels = np.unique(array)

                if not np.equal(labels, np.round(labels)).all():

                    raise ValueError(f"Non-integer mask labels: {labels}")

                observed_labels.update(int(v) for v in labels)

                row["mask_labels"] = ",".join(str(int(v)) for v in labels)

                row["tumor_voxels"] = int(np.count_nonzero(array))

                if row["tumor_voxels"] == 0:

                    raise ValueError("Empty tumor mask")

            del array

        except Exception as error:

            issues.append({"exam_id": exam, "modality": modality, "error": str(error)})



    row["audit_passed"] = len(issues) == start_issues

    rows.append(row)

    if index % 25 == 0 or index == len(folders):

        print(f"Audited {index}/{len(folders)} exams", flush=True)



manifest = pd.DataFrame(rows)

manifest.to_csv(results / "exam-manifest.csv", index=False)



report = {

    "exams": len(manifest),

    "patients": int(manifest["patient_id"].nunique()),

    "passing_exams": int(manifest["audit_passed"].sum()),

    "observed_mask_labels": sorted(observed_labels),

    "issues": issues,

}

(results / "dataset-audit.json").write_text(json.dumps(report, indent=2) + "\n")

print(json.dumps(report, indent=2), flush=True)

raise SystemExit(bool(issues))

