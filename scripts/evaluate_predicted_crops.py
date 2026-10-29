
from pathlib import Path

import hashlib

import json

import numpy as np

import pandas as pd

import torch

from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix



from train_classifier_oracle_crop import Classifier



root = Path("/scratch/markov3/users/arj94/brain-glioma")

inputs = root / "preprocessed/classifier_predicted_crop_v1/validation"

checkpoint_path = root / "checkpoints/grade_classifier_oracle_crop/checkpoint_best.pt"

output = root / "results/grade-predicted-crop-validation"



def main():

    torch.set_num_threads(2)

    if not torch.cuda.is_available():

        raise RuntimeError("GPU required")

    if output.exists():

        raise SystemExit(f"Output already exists: {output}")



    frame = pd.read_csv(root / "results/splits/validation.csv")

    assert len(frame) == 74 and frame["patient_id"].is_unique



    checkpoint = torch.load(

        checkpoint_path, map_location="cpu", weights_only=True

    )

    assert checkpoint["epoch"] == 17

    device = torch.device("cuda")

    model = Classifier().to(device)

    model.load_state_dict(checkpoint["model_state"])

    model.eval()

    probabilities = []



    with torch.no_grad():

        for start in range(0, len(frame), 8):

            images = []

            for _, row in frame.iloc[start:start + 8].iterrows():

                path = inputs / f"{row['exam_id']}.npz"

                with np.load(path) as sample:

                    image = sample["image"]

                    assert int(sample["grade"]) == int(row["grade"])

                    assert image.shape == (4, 96, 96, 64)

                    assert np.isfinite(image).all()

                    images.append(image.astype(np.float32))



            batch = torch.from_numpy(np.stack(images)).to(device)

            probabilities.extend(

                torch.softmax(model(batch), dim=1).cpu().numpy()

            )



    probabilities = np.asarray(probabilities)

    truth = frame["grade"].to_numpy(dtype=int)

    predicted = probabilities.argmax(axis=1) + 2

    matrix = confusion_matrix(truth, predicted, labels=[2, 3, 4])

    report = {

        "cases": len(frame),

        "checkpoint_epoch": int(checkpoint["epoch"]),

        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),

        "crop_source": "automatic segmentation predictions",

        "macro_f1": float(f1_score(

            truth, predicted, labels=[2, 3, 4],

            average="macro", zero_division=0

        )),

        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),

        "grade_order": [2, 3, 4],

        "confusion_matrix": matrix.tolist(),

        "recall_per_grade": {

            str(grade): float(matrix[i, i] / matrix[i].sum())

            for i, grade in enumerate([2, 3, 4])

        },

    }



    output.mkdir(parents=True)

    (output / "summary.json").write_text(

        json.dumps(report, indent=2) + "\n"

    )

    pd.DataFrame({

        "exam_id": frame["exam_id"],

        "true_grade": truth,

        "predicted_grade": predicted,

        "probability_grade_2": probabilities[:, 0],

        "probability_grade_3": probabilities[:, 1],

        "probability_grade_4": probabilities[:, 2],

    }).to_csv(output / "predictions.csv", index=False)

    print(json.dumps(report, indent=2), flush=True)

    print("AUTOMATIC-CROP VALIDATION COMPLETE", flush=True)



if __name__ == "__main__":

    main()

