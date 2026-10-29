
from pathlib import Path

import json

import numpy as np

import torch

from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix



from train_classifier_balanced import Classifier, MRIDataset



root = Path("/scratch/markov3/users/arj94/brain-glioma")

checkpoint_path = root / "checkpoints/grade_classifier_balanced_sampler/checkpoint_best.pt"

output = root / "results/grade-balanced-fit-check.json"



def main():

    if output.exists():

        raise SystemExit(f"Result already exists: {output}")

    if not torch.cuda.is_available():

        raise RuntimeError("GPU is required")



    torch.set_num_threads(2)

    device = torch.device("cuda")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)



    model = Classifier().to(device)

    model.load_state_dict(checkpoint["model_state"])

    model.eval()



    report = {

        "checkpoint_epoch": int(checkpoint["epoch"]),

        "checkpoint_validation_macro_f1": float(checkpoint["validation_macro_f1"]),

        "splits": {},

    }



    for split in ("train", "validation"):

        dataset = MRIDataset(split)

        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        truth, predicted = [], []



        with torch.no_grad():

            for images, targets in loader:

                logits = model(images.to(device))

                truth.extend(targets.tolist())

                predicted.extend(logits.argmax(dim=1).cpu().tolist())



        matrix = confusion_matrix(truth, predicted, labels=[0, 1, 2])

        report["splits"][split] = {

            "cases": len(truth),

            "macro_f1": float(f1_score(

                truth, predicted, labels=[0, 1, 2],

                average="macro", zero_division=0,

            )),

            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),

            "confusion_matrix": matrix.tolist(),

        }



    output.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2), flush=True)



if __name__ == "__main__":

    main()

