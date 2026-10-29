
from pathlib import Path

import json

import pandas as pd

from sklearn.model_selection import train_test_split



root = Path("/scratch/markov3/users/arj94/brain-glioma")

d = pd.read_csv(root / "results/exam-manifest.csv")

d = d[d["is_followup"].eq(False)].sort_values("patient_id").copy()



assert len(d) == 495

assert d["patient_id"].is_unique

assert d["audit_passed"].all()



train, remainder = train_test_split(

    d, test_size=0.30, stratify=d["grade"], random_state=42

)

validation, test = train_test_split(

    remainder, test_size=0.50,

    stratify=remainder["grade"], random_state=42

)



parts = {"train": train, "validation": validation, "test": test}

patient_sets = {name: set(frame["patient_id"]) for name, frame in parts.items()}

assert not patient_sets["train"] & patient_sets["validation"]

assert not patient_sets["train"] & patient_sets["test"]

assert not patient_sets["validation"] & patient_sets["test"]

assert len(set.union(*patient_sets.values())) == 495



output = root / "results/splits"

output.mkdir(parents=True, exist_ok=True)

if any(output.glob("*.csv")):

    raise SystemExit("Split files already exist; stopping to avoid overwriting.")



summary = {

    "seed": 42,

    "cohort": "Initial exams only; six follow-ups excluded",

    "patients": 495,

    "patient_overlap": 0,

    "splits": {},

}

for name, frame in parts.items():

    frame = frame.sort_values("patient_id")

    frame.to_csv(output / f"{name}.csv", index=False)

    counts = frame["grade"].value_counts().sort_index()

    summary["splits"][name] = {

        "patients": len(frame),

        "grade_counts": {str(k): int(v) for k, v in counts.items()},

    }



(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

print(json.dumps(summary, indent=2))

