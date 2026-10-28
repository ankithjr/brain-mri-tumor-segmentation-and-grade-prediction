"""Verify frozen grade predictions and quantify test-set sampling uncertainty.

Intervals condition on a fixed trained pipeline. Stratified patient bootstrap
preserves the observed number of patients in each grade; it does not capture
training variation, model selection, or generalization to another institution.
"""
import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GRADES = [2, 3, 4]
FROZEN_HASH = "b56365465a3c883aa1fa22c0bf73ae0685ea30025b4867bfabfcc5d63466b806"


def metrics(matrix):
    support = [sum(row) for row in matrix]
    predicted = [sum(row[j] for row in matrix) for j in range(3)]
    recall = [matrix[i][i] / support[i] for i in range(3)]
    f1 = [2 * matrix[i][i] / (support[i] + predicted[i]) for i in range(3)]
    return {
        "macro_f1": sum(f1) / 3,
        "balanced_accuracy": sum(recall) / 3,
        "accuracy": sum(matrix[i][i] for i in range(3)) / sum(support),
    }


def wilson(successes, total):
    z = 1.959963984540054
    fraction = successes / total
    denominator = 1 + z * z / total
    center = (fraction + z * z / (2 * total)) / denominator
    half = z * math.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total**2)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert args.replicates >= 1000
    with args.predictions.open() as stream:
        rows = list(csv.DictReader(stream))
    with args.test_split.open() as stream:
        test = list(csv.DictReader(stream))
    summary = json.loads(args.summary.read_text())
    assert len(rows) == len(test) == summary["cases"] == 75
    assert summary["checkpoint_epoch"] == 17
    assert summary["checkpoint_sha256"] == FROZEN_HASH
    assert len({row["exam_id"] for row in rows}) == 75
    assert len({row["patient_id"] for row in test}) == 75
    expected = {row["exam_id"]: int(row["grade"]) for row in test}
    assert set(expected) == {row["exam_id"] for row in rows}
    matrix = [[0] * 3 for _ in range(3)]
    groups = [[] for _ in range(3)]
    for row in rows:
        true, pred = int(row["true_grade"]), int(row["predicted_grade"])
        assert true == expected[row["exam_id"]]
        assert true in GRADES and pred in GRADES
        probabilities = [float(row[f"probability_grade_{grade}"]) for grade in GRADES]
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
        assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-5)
        assert GRADES[max(range(3), key=lambda i: probabilities[i])] == pred
        matrix[true - 2][pred - 2] += 1
        groups[true - 2].append(pred - 2)
    assert matrix == summary["confusion_matrix"]
    point = metrics(matrix)
    for name in ("macro_f1", "balanced_accuracy"):
        assert math.isclose(point[name], summary[name], abs_tol=1e-12)
    for i, grade in enumerate(GRADES):
        assert math.isclose(matrix[i][i] / sum(matrix[i]), summary["recall_per_grade"][str(grade)], abs_tol=1e-12)
    assert [len(g) for g in groups] == [9, 6, 60]

    rng = random.Random(args.seed)
    bootstrap = {name: [] for name in point}
    for _ in range(args.replicates):
        sample = [[0] * 3 for _ in range(3)]
        for i, group in enumerate(groups):
            for _ in group:
                sample[i][rng.choice(group)] += 1
        for name, value in metrics(sample).items():
            bootstrap[name].append(value)
    intervals = {name: np.quantile(values, [0.025, 0.975]).tolist() for name, values in bootstrap.items()}
    recall = {
        str(grade): {
            "correct": matrix[i][i], "patients": sum(matrix[i]),
            "recall": matrix[i][i] / sum(matrix[i]),
            "wilson_95_ci": wilson(matrix[i][i], sum(matrix[i])),
        }
        for i, grade in enumerate(GRADES)
    }
    majority_matrix = [[0, 0, len(group)] for group in groups]
    report = {
        "cohort": "held-out test", "patients": 75, "grade_order": GRADES,
        "checkpoint_sha256": FROZEN_HASH,
        "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "verification": "75 unique exams and patients exactly match archived test split; labels, argmax probabilities, summary metrics and checkpoint hash agree",
        "confusion_matrix": matrix, "metrics": point,
        "stratified_patient_bootstrap_95_ci": intervals,
        "bootstrap": {"replicates": args.replicates, "seed": args.seed, "method": "percentile; resample patients with replacement within each true grade; fixed grade counts 9/6/60"},
        "grade_recall": recall, "always_grade_4_baseline": metrics(majority_matrix),
        "limitations": "Fixed-model, within-cohort uncertainty only; excludes training and model-selection variability and external-site generalization. Grader trained using reference-mask crops and evaluated using automatically predicted crops.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "grade-test-uncertainty.json").write_text(json.dumps(report, indent=2) + "\n")
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1, 1.1]})
    normalized = np.asarray(matrix) / np.asarray(matrix).sum(axis=1, keepdims=True)
    axes[0].imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axes[0].set(xticks=range(3), yticks=range(3), xticklabels=GRADES, yticklabels=GRADES,
                xlabel="Predicted WHO grade", ylabel="Reference WHO grade", title="Confusion matrix")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{matrix[i][j]}\n{normalized[i,j]:.1%}", ha="center", va="center",
                         color="white" if normalized[i,j] > 0.6 else "#15293d")
    axes[0].set_xticks(np.arange(-0.5, 3, 1), minor=True)
    axes[0].set_yticks(np.arange(-0.5, 3, 1), minor=True)
    axes[0].grid(which="minor", color="white", linewidth=2)
    axes[0].tick_params(which="minor", bottom=False, left=False)
    for i, grade in enumerate(GRADES):
        item = recall[str(grade)]
        low, high = item["wilson_95_ci"]
        value = item["recall"]
        axes[1].errorbar(value, i, xerr=[[value-low], [high-value]], fmt="o", color="#145b79", capsize=5, markersize=8)
        axes[1].text(1.02, i, f"{item['correct']}/{item['patients']}", va="center", transform=axes[1].get_yaxis_transform())
    axes[1].set(yticks=range(3), yticklabels=[f"Grade {g}" for g in GRADES], xlim=(0, 1), ylim=(2.5, -0.5),
                xlabel="Recall (95% Wilson interval)", title="Per-grade sensitivity")
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Automatic MRI segmentation-to-grade pipeline | held-out test, n = 75", fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.89, f"Macro-F1 {point['macro_f1']:.3f}  |  Balanced accuracy {point['balanced_accuracy']:.1%}  |  Accuracy {point['accuracy']:.1%}", ha="center")
    fig.text(0.5, 0.035, "Matrix percentages are normalized within each true grade. Small grade-2/3 groups produce wide intervals.", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.78, bottom=0.19, left=0.08, right=0.92, wspace=0.48)
    fig.savefig(figures / "grade-test-confusion-recall.svg", bbox_inches="tight")
    fig.savefig(figures / "grade-test-confusion-recall.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
