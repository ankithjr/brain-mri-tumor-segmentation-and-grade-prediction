# Held-out grade evaluation and uncertainty

The frozen automatic pipeline was evaluated on 75 patient-disjoint initial exams: 9 grade 2, 6 grade 3, and 60 grade 4. Segmentation predictions define the crop at inference; reference masks were used to prepare classifier training crops. The grader is the epoch-17 checkpoint with SHA-256 b56365465a3c883aa1fa22c0bf73ae0685ea30025b4867bfabfcc5d63466b806.

All uploaded prediction IDs and grades were checked against the archived test split. Every probability vector is finite, sums to one within tolerance, and has an argmax matching the reported prediction. Recomputed metrics and confusion counts exactly match the uploaded summary.

| Metric | Estimate | 95% stratified bootstrap interval |
| --- | ---: | ---: |
| Macro-F1 | 0.6086 | 0.4593–0.7522 |
| Balanced accuracy | 0.6407 | 0.4685–0.8130 |
| Accuracy | 0.8000 | 0.7067–0.8800 |

The percentile bootstrap resamples patients with replacement within each true grade, preserving the observed 9/6/60 grade counts. There are 10,000 replicates with seed 42. These intervals describe fixed-model sampling uncertainty within the observed cohort composition. They do not include training variability, repeated validation-based model selection, or performance at another institution.

Per-grade recall uses 95% Wilson score intervals:

| Grade | Correct / total | Recall | 95% Wilson interval |
| --- | ---: | ---: | ---: |
| 2 | 5/9 | 0.5556 | 0.2667–0.8112 |
| 3 | 3/6 | 0.5000 | 0.1876–0.8124 |
| 4 | 52/60 | 0.8667 | 0.7583–0.9309 |

An always-grade-4 classifier has accuracy 0.8000, macro-F1 0.2963, and balanced accuracy 0.3333 on this test set. Accuracy alone hides minority-grade errors. No significance claim is made from these intervals or from the point-estimate comparison.

The test set has now been evaluated. Its results must not drive additional model tuning; future development should use development data and identify this test cohort as previously observed.

## Reproduce

From the repository root with NumPy and Matplotlib installed:

```bash
python scripts/analyze_grade_test.py --predictions results/grade-test-predictions.csv --summary results/grade-test-summary.json --test-split results/splits/test.csv --output results --replicates 10000 --seed 42
```

The script validates the inputs, writes the uncertainty JSON, and produces SVG and PNG figures. The vector figure is tracked in GitHub; the PNG can be regenerated.

![Test confusion matrix and per-grade recall intervals](../results/figures/grade-test-confusion-recall.svg)
