# Train / Test / Validation Split

The joined n-back table from [synthesis/](../synthesis/) (Shin 2018 + COG-BCI v4, 26,137 windows × 282 columns), split 70 / 20 / 10 by subject for downstream models. How every technique tried here and upstream compares with the source papers' methods is in [../technique_comparison.md](../technique_comparison.md).

## The split

Whole subjects, stratified by dataset and by whether Cz was interpolated (COG-BCI subjects 1–9), seed 0 ([scripts/01_split.py](scripts/01_split.py), [results/log_01_split.txt](results/log_01_split.txt)).

| | Train | Test | Validation | Total |
|---|---|---|---|---|
| Subjects | 38 (Shin 18, COG-BCI 20) | 11 (5, 6) | 6 (3, 3) | 55 |
| Windows | 17,956 (68.7 %) | 5,427 (20.8 %) | 2,754 (10.5 %) | 26,137 |
| Shin share of windows | 27.2 % | 25.4 % | 29.4 % | 27.1 % |
| Windows at n = 0 / 1 / 2 / 3 | 5,986 / 4,356 / 5,985 / 1,629 | 1,809 / 1,350 / 1,809 / 459 | 918 / 648 / 918 / 270 | |
| Artifact windows | 4.0 % | 5.0 % | 5.4 % | 4.4 % |

| Split | Subjects |
|---|---|
| train | cog-03 04 05 06 07 08 10 12 13 14 15 16 17 18 20 21 22 23 26 29; shin-VP002 003 004 005 008 009 010 011 012 013 015 016 019 021 022 024 025 026 |
| test | cog-01 02 19 24 27 28; shin-VP001 007 014 020 023 |
| val | cog-09 11 25; shin-VP006 017 018 |

- **No subject is in more than one split.** Every split has both datasets and all four levels.
- **The shares are 68.7 / 20.8 / 10.5 rather than exactly 70 / 20 / 10**, because whole subjects are allocated and COG-BCI subjects have 2.3× the windows of Shin's.
- **The validation set is small.** Six subjects means a validation accuracy has a standard error of roughly 3–4 points. Use it for model selection and early stopping, and report the test set once.
- **Both weak subjects are in train.** shin-VP004 and cog-13 have > 20 % artifact windows. Two of the four COG-BCI subjects with a 2.5 s SOA in some files (cog-01, 02) are in test.

## Files

| File | Contents |
|---|---|
| `data/train.csv`, `data/test.csv`, `data/val.csv` | rows of `nback_features.csv`, all 282 columns unchanged, artifact windows kept ([column dictionary](../synthesis/README.md#columns)) |
| `data/split_subjects.csv` | one row per subject: dataset, Cz interpolation, split, windows, levels |
| `data/split_manifest.json` | technique, strata, seed, subjects per split, and the SHA-256 of the source CSV |
| `results/split_comparison.csv` | every technique × seed scored |
| [nbacksplit.py](nbacksplit.py) | `load(split, clean=False)`, `feature_columns`, `zscore_within_subject`, `windows` (the raw EEG arrays for signal models) |

```python
import sys; sys.path.insert(0, "analysis")
from nbacksplit import load, feature_columns, zscore_within_subject, windows

tr = zscore_within_subject(load("train", clean=True))
X, y = tr[feature_columns(tr)].to_numpy(), tr.n.to_numpy()
x_raw = windows(tr)          # (n, 26, 800) float32 µV; needs synthesis/cache/windows/
```

Within-subject z-scoring needs no labels and no other subject, and each subject sits in one split, so normalizing each split on its own does not leak. It is what removes dataset identity from the features (91 % → majority rate, [synthesis/README.md](../synthesis/README.md#viability-for-ml)).

## Techniques compared

Each technique cuts its units 70 / 20 / 10 by largest remainder within each stratum, over 20 seeds. The probe is the synthesis baseline: 0- vs 2-back logistic regression, clean windows, fitted on train and scored on test.

| Technique | Subjects in 2+ splits | Mean shares | Worst share error | Worst Shin-mix error | Probe, z-scored | Probe, raw |
|---|---|---|---|---|---|---|
| A. Windows, stratified by dataset × n | 55 of 55 | 70.0 / 20.0 / 10.0 | 0.0 | 0.0 | 75.1 % (73.7–76.6) | 71.9 % (71.0–72.9) |
| B. Blocks, stratified by dataset | 55 | 70.0 / 20.1 / 10.0 | 0.1 | 0.3 | 72.3 % (70.8–73.9) | 67.0 % (63.4–70.7) |
| C. Sessions, stratified by dataset | 36.6 | 69.9 / 19.8 / 10.3 | 0.6 | 1.5 | 71.2 % (66.6–76.1) | 66.3 % (62.9–69.8) |
| D. Subjects, unstratified | 0 | 71.4 / 19.7 / 8.8 | 5.0 | 36.0 | 68.4 % (62.4–76.1) | 63.3 % (58.9–71.9) |
| **E. Subjects, stratified by dataset × Cz (chosen)** | **0** | **69.0 / 20.3 / 10.7** | **1.5** | **3.0** | **70.3 % (66.9–74.5)** | **65.6 % (60.8–72.0)** |

Errors are in percentage points, worst case over the 20 seeds. Probe ranges are min–max.

- **Splitting below the subject leaks.** A, B and C put every subject (C: two thirds of them) in more than one split. Their probe scores run 1–5 points above the subject-level splits because the model has seen other windows from the test subjects. A is the leakiest: neighbouring 4 s windows of the same block land on both sides.
- **An unstratified subject split can lose a dataset.** D's worst seed puts the Shin share of one split 36 points off the pool's 27 %, and its smallest split–level cell drops to 90 windows. Its probe varies over 14 points between seeds.
- **E keeps subjects whole and the mix fixed.** Stratifying by dataset holds every split at 25–29 % Shin. Stratifying COG-BCI by Cz interpolation spreads the 9 subjects without a recorded Cz 6 / 2 / 1. On seed 0 the probe scores 68.2 % on test, against 69.7 % for subject-wise 10-fold CV in synthesis.
- **Leave-one-dataset-out** (train Shin, test COG-BCI and the reverse, 65.3 / 63.9 %) remains the cross-dataset check. It cannot give 70 / 20 / 10, so it complements the split rather than replacing it.

## Workflow

Uses the synthesis environment ([../synthesis/requirements.txt](../synthesis/requirements.txt)):

```bash
cd analysis
../synthesis/.venv/bin/python -m pytest -q tests
../synthesis/.venv/bin/python scripts/01_split.py > results/log_01_split.txt   # ~1.5 min
```

The split is deterministic for a given source CSV. `split_manifest.json` records the source's checksum, so a regenerated `nback_features.csv` that changed can be detected. The CSVs total 65 MB (train 44 MB, under GitHub's 50 MB warning size).
