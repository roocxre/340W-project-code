# Analysis: split, EDA and load classifiers

The joined n-back table from [synthesis/](../synthesis/) (Shin 2018 + COG-BCI v4, 26,137 windows × 282 columns), split 70 / 20 / 10 by subject. It is then explored and used to compare classifiers of n-back load with the baselines from Yedukondalu et al. 2025 and EEGNet. How every technique tried here and upstream compares with the source papers' methods is in [../technique_comparison.md](../technique_comparison.md).

**Results in brief** (test balanced accuracy on held-out subjects, [details](#results)):

| Task (chance) | Yedukondalu ensemble, corrected CV | EEGNet | Best candidate (chosen on validation) |
|---|---|---|---|
| 0 vs 2-back, pooled (50 %) | 61.8 | 69.1 | 71.4, gated soft vote |
| Shin 0 / 2 / 3-back (33.3 %) | 38.6 | 40.8 | 48.4, random forest |
| COG-BCI 0 / 1 / 2-back (33.3 %) | 42.1 | 50.1 | 50.5, XGBoost |

- **RQ1.** Load level can be decoded from new subjects' EEG above chance: kappa 23–26 for three levels, 43 for 0 vs 2.
- **Every candidate beats the corrected Yedukondalu pipeline.** The margins are 3–12 points, and 15 of the 18 paired intervals exclude zero.
- **No candidate clearly beats EEGNet or logistic regression on band power, except on Shin.** Elsewhere they are within 3 points.
- **RQ2.** The chance-corrected score for three levels is 53–60 % of the score for presence of load.
- **The paper's leaky protocol inflates its estimate by 15–37 points** while barely changing the model it picks.

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

## Modelling

### Tasks

Clean windows only (`artifact == False`, 95.6 %). Every task uses the same subject split.

| Task | Data | Classes | Train / val / test windows | Test subjects |
|---|---|---|---|---|
| `bin` | both datasets | 0 vs 2-back (presence of load) | 11,488 / 1,733 / 3,445 | 11 |
| `shin3` | Shin 2018 | 0 / 2 / 3-back | 4,562 / 802 / 1,356 | 5 |
| `cog3` | COG-BCI | 0 / 1 / 2-back | 12,673 / 1,804 / 3,798 | 6 |

Classes are nearly balanced: in every task and split the largest class is at most 5 % bigger than the smallest ([results/eda_class_balance.csv](results/eda_class_balance.csv)).

### Models

| Model | Group | Input | Normalization | Tuning |
|---|---|---|---|---|
| Yedukondalu ensemble, leaky CV | baseline | R-LMD, 5 modes × 6 features × 26 channels = 780 | z-score over all subjects pooled (as published) | BAO feature selection (30 agents, 100 iterations, 5-fold KNN error) and 30 TPE trials over bagging / AdaBoost / boosting, both on **window-level** folds |
| Yedukondalu ensemble, corrected CV | baseline | same | same | same, on **subject-wise** folds |
| Logistic regression | baseline | band power, 260 | z-score within subject | C, subject-wise 5-fold |
| EEGNet-8,2 | baseline | raw EEG, 25 × 400 (sum-zero projection, 100 Hz) | Euclidean alignment per subject | early stopping on val loss; 3 seeds averaged |
| RBF SVM, MLP | candidate | band power | z-score within subject | grid, subject-wise 5-fold, balanced accuracy |
| Random forest, XGBoost | candidate | band power + R-LMD, 1,040 | z-score within subject | grid, subject-wise 5-fold |
| Riemannian tangent space | candidate | theta / alpha / beta covariances (OAS), 900 | each subject re-centred to its own Riemannian mean | C, subject-wise 5-fold |
| Gated soft vote | candidate | mean probability of the candidates that pass the gate | — | gate decided on validation |

- **The Yedukondalu baselines run the eeg-load reproduction unchanged.** Feature extraction, BAO and the six optimizable classifiers all come from [../eeg-load/loadrepro/](../eeg-load/loadrepro/). The ensemble is the paper's proposed model. The other five families (tree, discriminant, naive Bayes, KNN, SVM) are in the leakage table.
- **Each normalization uses only that subject's own unlabelled windows.** Each subject is in exactly one split, so none of them leaks.
- **The vote gate is decided on validation only.** A candidate joins the vote when it clears all three of:
  - every class's recall above 1/k;
  - every class's precision above that class's prevalence;
  - the lower 95 % bound of its AUC, bootstrapped over validation subjects, above 50.

  On `cog3` the gate excluded random forest (0-back recall 30.6 % on validation) and the Riemannian model ([results/eval_vote_gate.csv](results/eval_vote_gate.csv)).
- **The best candidate is the one with the best validation balanced accuracy.** The test split is scored once per model and never used for a choice.

## Results

### Held-out test subjects

Balanced accuracy in % with its 95 % bootstrap interval over test subjects, then AUC (macro one-vs-rest for three classes). Bold marks the candidate picked on validation ([results/eval_summary.csv](results/eval_summary.csv), [figure](results/figures/eval_summary.png)).

| Model | Group | `bin` bacc | AUC | `shin3` bacc | AUC | `cog3` bacc | AUC |
|---|---|---|---|---|---|---|---|
| Yedukondalu ensemble, leaky CV | baseline | 60.5 (52–72) | 66.3 | 39.4 (36–43) | 57.7 | 48.6 (33–71) | 68.6 |
| Yedukondalu ensemble, corrected CV | baseline | 61.8 (53–73) | 69.0 | 38.6 (37–41) | 56.1 | 42.1 (34–53) | 58.4 |
| Logistic regression (band power) | baseline | 68.8 (58–77) | 75.8 | 48.7 (46–52) | 68.0 | 48.7 (39–60) | 67.0 |
| EEGNet (raw EEG) | baseline | 69.1 (60–77) | 75.8 | 40.8 (35–49) | 62.8 | 50.1 (41–60) | 67.6 |
| RBF SVM | candidate | 69.6 (60–78) | 76.6 | 49.9 (47–55) | 67.9 | 49.2 (37–61) | 66.1 |
| MLP | candidate | 65.3 (56–76) | 71.0 | 45.2 (43–48) | 63.9 | 48.3 (34–69) | 66.7 |
| Random forest | candidate | 70.6 (59–82) | 77.5 | **48.4 (44–54)** | 68.5 | 51.5 (36–71) | 71.5 |
| XGBoost | candidate | 69.4 (59–79) | 77.4 | 49.9 (48–53) | 68.4 | **50.5 (36–66)** | 69.3 |
| Riemannian tangent space | candidate | 70.2 (62–78) | 77.2 | 46.6 (43–51) | 65.1 | 46.8 (38–59) | 64.0 |
| Gated soft vote | candidate | **71.4 (60–82)** | 79.0 | 50.7 (48–54) | 69.8 | 50.6 (35–70) | 68.8 |

Confusion matrices: [bin](results/figures/eval_confusion_bin.png), [shin3](results/figures/eval_confusion_shin3.png), [cog3](results/figures/eval_confusion_cog3.png). ROC curves: [eval_roc.png](results/figures/eval_roc.png). Per-class precision and recall: [results/eval_per_class.csv](results/eval_per_class.csv).

### Paired differences

Test balanced accuracy of each model minus the baseline, in points, with a 95 % paired bootstrap interval over test subjects. Bold marks an interval that excludes zero.

| Model | vs Yedukondalu corrected: `bin` | `shin3` | `cog3` | vs EEGNet: `bin` | `shin3` | `cog3` |
|---|---|---|---|---|---|---|
| RBF SVM | **+7.8 (+0.6 to +15.6)** | **+11.3 (+6.5 to +16.7)** | **+7.2 (+0.6 to +14.1)** | +0.5 (−2.1 to +3.7) | **+9.1 (+5.0 to +12.8)** | −0.8 (−4.9 to +2.4) |
| MLP | +3.5 (−1.2 to +9.1) | **+6.6 (+4.5 to +8.9)** | +6.2 (−1.6 to +16.7) | −3.8 (−8.6 to +1.8) | +4.4 (−2.2 to +10.5) | −1.8 (−10.7 to +11.2) |
| Random forest | **+8.8 (+2.1 to +15.7)** | **+9.8 (+3.8 to +15.9)** | **+9.4 (+0.7 to +18.6)** | +1.5 (−3.0 to +6.4) | **+7.6 (+3.8 to +11.2)** | +1.4 (−5.6 to +11.8) |
| XGBoost | **+7.5 (+1.0 to +14.3)** | **+11.2 (+7.3 to +15.2)** | **+8.4 (+0.1 to +16.3)** | +0.3 (−3.1 to +3.8) | **+9.1 (+3.5 to +13.7)** | +0.4 (−6.4 to +8.0) |
| Riemannian tangent space | **+8.4 (+2.5 to +14.6)** | **+8.0 (+3.5 to +12.5)** | **+4.7 (+2.6 to +7.2)** | +1.1 (−1.2 to +4.4) | **+5.8 (+1.0 to +11.5)** | −3.3 (−8.0 to +1.3) |
| Gated soft vote | **+9.6 (+3.3 to +16.6)** | **+12.1 (+9.1 to +15.9)** | +8.5 (−0.6 to +18.7) | +2.3 (−1.5 to +6.8) | **+9.9 (+4.1 to +15.1)** | +0.5 (−7.4 to +12.0) |
| Logistic regression | +7.0 (−0.5 to +15.3) | **+10.1 (+6.2 to +14.5)** | **+6.7 (+1.9 to +12.5)** | −0.3 (−3.5 to +3.5) | **+7.9 (+2.0 to +13.0)** | −1.4 (−3.2 to +0.7) |
| EEGNet | **+7.3 (+1.0 to +13.8)** | +2.2 (−5.2 to +10.6) | **+8.0 (+3.3 to +14.3)** | — | — | — |

- **The Yedukondalu pipeline is the weakest model on every task.** Even plain logistic regression on band power beats it by 7–10 points. Two things in the pipeline cost it accuracy on new subjects:
  - it z-scores across subjects pooled, so subject differences are left in the features;
  - BAO keeps only 27–78 of the 780 R-LMD features, chosen by a KNN on pooled-normalized data. Even on subject folds, that search finds little that carries over to new subjects.
- **The "more robust" models do not clearly beat the simple ones.** Random forest, XGBoost, the Riemannian model and the vote all land within 3 points of logistic regression and EEGNet on `bin` and `cog3`, and every paired interval against EEGNet there includes zero. With 6–11 test subjects, a difference under about 5 points cannot be resolved.
- **Shin is the exception, and the reason is EEGNet's weakness there.** EEGNet has only 18 training subjects on `shin3`, and every feature model beats it by 4–10 points (the interval excludes zero for 6 of 7).
- **The vote helps a little and never hurts.** It has the highest AUC on `bin` (79.0) and `shin3` (69.8), but its accuracy gain over its best member is under 1 point.
- **The 20 s causal smoothing adds 2–3 points on average** (−2 to +5 by model). It averages the probability over the window and the 4 before it in the same block (`smooth20_bacc` in the summary CSV). Load is sustained over a block, so a few windows of evidence help.

### Where the errors are

Per-class recall on test ([results/eval_per_class.csv](results/eval_per_class.csv)):

- **The level that differs most in difficulty is recognised best.**
  - On `cog3`, 2-back recall is 54–68 % for the candidates, but 0-back and 1-back are confused with each other (40–48 %).
  - On `shin3` it is the reverse: 0-back recall is 52–62 %, and 2-back and 3-back are confused (40–46 %).
- **The confusions follow behaviour.** COG-BCI's 0 and 1-back are equally easy (accuracy 0.99 and 0.99, RT 0.41 and 0.47 s). Shin's 2 and 3-back are both demanding (0.91 and 0.82, RT 0.78 and 0.85 s) ([results/eda_behavior.csv](results/eda_behavior.csv)).
- **The models read "easy vs demanding" more than the exact n.** That is the boundary the EEG changes in the EDA mark.

### Leakage in the published protocol

Each Yedukondalu family's development CV estimate against the same family on held-out test subjects ([results/eval_leakage.csv](results/eval_leakage.csv), [figure](results/figures/eval_leakage.png)):

| Ensemble | Window-level CV (leaky) | Same settings, subject-wise | Test, leaky | Test, corrected |
|---|---|---|---|---|
| `bin` | 88.2 | 55.6 | 60.6 | 61.9 |
| `shin3` | 54.2 | 40.9 | 39.4 | 38.6 |
| `cog3` | 85.4 | 37.0 | 48.6 | 42.2 |

These are accuracies in %; the subject-wise and test columns are almost identical to balanced accuracy, since the classes are balanced.

- **Window-level folds overstate the ensemble's accuracy by 15–37 points.** KNN (12–33) and the ensemble are hit hardest, because they can match a test window to its neighbours in the same block. Naive Bayes and the discriminant classifiers, which cannot memorize, are inflated by 1–8 points.
- **On held-out subjects, leaky and corrected land within about 1 point of each other on `bin` and `shin3`.** The leak corrupts the estimate more than the model.
  - The exception is the `cog3` ensemble, where the leaky search picked a large AdaBoost model that happened to test 6 points higher.
  - Its interval (33–71) spans most of the range, and validation (37.7) did not foresee it.

### RQ2: levels vs presence of load

Chance-corrected balanced accuracy, (bacc − chance) / (100 − chance), equals Cohen's kappa for balanced classes ([results/eval_rq2.csv](results/eval_rq2.csv)):

| Task | Classes | Best candidate | Kappa | AUC | EEGNet kappa | Yedukondalu corrected kappa |
|---|---|---|---|---|---|---|
| 0 vs 2-back | 2 | gated soft vote | 42.8 | 79.0 | 38.2 | 23.7 |
| Shin 0 / 2 / 3 | 3 | random forest | 22.6 | 68.5 | 11.2 | 8.0 |
| COG-BCI 0 / 1 / 2 | 3 | XGBoost | 25.8 | 69.3 | 25.2 | 13.2 |

Telling levels apart reaches 53–60 % of the chance-corrected performance for telling load from no load, for the best model. It loses about 10 points of AUC.

### EDA

The EDA covers train subjects only: [scripts/03_eda.py](scripts/03_eda.py), [results/log_03_eda.txt](results/log_03_eda.txt), and the `eda_*` figures.

- **Behaviour.** Accuracy falls and RT rises with n. COG-BCI goes from 0.99 to 0.89 accuracy and from 0.41 to 0.59 s. Shin goes from 0.99 to 0.82 and from 0.37 to 0.85 s.
- **Spectra** ([eda_psd.png](results/figures/eda_psd.png), [eda_topomap.png](results/figures/eda_topomap.png), [eda_roi.png](results/figures/eda_roi.png)). Both datasets show the classic working-memory pattern:
  - frontal theta rises (only 3–4 subjects per dataset go the other way);
  - parieto-occipital alpha falls in 16 of 18 and 18 of 20 subjects;
  - central beta falls in 17 of 18 and 20 of 20 subjects.

  The Friedman test has p < 0.003 for all six ROI × dataset tests. Alpha decreases at nearly every electrode after Bonferroni correction.
- **Which features separate the levels** ([eda_anova.png](results/figures/eda_anova.png), [results/eda_anova.csv](results/eda_anova.csv)).
  - The median ANOVA F across levels is 32–128 for band power and 1.8–2.7 for R-LMD.
  - Only R-LMD's first mode carries signal. Its PSD, Rényi and differential entropy at CP, P and POz channels reach the F of the best alpha and beta powers at the same channels.
  - **R-LMD is not redundant overall, but its load signal mostly is.** Band power predicts 19–21 % of all R-LMD variance (ridge regression, [results/eda_rlmd_redundancy.csv](results/eda_rlmd_redundancy.csv)): 49–50 % of mode 1, and 3–7 % of modes 4–5. The part band power cannot predict sits in the higher modes, which carry no load signal (F ≈ 1–8). That fits random forest and XGBoost on band power + R-LMD gaining 1–2 points at most.
- **PCA** ([eda_pca.png](results/figures/eda_pca.png), [results/eda_pca.csv](results/eda_pca.csv)).
  - **Across subjects, the leading components are mostly identity.** With band power z-scored across all subjects (as the Yedukondalu pipeline does), subject explains 64–79 % of the variance of PC1–3 in every task, and load level at most 2.6 %.
  - **Within each subject, PC1 becomes the most load-related direction.** After z-scoring each subject on its own, subject identity drops to 0 and level explains 18 % of PC1's variance on `shin3`, 11 % on `bin` and 7 % on `cog3`. That PC1 is whole-scalp alpha power: absolute and relative alpha load most strongly, evenly over the scalp. It is lower at higher n.
  - **This is why within-subject normalization matters.** Normalizing turns the largest axis of variation from "who" into "how loaded".
  - **PCA does not help as preprocessing.** Logistic regression on the top 100, 50, 20 or 10 components scores 70.9, 70.8, 69.0 and 68.2 % in subject-wise CV on `bin`, against 71.1 % on all 260 features ([results/eda_pca_preprocessing.csv](results/eda_pca_preprocessing.csv)). Only PC1 carries much load signal, so cutting components drops information without removing noise the regularized models do not already handle. The models therefore use no PCA.

### Caveats

- **The test set is small.** It has 11 subjects for `bin` and 5–6 for the ternary tasks, so the intervals are wide. The `shin3` intervals look narrow only because 5 subjects is too few for the bootstrap to see the between-subject spread; treat them as optimistic.
- **The validation set has 6 subjects, so it is a noisy guide.** On `cog3` it excluded random forest from the vote, though random forest scored highest on test.
- **The Yedukondalu baselines use 4 s windows from these datasets, not the paper's MAT / STEW recordings.** They measure how the published pipeline does on this task. They are not a reproduction of the paper's numbers, which is in [../eeg-load/](../eeg-load/).

## Workflow

From a fresh clone, `python run_pipeline.py` at the repository root runs every stage below after the eeg-load and synthesis stages they depend on ([root README](../README.md)). It writes the same log files. It uses one environment for the whole repository: `.venv/` at the root, with Python 3.12 and the pinned [../requirements.txt](../requirements.txt). [requirements.txt](requirements.txt) here lists the packages this project adds to synthesis's. To run stages by hand, from `analysis/`:

```bash
../.venv/bin/python scripts/01_split.py > results/log_01_split.txt   # ~1.5 min
../.venv/bin/python -m pytest -q tests
../.venv/bin/python scripts/02_rlmd_features.py > results/log_02_rlmd_features.txt
../.venv/bin/python scripts/05_models.py --models lr riemann svm mlp > results/log_05_models_part1.txt
../.venv/bin/python scripts/05_models.py --models rf xgb --jobs 4 > results/log_05_models_part2.txt
../.venv/bin/python scripts/06_eegnet.py > results/log_06_eegnet.txt
../.venv/bin/python scripts/03_eda.py > results/log_03_eda.txt
../.venv/bin/python scripts/04_baselines.py --task bin --protocol corrected   # and shin3 / cog3, then leaky
../.venv/bin/python scripts/07_evaluate.py > results/log_07_evaluate.txt
```

`scripts/run_rest.sh` (zsh, macOS) is the earlier driver for the stages after R-LMD; `run_pipeline.py` replaces it.

| Stage | Script | What it does | Time on an M1 (8 cores) |
|---|---|---|---|
| 1 | `01_split.py` | subject split, technique comparison | 1.5 min |
| 2 | `02_rlmd_features.py` | R-LMD features per window, cached per subject in `cache/rlmd/` | ~4 h |
| 3 | `03_eda.py` | behaviour, spectra, topomaps, ROI tests, ANOVA, R-LMD redundancy, PCA (train only) | 15 s |
| 4 | `04_baselines.py --task T --protocol {leaky,corrected}` | Yedukondalu pipeline: BAO, then six optimizable classifiers | 0.5–1.7 h per run, ~7 h for all six |
| 5 | `05_models.py` | logistic regression, SVM, MLP, random forest, XGBoost, Riemannian | ~1.5 h alone, much longer while sharing the CPU |
| 6 | `06_eegnet.py` | EEGNet, 3 seeds per task, on MPS | ~30 min alone |
| 7 | `07_evaluate.py` | vote gate, test metrics, paired bootstrap, figures, CSVs | 1 min |

- **Every stage caches its finished pieces in `cache/`, so an interrupted run resumes.** The caches are git-ignored. Predictions are saved in `results/predictions/<task>__<model>.npz`.
- **Pause and resume the heavy stages with `kill -STOP` and `kill -CONT`** on the process and its workers.

The split is deterministic for a given source CSV. `split_manifest.json` records the source's checksum, so a regenerated `nback_features.csv` that changed can be detected. The CSVs total 65 MB (train 44 MB, under GitHub's 50 MB warning size).
