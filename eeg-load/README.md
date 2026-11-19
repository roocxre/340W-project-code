# EEG Cognitive Load Reproduction

Reproduction, in Python, of Yedukondalu et al. 2025, "Cognitive load detection through EEG lead wise feature optimization and ensemble classification", *Scientific Reports* 15:842 ([docs/s41598-024-84429-6.pdf](docs/s41598-024-84429-6.pdf)). The paper's analysis was done in MATLAB R2022a and no code was released.

The results table is in [results/paper_vs_ours.md](results/paper_vs_ours.md), and the figures are in [results/figures/](results/figures/).

## Results

Accuracy in %, 10-fold cross-validation over segments as in the paper. "Subject-wise" re-scores the same tuned model with no subject in both training and test folds.

| Target | Paper | Ours | Ours, subject-wise |
|---|---|---|---|
| MAT overall, ensemble (Table 1) | 97.4 | 88.3 | 68.7 |
| MAT overall, best classifier | 97.4 (ensemble) | 91.6 (KNN) | 64.4 |
| STEW overall, ensemble (Table 2) | 96.1 | 95.0 | 75.4 |
| STEW overall, best classifier | 96.1 (ensemble) | 97.0 (KNN) | 70.8 |
| MAT best single lead (Table 3) | 94.5 (F3) | 77.2 (Fp1) | 71.6 |
| STEW best single lead (Table 4) | 94.0 (F3) | 77.9 (F8) | 72.8 |

- **STEW overall reproduces.** KNN, SVM and the ensemble land at 95-97 %, against the paper's 94.7-96.1 %.
- **MAT overall is 6-9 points short.** KNN reaches 91.6 % and the ensemble 88.3 %, against 96.2 % and 97.4 %.
- **Tree, discriminant and naive Bayes do not reproduce on either dataset.** They reach 55-84 % here against 90-95 % in the paper.
- **Lead-wise results do not reproduce.** Single leads give 59-78 % here against 64-94 % in the paper, F3 is not the best lead on either dataset, and our lead ranking does not correlate with the paper's.
- **The tuning sweep (Fig. 9) does not reproduce.** Accuracy here is flat in the number of BAO iterations and does not peak at L = 5.
- **The high accuracies come from recognising recordings, not cognitive load.** With subjects kept apart, every model drops to 53-76 %.

### Why the high numbers are recording recognition

Every subject contributes 15 (MAT) or 37 (STEW) consecutive segments per condition, and segment-level cross-validation puts neighbours of each test segment in the training set. A classifier then only has to recognise which recording a segment came from. A plain baseline shows the size of the effect on MAT: log band power in 5 bands per lead with 1-nearest-neighbour scores 94 % under segment-level CV and 66 % subject-wise.

### Why lead-wise is out of reach

One lead of a 4 s segment does not carry enough to identify the recording. The full 1-45 Hz log spectrum of a single lead, which contains more than any six-feature summary of its modes, gives 63-80 % under segment-level CV (F3: 67 % on MAT, 70 % on STEW). The paper's lead-wise tables are also nearly identical across the two datasets at shared electrode positions, despite different subjects, hardware and tasks: F3 94.5 / 94.0, F4 92.9 / 93.4, O1 90.5 / 90.5, F8 90.0 / 89.5, and T6 / P8 at 88.1 / 87.2 with the same unusual 99 % sensitivity.

## Methods

### Data facts

- **MAT** (PhysioNet `eegmat`): 36 subjects, 19 EEG leads, 500 Hz. The EDF files are already filtered by the dataset authors (0.5-45 Hz, 50 Hz notch, ICA), which is the preprocessing the paper cites, so nothing more is applied.
- **MAT segments:** the paper's confusion matrix has 540 segments per class, which is 15 non-overlapping 4 s segments per subject and condition. The task recording is 62 s; the rest recordings are 80-188 s, and the last 60 s is used.
- **STEW** (IEEE DataPort): 48 subjects, 14 Emotiv leads, 128 Hz, 150 s per condition, giving 37 segments each (3552 in total, as in the paper). The files are raw, with a DC offset near 4200 and artifacts of several hundred µV.
- **STEW preprocessing** follows the paper: 1 Hz high-pass, 50 Hz notch, artifact subspace reconstruction (`asrpy`, cutoff 20), average reference.

### Pipeline

| Stage | Script | What it does |
|---|---|---|
| 1 | `01_features.py` | 4 s segments, R-LMD per lead into 5 modes (4 product functions and the residual), 6 features per mode: skewness, kurtosis, PSD, fuzzy, Renyi and differential entropy. 570 features per MAT segment, 420 per STEW segment. |
| 2 | `02_overall.py` | BAO feature selection, then the six optimized classifiers under 10-fold CV. Tables 1 and 2. |
| 3 | `03_leadwise.py` | BAO and the optimized ensemble on each lead's 30 features. Tables 3 and 4. |
| 4 | `04_tuning.py` | MAT accuracy against decomposition level (3, 5, 7, 9) and BAO iterations (25-100). Fig. 9. |
| 5 | `05_figures.py` | Figures 2-9 in the paper's layout. |
| 6 | `06_report.py` | Paper-vs-ours tables and the ANOVA statistics. |

### What was ported and what was rebuilt

- **R-LMD** is ported line by line from the original authors' MATLAB code, which is copied under [reference/rlmd/](reference/rlmd/) ([reference/SOURCES.txt](reference/SOURCES.txt)). The paper names no repository; this is the implementation its R-LMD reference points to. MATLAB's `smooth` and the automatic binning of `histcounts` are re-implemented from their documented behaviour. The decomposition of a sample segment matches the paper's Fig. 2 in structure and amplitude.
- **Optimized classifiers** rebuild MATLAB Classification Learner's six "optimizable" models on scikit-learn, with the app's search spaces, 30 tuning iterations and the 10-fold CV error as objective. Optuna's TPE sampler stands in for MATLAB's Bayesian optimization, and gradient boosting stands in for GentleBoost and LogitBoost.
- **BAO** is written from the paper's equations.

### Choices the paper does not specify

| Choice | Here | Basis |
|---|---|---|
| Which 60 s of MAT rest | last 60 s | arbitrary; first, middle and last give the same baseline accuracy |
| PSD as one number | mean of the Welch PSD (`pwelch` defaults) | guess |
| Fuzzy entropy | m = 2, r = 0.2 SD, n = 2 | usual defaults |
| Renyi entropy | quadratic, Gaussian kernel, Silverman bandwidth | paper's Eq. 9 |
| Feature scaling | z-score per feature | guess |
| BAO population | 30 agents | reverse-engineered |
| BAO wrapper | KNN, k = 5, weight 0.99 on the error | steps of the paper's Fig. 3 |
| BAO error estimate | 5-fold CV | reverse-engineered |
| BAO positions | binary, redrawn through the S-shaped transfer function | reverse-engineered |

### Reverse-engineering notes

The first version used the literal reading: 10 agents, continuous positions, and KNN error on a 20 % hold-out, which is what the step size of the paper's Fig. 3a implies. It gave 82.0 % (MAT) and 92.9 % (STEW) for the ensemble. The versions tried from there:

- **BAO search.** The informative features are the fuzzy, Renyi and differential entropy of modes 1 and 2; the rest mostly add noise for KNN. The literal BAO kept 81 of 570 features and missed them. Binary positions, 30 agents and a cross-validated error find them (25 features kept) and give the numbers above. Hand-picking "modes 1-2, no skewness or kurtosis" gives the same level, so the selection is no longer the limit.
- **Starting fitness.** The paper's BAO starts at a fitness of 0.055, meaning KNN is already 95 % accurate on a random half of its features. Ours starts at 0.31 (MAT) and 0.18 (STEW). No definition of the six features that I tried in calibrated µV comes near 0.055 on MAT, so the paper's feature values differ from ours in a way its text does not explain.
- **Uncalibrated units.** The paper's data link is PhysioNet's ATM export tool, whose `.mat` files hold raw digital values. The MAT EDF files scale every lead of every recording to its own min-max, so raw values carry a per-recording gain and offset. The segment mean and log SD of the raw values alone give 99.5 % with 1-NN on all leads. Pushed through R-LMD and the six features, raw units give 90-93 % overall and 72-80 % per lead, so this does not explain the paper either, and it was not adopted.
- **R-LMD sensitivity.** Another Python port of the same MATLAB code (PySDKit) departs from the MATLAB source in several places and produces visibly different modes from ours on the same segments. Mode features are sensitive to such details; without MATLAB the port cannot be checked numerically against the original.

## Workflow

Python 3.12 with the packages in [requirements.txt](requirements.txt):

```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests
.venv/bin/python scripts/01_features.py     # about 25 min on 8 cores
.venv/bin/python scripts/02_overall.py
.venv/bin/python scripts/03_leadwise.py
.venv/bin/python scripts/04_tuning.py
.venv/bin/python scripts/05_figures.py
.venv/bin/python scripts/06_report.py
```

Datasets go in `data/eegmat/` and `data/STEW Dataset/`; features are cached in `cache/`. Both are git-ignored.
