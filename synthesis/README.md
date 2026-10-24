# Joined N-back EEG Dataset

Shin 2018 ([eeg-analysis/](../eeg-analysis/)) and COG-BCI v4 (Hinss 2023) joined into one table of spectral features for classifying n-back load. Why these two and not iNCog, MAT or STEW is in [PROVENANCE.md](PROVENANCE.md).

The product is [results/nback_features.csv](results/nback_features.csv): one row per 4 s window of task EEG, 26,137 rows × 282 columns.

## Results

### What is in the table

| | Shin | COG-BCI | Total |
|---|---|---|---|
| Subjects | 26 | 29 | 55 |
| Levels (n) | 0, 2, 3 | 0, 1, 2 | 0–3 |
| Windows | 7,075 | 19,062 | 26,137 |
| Windows per level | 2,358–2,359 | 6,354 | |
| Task EEG per level | 157 min | 424 min | |
| Windows per subject | 270–297 | 648 (702–756 with 2.5 s SOA files) | |
| Artifact windows (> 150 µV peak-to-peak) | 5.0 % | 4.1 % | 4.4 % |

| n | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Windows | 8,713 | 6,354 | 8,712 | 2,358 |
| Datasets | both | COG-BCI | both | Shin |

- **Classes are balanced within each dataset**, but not across the pool: 1-back exists only in COG-BCI and 3-back only in Shin.
- **COG-BCI is 73 % of the rows.** It has more subjects, three sessions each and longer blocks, so weight by subject when pooling.
- **No missing feature values.** `task_position` is empty for Shin (no equivalent) and `n_targets` is empty for Shin's 0-back (its markers do not code targets).

### Viability for ML

Logistic regression on the 260 features, z-scored within each subject, artifact windows dropped, and no subject in both training and test ([results/log_04_viability.txt](results/log_04_viability.txt)).

| Task | Accuracy | Chance | Subjects above chance |
|---|---|---|---|
| Shin 0 / 2 / 3 | 47.6 % ± 6.0 | 33 % | 26/26 |
| COG-BCI 0 / 1 / 2 | 46.5 % ± 9.0 | 33 % | 28/29 |
| **Pooled 0 vs 2** | **69.7 % ± 9.2** | 50 % | 53/55 |
| Pooled low vs high (Shin 0/3, COG-BCI 0/2) | 70.2 % ± 9.6 | 50 % | 53/55 |
| Pooled 0 / 1 / 2 / 3 | 43.4 % ± 8.1 | 33 % (majority) | 51/55 |
| **Train Shin, test COG-BCI, 0 vs 2** | **65.3 % ± 8.5** | 50 % | 28/29 |
| **Train COG-BCI, test Shin, 0 vs 2** | **63.9 % ± 6.9** | 50 % | 26/26 |

- **The join works.** A model trained on one dataset classifies 0- vs 2-back in the other at 64–65 %, with nearly every subject above chance. The datasets share a load signal, not just a label.
- **Pooling does not hurt either dataset.** Pooled 0 vs 2 scores 68 % on Shin's subjects and 71 % on COG-BCI's.
- **The signal is physiological and consistent.** Parietal alpha falls with load in 25/26 Shin and 27/29 COG-BCI subjects at the top level (p < 10⁻⁴). Frontal theta rises (p < 0.02 in both). COG-BCI's paper reported neither effect.
- **Behavior confirms the labels.** Accuracy falls (Shin 0.99 → 0.92 → 0.83; COG-BCI 0.99 → 0.99 → 0.89) and RT rises with n in both.
- **Dataset identity is removed by per-subject normalization.** It is decodable from the raw features at 91 % (majority 73 %), but at exactly the majority rate after z-scoring within subject, including on the shared levels only. Leaving features un-normalized would let a model learn the dataset instead of the load.
- **Window-level accuracy is the conservative number.** The ternary scores are below the 58.5 % reported in [PROVENANCE.md](PROVENANCE.md) for Shin because that check averaged whole 40 s series. Pooling windows per block, or using sequence models, should recover some of the difference.

### Recommended use

- **Normalize within subject** before training.
- **Primary target: 0 vs 2** (both datasets), or low vs high. Treat 1- and 3-back as single-dataset levels, scored separately or used as ordinal neighbours.
- **Validate by subject**, and report leave-one-dataset-out as well.
- **Drop or down-weight `artifact == True`**, and decide whether to keep the 8 interrupted COG-BCI sessions (6.8 % of rows).
- **Weight samples by subject**, so COG-BCI does not dominate.

### Known weak spots

- **Residual ocular activity in two subjects.** In shin-VP004 (artifact rate 52 %) and cog-13 session 2 (59 %), ICLabel did not rate the eye components above the 0.9 removal threshold. Both are captured by `artifact`. Lowering the threshold to 0.8 and rerunning stage 2 (~1 h) is the fix if they matter.
- **COG-BCI's Fp1/Fp2 are interpolated in 8 of 87 sessions.** They sit next to the Fpz reference and sometimes show too little signal to pass the correlation check.
- **Cz is interpolated for COG-BCI subjects 1–9**, as it was not recorded (`cz_interpolated`).

## Data structure

**A CSV fits feature-based models** (logistic regression, SVM, trees, gradient boosting, MLPs on features). Models that learn from the signal itself (EEGNet, CNNs, transformers, Riemannian methods on covariance) need the time series, which a flat table cannot hold well: 26 × 800 samples per window would mean 20,800 columns. Those windows are saved in `cache/windows/<subject>.npz` as `x`, an array of shape `(n_windows, 26, 800)` in float32 µV at 200 Hz. Each row of the CSV maps to one via `subject` and `npz_index`. The cache is 2.1 GB and git-ignored; stage 2 regenerates it.

## Columns

| Column | Meaning |
|---|---|
| `row_id` | 0 … 26136 |
| `dataset` | `shin` or `cog` |
| `subject` | `shin-VP001` … `shin-VP026`, `cog-01` … `cog-29` |
| `session` | 1–3. Shin: three sessions on one day. COG-BCI: one week apart |
| `run` | source recording: Shin session (`VP001_S1`), COG-BCI level file (`sub-01_ses-S1_zeroBACK`) |
| **`n`** | **label: the n-back level** |
| `block_in_run` | block index within the run. Shin: series position in the session (1–9). COG-BCI: block within the level file (1–3) |
| `window`, `t_in_block_s` | window index in the block, and its start time from the first trial |
| `task_position` | COG-BCI only: position of this level's file among the session's 8 tasks (from `notebook.mat`) |
| `interrupted` | COG-BCI only: experimenter flagged the session (notes in [results/cogbci_notebook.csv](results/cogbci_notebook.csv)) |
| `soa_s` | trial onset interval in the block (Shin ~2.2 s; COG-BCI 2.008 s, 2.516 s in 15 files) |
| `n_trials`, `n_targets` | trial onsets and target trials inside the window |
| `block_accuracy`, `block_mean_rt_s` | behavior over the block (definitions under Methods) |
| `cz_interpolated` | Cz was not recorded and was interpolated (COG-BCI 1–9) |
| `n_bad_channels`, `n_ica_removed` | per-session preprocessing counts |
| `ptp_max_uv`, `artifact` | largest peak-to-peak amplitude in the window; `artifact` = above 150 µV |
| `npz_index` | row of this window in `cache/windows/<subject>.npz` |
| `abs_<band>_<ch>` | log10 absolute band power, µV² (130 columns) |
| `rel_<band>_<ch>` | band power as a share of 1–40 Hz power (130 columns) |

Bands: delta 1–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–40 Hz. Channels (26): Fp1 Fp2 AFz F1 F2 FC5 FC1 FC2 FC6 T7 C3 Cz C4 T8 CP5 CP1 CP2 CP6 P7 P3 Pz P4 P8 POz O1 O2.

## Methods

### Pipeline

| Stage | Script | What it does |
|---|---|---|
| 0 | `00_cogbci_notebook.py` | Exports COG-BCI's `notebook.mat` (a MATLAB table) to [results/cogbci_notebook.csv](results/cogbci_notebook.csv): date, task order, interruptions |
| 1 | `01_fetch.py` | Range-reads only the n-back files from the source archives and checks every file's CRC-32. Shin: 78 files (2.3 GB) into `eeg-analysis/Cleaned Data/`. COG-BCI: 522 files (9.4 GB compressed) into `data/cogbci/` |
| 2 | `02_harmonize.py` | Per session: harmonize (below), cut windows, write `cache/windows/<subject>.npz` |
| 3 | `03_table.py` | Spectral features, notebook metadata, artifact flag → `nback_features.csv`, plus `preprocessing_qa.csv` |
| 4 | `04_viability.py` | Dimensions, quality, behavior, spectral effects, baselines, dataset identity |

### Harmonization ([nbackjoin/preprocess.py](nbackjoin/preprocess.py))

The same steps for both datasets, run per recording session (Shin: one run; COG-BCI: the three level files together for ICA):

1. 1 Hz high-pass (removes COG-BCI's DC offsets of several mV), and 50 Hz notch plus harmonics.
2. Resample to 200 Hz (COG-BCI from 500 Hz).
3. Interpolate broken channels only: robust spread below 0.1× or above 10× the median channel, or largest |r| with any other channel below 0.2, measured in the recording reference.
4. Common average reference over each dataset's native EEG channels (28 Shin, 62 COG-BCI).
5. ICA (extended Infomax via Picard), ICLabel, and removal of eye, muscle and heart components with probability > 0.9, COG-BCI's own published criterion. No EOG channel is used, because COG-BCI has none. On Shin, whose EOG is available as a check, Fp1/Fp2–VEOG correlation falls from 0.92 to 0.15. Components removed per session: Shin 4.6 (2.1 eye), COG-BCI 7.8 (2.0 eye).
6. 40 Hz low-pass.
7. Interpolate Cz where it was not recorded, keep the 26 shared channels, and re-reference to their common average.

### Windows and features ([nbackjoin/features.py](nbackjoin/features.py))

- **Windows:** 4 s, non-overlapping, from a block's first trial onset to its last onset + SOA. Instructions, Shin's 20 s rests and the gaps between blocks are excluded. Onsets come from markers, which absorbs the different SOAs.
- **Features:** Welch PSD per window (2 s Hann segments, 50 % overlap, 0.5 Hz resolution), integrated over each band.
- **Behavior:**
  - Shin: fraction correct among trials that have an answer (the first n of a series do not), and mean RT of correct trials. From `behavior.zip`, checked against every marker by eegrepro.
  - COG-BCI: (correct hit responses + non-targets without a false alarm) / trials, and mean RT from hit onset to correct response. From the LSL markers.

### Choices and their basis

| Choice | Here | Basis |
|---|---|---|
| Rate | 200 Hz | Shin is distributed at 200 Hz |
| Band | 1–40 Hz | iNCog's 30 Hz limit no longer applies once it is left out |
| Channels | 26 shared, Cz interpolated for COG-BCI 1–9 | [PROVENANCE.md](PROVENANCE.md) |
| Reference | common average of the 26 | both datasets have a single reference, so this is exact |
| Ocular removal | ICA + ICLabel, p > 0.9 | COG-BCI's validation pipeline; works without EOG |
| Window | 4 s, no overlap | ≥ 2 Welch segments per window; several windows per Shin block |
| Artifact flag | 150 µV peak-to-peak, flag only | rows are kept so the threshold can be changed downstream |
| Label | n | the project's definition of load |

### Rejected bad-channel rules

Three earlier rules interpolated real channels. They are recorded here because the failure modes matter for anyone changing step 3:

- **SD-based rules** flagged Fp1/Fp2: blinks inflate their SD.
- **A 5× robust-spread rule** still flagged them in 6 Shin subjects. Eye movements during digit reading are continuous, putting Fp at up to 4.2× the median.
- **Measuring on an average reference** spreads a broken channel into all the others.
- **Measuring on a median reference** strips Cz, which tracks the per-sample median, so Cz looked decorrelated in 4 Shin subjects.

## Workflow

Python 3.12 with the packages in [requirements.txt](requirements.txt):

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/00_cogbci_notebook.py
.venv/bin/python scripts/01_fetch.py all --jobs 6        # ~12 GB of downloads; resumable
.venv/bin/python scripts/02_harmonize.py all --jobs 6    # ~1 h on 8 cores; per-subject cache
.venv/bin/python scripts/03_table.py
.venv/bin/python scripts/04_viability.py
```

Stage 2 reuses Shin's validated loader from [eeg-analysis/eegrepro/](../eeg-analysis/eegrepro/). `data/`, `cache/` and `eeg-analysis/Cleaned Data/` are git-ignored. The CSV is 65 MB, over GitHub's 50 MB warning size; commit it with Git LFS, or regenerate it from stages 1–3.

## Licenses

COG-BCI is CC BY 4.0. Shin's dataset page states no license and asks for citation of Shin et al. 2018 and 2017 and Blankertz et al. 2010.
