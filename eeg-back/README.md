# iNCog-EEG Reproduction

Reproduction, in Python, of the Technical Validation in Shafi et al. 2026, "A Multitasking EEG Dataset for Cognitive Workload Assessment with Controlled Task Difficulty", *Scientific Data*, doi:10.1038/s41597-026-07607-3 ([docs/s41597-026-07607-3_reference.pdf](docs/s41597-026-07607-3_reference.pdf)). The authors released their analysis functions as [EEGCWLD](https://github.com/fariya17/EEGCWLD) (copied under [reference/EEGCWLD/](reference/EEGCWLD/)), and the scripts here call those functions unmodified.

The results table is in [results/paper_vs_ours.md](results/paper_vs_ours.md), and the figures are in [results/figures/](results/figures/).

## Results

| Target | Paper | Ours |
|---|---|---|
| Fig. 5, PSD per workload (Fp1-A1) | NW peak 10.16 Hz; LW 1.37 Hz / ×0.42; MW 2.54 Hz / ×0.75; HW 2.34 Hz / ×1.49 | **Exact**, from sub02 with the released `compute_psd_mne` |
| Fig. 6, noisy/clean power | 4.01 | No released pair matches; group noisy/clean ratio 0.73 |
| Fig. 7, clean vs noisy correlation | two no-workload recordings, one clean, one noisy | **Exact** (RMSE 0.008), but the panels are `sub07_nw` and `sub07_hw`: the same clean subject at rest and under high workload |
| Fig. 8, BEAM theta/alpha maps | frontal theta up, posterior alpha down under workload | Maps not reproducible; theta claim fails (15/30, p = 0.47), alpha claim holds weakly (19/30, p = 0.04) |
| RM-ANOVA, frontal theta, LW/MW/HW | F(2, 58) = 4.21, p = 0.02, monotonic | F = 0.27, p = 0.76; 0 of 30 analysis variants reach p < 0.05 |
| Table 3, MLP / XGBoost | see below | see below |

- **The figures that reproduce are single recordings, and they are not representative.** Fig. 5 is subject 02 alone. Across the 30 clean subjects the lw < mw < hw power ordering appears in 5, the chance rate, and only 5 have an alpha peak at rest.
- **Fig. 7 is mislabelled.** The digitised heatmaps match one recording each with r = 0.999. Both are from clean subject 07, so the figure contrasts rest with high workload, not clean with noisy data. At the group level, the noisy subjects' rest correlations are not significantly lower (median r 0.17 vs 0.21, p = 0.06), and the group difference spans −0.01 to +0.11, against −0.27 to +0.53 in the figure.
- **Fig. 6 and Fig. 8 cannot be traced to released files.** Fig. 6's clean trace matches record 276 (`sub31_lw`, filed under *noisy* in the release), and its noisy trace is record 256, which is not released. Fig. 8 was drawn by other software, probably the acquisition program's built-in BEAM display, and no recording matches it better than chance.
- **Frontal theta does not increase with workload.** Neither the paper's ANOVA nor its BEAM claim holds in any of the analysis variants tried. Alpha is lower under workload than at rest in most subjects, over the whole scalp rather than only posteriorly.

### Table 3: baseline classification

Accuracy in %, log band power features on 2 s windows. "Segment" pools windows across subjects in stratified 10-fold CV; "subject" is leave-one-subject-out. A second, 256-feature set gives the same picture ([results/paper_vs_ours.md](results/paper_vs_ours.md)).

| Task | Group | Model | Paper | Majority class | Ours, segment | Ours, subject |
|---|---|---|---|---|---|---|
| Rest vs workload | clean | MLP | 83.6 | 74.5 | 80.0 | 79.7 |
| Rest vs workload | clean | XGBoost | 87.5 | 74.5 | 79.9 | 79.8 |
| Rest vs workload | noisy | MLP | 71.3 | 74.9 | 84.8 | 83.6 |
| Rest vs workload | noisy | XGBoost | 78.1 | 74.9 | 86.2 | 85.0 |
| Low / moderate / high | clean | MLP | 79.8 | 33.4 | 33.5 | 33.2 |
| Low / moderate / high | clean | XGBoost | 82.8 | 33.4 | 33.2 | 34.7 |
| Low / moderate / high | noisy | MLP | 64.0 | 33.6 | 34.9 | 34.7 |
| Low / moderate / high | noisy | XGBoost | 69.2 | 33.6 | 31.6 | 31.2 |

- **Workload level cannot be classified.** Every ternary model is at chance (31-36 %), even with segment-level CV, which lets the model see other windows of the test recording. In the two other reproductions in this repository, segment-level CV inflated accuracy by recognising recordings; here the recordings are not recognisable (see Data provenance), so it does not.
- **Rest vs workload is weakly separable.** Clean accuracy is 5 points above always predicting "workload" (balanced accuracy 64 %). The paper's binary numbers sit 9-13 points above the majority class for clean data, and its noisy MLP (71.3 %) is below it.
- **The noisy group classifies better, not worse.** Rest vs workload reaches 84-87 % on subjects 31-40 against 80 % on 01-30. Ten-subject subsets of the clean group stay at 79-80 %, so the smaller sample does not explain it.
- **The paper's features and validation are unknown**, so its exact numbers cannot be rebuilt. Two feature sets give the same result, and a classifier trained within each subject cannot tell that subject's three task recordings apart either (median 33 %, stage 1), so other spectral features are unlikely to change it.

## Data provenance

These findings matter for joining iNCog-EEG with the other n-back datasets.

- **What the labels are.** Each subject has one continuous recording per condition: rest (`nw`) and three multitasking levels whose memory component is a 2-back (`lw`), 3-back (`mw`) and 4-back (`hw`) letter task (paper, Table 2). The n-back runs at the same time as arithmetic, multiple-object tracking and a go/no-go task, whose difficulty rises with it. There are no event markers, no trial structure and no behavioural data (the paper's Limitations say none was recorded).
- **Recordings are about 53 s, not 5 min.** 153 of 160 files are 53-54 s; the longest is 93 s. The paper's Methods say each condition lasted 5 minutes and that the software saved "approximately 1 minute". Which minute was kept is not stated.
- **Acquisition order is recoverable.** The EDF patient field holds the KT88's running record number (18-368). It rises strictly through subjects and, within each subject, nw < lw < mw < hw. So condition is fully confounded with time in session, about 15 min apart per step; the paper's Fig. 1 protocol says the same.
- **Headers are unreliable.** Every file has the same start date and time (01.04.13 11:12:13, a device default), and the declared record count (90-158 s) never equals the stored data (53-93 s).
- **The device filtered the signal.** The mean spectrum falls 12.8 dB between 30 and 40 Hz and has no 50 Hz line peak, with a high-pass near 0.5 Hz. The paper says "no preprocessing has been applied", and lists power-line interference among the noisy group's artifacts, but none is present.
- **16 undocumented channels.** `Add_lead1-16` carry EEG-like signal with the same spectrum as the scalp leads and correlate up to 0.83 with them. They are not described anywhere and are excluded here, as in the paper's figures.
- **The "noisy" subjects 31-40 are not noisier.** On standard deviation, 99th-percentile amplitude, fraction of samples over 100 µV and low-frequency power, they sit inside the clean group (p = 0.05-0.94), and their beta power is lower (p = 0.03). The eight highest-amplitude subjects are all "clean" (09, 10, 20, 22, 17, 19, 11, 18).
- **Weak individual signature.** A 5-nearest-neighbour classifier on 2 s log spectra identifies the subject in 16 % of windows (chance 2.5 %) and the recording in 4.6 %. The first half of a recording's spectrum is closer to its own second half in only 14 % of files. Inter-channel correlation is low for ear-referenced EEG (Fp1-Fp2 median r = 0.31), and leads sharing a reference correlate only slightly more than leads that do not (0.23 vs 0.14). The cause cannot be determined from the files.
- **No duplicated recordings.** The strongest lagged cross-correlation between two different files is 0.36.

## Methods

### Pipeline

| Stage | Script | What it does |
|---|---|---|
| 1 | `01_audit.py` | Data audit: lengths, headers, record numbers, Add_lead channels, filter shape, clean vs noisy artifact measures, duplicate search. |
| 2 | `02_psd.py` | Figs 5 and 6 with the released `compute_psd_mne` / `compute_psd_welch` and `analyze_bands`. Identifies Fig. 5's subject and searches all 25,440 ordered pairs for Fig. 6. Repeats Fig. 5's statistics for every clean subject. |
| 3 | `03_correlation.py` | Fig. 7: digitises the paper's heatmaps and matches them against the released `compute_correlation` of every recording; group-level clean vs noisy comparison. |
| 4 | `04_beam.py` | Fig. 8: released `compute_topomap` band power, group-mean maps, digitised-map search, and per-subject tests of the theta and alpha claims. |
| 5 | `05_anova.py` | Repeated-measures ANOVA on frontal theta over 30 variants (5 channel sets × 3 power measures × linear/log). |
| 6 | `06_classification.py` | Table 3: MLP and XGBoost, binary and ternary, clean and noisy, under segment-level and leave-one-subject-out CV, plus a sample-size control. |
| 7 | `07_report.py` | Paper-vs-ours tables. |

### What came from the released code

The repository has three function modules and no driver scripts, data paths or figure code. [backrepro/released.py](backrepro/released.py) loads them unmodified from `reference/`. The only intervention is restoring `np.trapz`, which NumPy 2.4 removed. The released functions used are `compute_psd_mne`, `compute_psd_welch` and `analyze_bands` (PSD), `compute_correlation` (correlation), `compute_topomap` and its band definitions (BEAM), and `DEFAULT_BANDS` (classification features).

The paper's figures were not drawn with the released plotting functions: Fig. 7's colormaps contradict `plot_correlation_with_difference`'s rule for choosing them, `plot_beam_grid` scales each map separately with no colorbar, and Fig. 8 is not MNE output. The figures here use the paper's layouts.

### How recordings were identified

- **Fig. 5:** the four peak frequencies and three power ratios fall on exact Welch bins (10.16 Hz = 104 × 200/2048), so a match is exact or absent. Only sub02 with `compute_psd_mne` gives all seven.
- **Fig. 7:** [backrepro/digitize.py](backrepro/digitize.py) extracts the bitmap from the PDF and inverts the `coolwarm` colormap (mean colour error 0.006), giving each 16×16 matrix to about ±0.02. Each panel matches one recording at RMSE 0.008; the runner-up is at 0.075.
- **Fig. 8:** the same approach, sampling each map at the 16 electrode positions and rank-correlating. Every best match is within the range that randomly shuffled maps reach.

### Choices the paper does not specify

| Choice | Here | Basis |
|---|---|---|
| Channels | the 16 documented leads, `Add_lead` dropped | the paper's figures |
| Frontal / posterior regions | Fp1 Fp2 F3 F4 F7 F8 (ANOVA); Fp1 Fp2 F3 F4 vs P3 P4 O1 O2 (BEAM claims) | montage has no midline leads; ANOVA also run on four other sets |
| Theta power for the ANOVA | released `compute_topomap` (primary); also `analyze_bands` absolute and relative, linear and log | the paper's own BEAM measure |
| "Workload" in BEAM | mean of lw, mw, hw | the paper pools them |
| Classification features | 2 s windows; log band power in the released bands (80), or with relative power and time-domain descriptors (256) | standard; 2 s matches `compute_topomap` |
| Classification validation | stratified 10-fold over windows, and leave-one-subject-out | paper gives none |
| MLP | (128, 64), early stopping | default-like |
| XGBoost | 300 trees, depth 4, learning rate 0.1 | default-like |

## Workflow

Python 3.12 with the packages in [requirements.txt](requirements.txt):

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q tests
for s in 01_audit 02_psd 03_correlation 04_beam 05_anova 06_classification 07_report; do
    .venv/bin/python scripts/$s.py > results/log_$s.txt
done
```

Stages 1-5 take a few minutes in total; stage 6 takes about 10 minutes on 8 cores.

## Data

Download `iNCog-EEG (ideal vs. Noisy Cognitive EEG for Workload Assessment) Dataset.zip` from https://doi.org/10.6084/m9.figshare.30003127.v2 and unzip it into `data/iNCog-EEG/` (git-ignored), giving `data/iNCog-EEG/sub01/sub01_nw.edf` and so on. Checksums are in [reference/SOURCES.txt](reference/SOURCES.txt).

## Licenses

The dataset is CC BY 4.0 (the paper restricts it to non-commercial research use). The released code is MIT. The paper is CC BY-NC-ND 4.0.
