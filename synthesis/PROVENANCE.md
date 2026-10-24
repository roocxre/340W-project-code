# N-back Dataset Provenance

Which EEG datasets can be joined into one n-back cognitive load corpus, what has to be harmonized first, and which ones to leave out. The target label is n, the n-back level.

## Verdict

| Dataset | Source in this repo | n levels | Use |
|---|---|---|---|
| **Shin 2018** (EEG-NIRS) | [eeg-analysis/](../eeg-analysis/) | 0, 2, 3 | **Core.** Clean n-back, a strong label signal, per-trial behavior |
| **COG-BCI** (Hinss 2023) | new, Zenodo | 0, 1, 2 | **Core.** Same paradigm family as Shin, 29 subjects × 3 sessions, raw data |
| **iNCog-EEG** (Shafi 2026) | [eeg-back/](../eeg-back/) | "2, 3, 4" inside multitasking | **Leave out of training.** At most an out-of-distribution test |
| MAT, STEW (Yedukondalu 2025) | [eeg-load/](../eeg-load/) | none | **Out.** Mental arithmetic and SIMKAP, not n-back |
| OpenNeuro ds007169 | candidate | 1, 2, 3, 4 | **Optional, held-out test only.** Bridges 1–4-back, but n is confounded with time |

- **Shin and COG-BCI can be joined.** Both use digit stimuli at a ~2 s SOA in blocked levels with a 0-back control. Level order is interleaved (Shin) or randomized (COG-BCI), and both datasets have per-trial behavior and unfiltered source data. They share 25 scalp channels, or 26 if Cz is interpolated, and the 0- and 2-back levels.
- **iNCog should not be joined.** Its labels are multitasking conditions, not n-back levels. Condition is fully confounded with time in session. Its split ear reference cannot be converted to a common reference. And the label carries no EEG signal: ternary accuracy is at chance even within a subject ([eeg-back/README.md](../eeg-back/README.md)). The same features on the same 12 channels decode n in Shin at 58.5 % (chance 33 %).
- **The "join" is a union, not a key join.** No subject, session or recording appears in more than one dataset. Joining means mapping each dataset onto a common schema, then inner-joining on columns (channels, features) and on the label domain (levels present in more than one dataset).
- **The link in the request points at COG-BCI version 1, which should not be used.** Versions 2 and 3 fix "bugs in the data of participants 17 to 24", and version 4 fixes "an electrode name mismatch". Use record [7413650](https://zenodo.org/records/7413650) (v4). Its subject archives have different checksums from v1's.

## Side-by-side

| | Shin 2018 | COG-BCI v4 | iNCog-EEG | ds007169 |
|---|---|---|---|---|
| Subjects | 26 | 29 | 40 (30 "clean", 10 "noisy") | 18 released of 20 |
| Sessions | 3, same visit, concatenated per file | 3, one week apart | 1 | 1 |
| Levels | 0, 2, 3 | 0, 1, 2 | rest, then 2/3/4 inside multitasking | 1, 2, 3, 4 |
| Stimuli | digits 0–9 | digits 1–9 | letters, plus arithmetic, object tracking, go/no-go | letters |
| SOA | ~2.2 s measured (0.5 s digit) | 2.008 s; **2.516 s in 15 of 261 files** | unknown | 1.7 s (1.0 s letter) |
| Response | 2 buttons, every trial | 1 button, targets only (1/3) | none recorded | spacebar on matches |
| Target rate | 30 % | 33 % (16 of 48) | n/a | in events |
| Task time per level per subject | 9 × 40 s = 6 min | 3 sessions × 3 × 48 trials ≈ 14.5 min | ~53 s | 100 trials ≈ 2.8 min, one block |
| Level order | one fixed interleaved schedule, the same for all subjects | task order pseudo-randomized per session | fixed nw → lw → mw → hw, ~15 min apart | fixed 1 → 2 → 3 → 4 |
| Markers / behavior | per trial; `behavior.zip` matches all 14,040 trials | per trial (LSL) + behavioral `.mat` + RSME/KSS ratings | none | per trial, with accuracy |
| Amplifier | BrainAmp | actiCHamp, 64 actiCAP | KT88 | io:bio mobile |
| EEG channels | 28 + HEOG/VEOG | 62 + ECG (TP9 used for ECG; no Cz for subjects 1–9) | 16 + 16 undocumented `Add_lead` | 19 + ECG |
| Reference | TP9 (not stored) | Fpz (not stored) | ipsilateral ear, A1 left / A2 right | linked ears |
| Sampling as shared | 200 Hz (recorded at 1000) | 500 Hz | 200 Hz | 250 Hz |
| Filtering as shared | none | none (DC offsets of several mV, 50 Hz line ~30 dB above floor) | device band ~0.5–30 Hz, no line peak | 50 Hz notch, 125 Hz low-pass |
| Format | BBCI `.mat` | BIDS-like EEGLAB `.set/.fdt` | EDF, unreliable headers | BIDS BrainVision |
| License | see dataset page | CC BY 4.0 | CC BY 4.0 (paper: non-commercial) | CC0 |

## Findings by dataset

### COG-BCI (new)

Checked on subject 1 locally and on the events of all 29 subjects through HTTP range reads of the `.set` files ([scripts/cogbci_scan_events.py](scripts/cogbci_scan_events.py), output in [results/cogbci_nback_events.jsonl](results/cogbci_nback_events.jsonl)).

- **Complete.** All 261 n-back files (29 × 3 sessions × 3 levels) are present, each with exactly 144 trials (3 blocks of 48) and one EEGLAB boundary at t = 0, so there are no mid-recording splices.
- **The SOA is not uniform.** The documented 500 ms digit plus 1500 ms blank (2.0 s) holds in 246 files. The other 15, sub-01 S1–S2 and sub-02–04 S1, run at 2.516 s with blocks of 127 s instead of 102 s. Any epoching that assumes a fixed SOA, or a fixed window count per block, must read trial onsets from the markers.
- **Marker quirks.**
  - 0-back emits the block-start code `6011` twice per block, 4 s apart (instruction, then start), in every file.
  - The 2-back conflict-onset code `6223` never occurs, although the trigger list defines it. `6233` ("conflict error") occurs in every 2-back file.
  - Hits are coded at onset (`6x22`), so target and non-target trials can be separated without the behavioral files.
- **Hit count.** 48 hits in 144 trials confirms 16 per block (1/3), as documented.
- **Raw signal.** Data are unfiltered and DC-coupled, with per-channel offsets up to ~20 mV and a strong 50 Hz line. A high-pass and line removal are needed before anything else.
- **Channels.** 63 channels: 62 EEG and `ECG1` in position 10, the TP9 slot. FCz is present because the reference is Fpz. Cz is absent for subjects 1–9 and must be interpolated or dropped. Per-session 3-D electrode positions are provided.
- **Behavior files are MATLAB objects.** `behavioral/*.mat` and `notebook.mat` load in scipy only as `MatlabOpaque` (MATLAB tables). `notebook.mat` decodes with `mat-io` ([results/cogbci_notebook.csv](results/cogbci_notebook.csv)). It shows that each n-back level occupies every one of the 8 task positions in some session (so n is not confounded with time in session), and that 8 sessions were flagged as interrupted. The notes for 5 of them concern an n-back run (redone in 3, recorded later in 1, followed by a crash in 1); the other 3 have no note. `RSME.txt` and `KSS.txt` are plain CSV and give a subjective workload rating per condition and session.
- **The archives are not laid out alike.** sub-01 and sub-02 unpack to `sub-XX/ses-SY/`, while sub-03 to sub-29 nest an extra `sub-XX/sub-XX/ses-SY/` level. Code that assumes one layout misses files.
- **What the paper found.** Workload had no significant effect on frontal theta or on alpha (only a session effect on theta). A 3-class Riemannian MDM classifier with within-subject 5-fold CV reached 65 %. Heart rate rose and HRV fell with n. EDA on the joined set should check whether COG-BCI shows the alpha drop that Shin shows.

### Shin 2018

- **The label signal is strong on the channels that survive any join.** On the 12 channels shared with iNCog, log alpha falls from 0- to 2-back in 24 of 26 subjects (p < 10⁻⁶) and to 3-back in 23 of 26. Theta does not change. Leave-one-subject-out logistic regression on per-subject z-scored band power gives 58.5 % for 0/2/3 (all 26 subjects above chance), 80.1 % for 0 vs 2, and 59.8 % for 2 vs 3 ([scripts/shin_nback_signal.py](scripts/shin_nback_signal.py), [results/shin_nback_signal.txt](results/shin_nback_signal.txt)).
- **Order.** Every subject ran the same 27-series schedule, `3 2 0 2 0 3 0 3 2 | 2 0 3 0 3 2 3 2 0 | 0 3 2 3 2 0 2 0 3`. Levels are interleaved, so mean position is equal across levels, but the schedule is not counterbalanced across subjects.
- **Measured SOA is ~2.2 s**, not the nominal 2.0 s.
- **The raw files are no longer on disk.** Only `eeg-analysis/cache/paper/` remains, and it is already 1–40 Hz filtered and AAR ocular-cleaned. The other datasets would not get that cleaning, so the source `.mat` files should be re-downloaded and every dataset taken through the same pipeline.
- **No 1-back.** Its 0-back requires a button press on every trial, like its other levels.

### iNCog-EEG (eeg-back)

The full audit is in [eeg-back/README.md](../eeg-back/README.md#data-provenance). The points that decide the join:

- **The label is not n.** Each level changes the n-back, arithmetic, object tracking and go/no-go difficulty together, so a model cannot attribute an effect to n. `nw` is rest, not a 0-back control.
- **Condition is confounded with time.** The KT88 record numbers rise nw < lw < mw < hw in every subject, about 15 min apart.
- **Too little data per label.** ~53 s per condition (not the 5 min stated), with no markers and no behavior.
- **The reference cannot be harmonized.** Left leads are referenced to A1 and right leads to A2. Without the A1–A2 signal, no common average or linked-ear reference can be reconstructed, so spatial features (covariance, asymmetry, topography) are not comparable with the other datasets.
- **The device filtered the signal** to ~30 Hz, which would cap a joined pipeline's band at 30 Hz if iNCog were kept.
- **No recoverable signal.** Workload level is at chance under every feature set and validation scheme tried, including within subject. Its "noisy" group is not noisier.

### MAT and STEW (eeg-load)

Neither is an n-back task: MAT is mental arithmetic and STEW is SIMKAP multitasking. They fall outside the stated scope. Their only possible use is a cross-task transfer test after the n-back classifiers exist.

## Joining Shin and COG-BCI

### Channels

| Set | Channels |
|---|---|
| Shin ∩ COG-BCI (25) | Fp1 Fp2 AFz F1 F2 FC1 FC2 FC5 FC6 T7 T8 C3 C4 CP1 CP2 CP5 CP6 P3 P4 P7 P8 Pz POz O1 O2 |
| + Cz (26) | interpolate Cz for COG-BCI subjects 1–9 from the 3-D positions |
| Shin only | AFF5h, AFF6h (drop) |
| All three with iNCog (12) | Fp1 Fp2 C3 C4 T7 T8 P3 P4 P7 P8 O1 O2 |
| With ds007169 (14) | Shin ∩ ds007169: Fp1 Fp2 T7 T8 C3 Cz C4 P7 P3 Pz P4 P8 O1 O2 |

Shin has no Fz, F3/F4 or F7/F8. Frontal measures in the joined set have to use F1/F2/AFz/FC1/FC2.

### Harmonization steps

1. **Source versions.** Pin COG-BCI v4 (record 7413650) and record checksums. Re-fetch Shin's raw `.mat` files rather than reusing the AAR-cleaned cache.
2. **Channels.** Pick the 25 (or 26) shared channels. Drop COG-BCI's `ECG1` and Shin's EOG after any ocular step that uses them.
3. **Reference.** Re-reference both to the common average of the shared channels. Both have a single reference (TP9, Fpz), so this is exact.
4. **Sampling and band.** Resample COG-BCI to 200 Hz to match Shin. High-pass at 1 Hz first (COG-BCI's DC offsets), notch at 50 Hz, then band-limit to 1–40 Hz.
5. **Artifacts.** Apply one method to both datasets. Shin has EOG channels and COG-BCI does not, so use a method that needs no EOG reference (ICA with ICLabel, as COG-BCI's own pipeline does, or ASR).
6. **Epochs.** Take windows within task blocks only, excluding instructions and Shin's 20 s rest. Read trial onsets from markers to handle Shin's 2.2 s SOA and COG-BCI's 2.0 s / 2.5 s split. Trial-locked epochs suit ERPs; 2–5 s windows suit spectral features.
7. **Schema.** Store one row per window: `dataset, subject (namespaced, e.g. shin-VP001 / cog-01), session, block, window, n, n_targets, accuracy`, plus features. Join on the shared feature columns.

### Label domain and confounds

- **Levels unique to one dataset leak the dataset.** Training on Shin {0, 2, 3} ∪ COG-BCI {0, 1, 2} lets a model predict "3" from anything that marks a recording as Shin's, and "1" from anything that marks it as COG-BCI's. Options, cleanest first:
  - shared levels only, {0, 2};
  - low/high within dataset, 0 vs the highest level, which is 2 in COG-BCI and 3 in Shin;
  - all levels, scored on {0, 2} as well, with dataset identity checked for decodability from the features.
- **Normalize per subject.** Amplifier, cap and reference differ, so z-score features within subject across all of that subject's levels. In Shin this normalization gives the 58.5 % above.
- **Validate across groups.** Use leave-one-subject-out within the pool and leave-one-dataset-out across it. In the two eeg-load datasets, segment-level CV inflated accuracy by 20–30 points through recording recognition.
- **Volume.** COG-BCI contributes about 2.7× Shin's minutes per level. Weight by subject, not by window.
- **Motor load.** It is constant across levels within each dataset (every trial in Shin, targets only in COG-BCI) but differs between them. Per-subject normalization absorbs the offset. Do not compare raw mu/beta power across datasets.

## Other datasets

| Dataset | Verdict | Why |
|---|---|---|
| [OpenNeuro ds007169](https://openneuro.org/datasets/ds007169) (Barras & Booth) | optional, held out | 18 subjects, letters, 1- to 4-back, 19 channels (10-20), linked ears, 250 Hz, per-trial accuracy, ECG and pupil, CC0. But all three subjects checked ran 1 → 2 → 3 → 4 in a single block each, so n is confounded with time-on-task, as in iNCog. LSL dropped 0.15–0.54 % of samples in bursts of up to 53 samples, and its README does not say whether event onsets account for them. Its 1- and 3-back levels would link COG-BCI and Shin. Best used as a leave-dataset-out test. |
| [HHU n-back](https://ieee-dataport.org/documents/hhu-n-back-task-eeg-dataset) | skip | 10 subjects, spatial (not digit) n-back, IEEE DataPort subscription, no files listed, already epoched |
| [OpenNeuro ds007554](https://openneuro.org/datasets/ds007554/versions/1.0.0) (CMx7-MM) | skip | a single auditory 2-back level, no load contrast |
| ds003838 (digit span), ds003655 (Sternberg) | skip | working memory, but not n-back |
| COG-BCI MATB, resting state | out of scope | Same subjects, so it could later test whether a workload model transfers across tasks |
