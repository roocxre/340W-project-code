# Machine learning for working-memory load classification

| Directory | Contents |
|---|---|
| `gait-analysis/` | Parent Paper 1 codebase |
| `eeg-analysis/` | Parent Paper 2 codebase (Shin et al. 2018 reproduction) |
| `eeg-load/` | Parent Paper 3 codebase (Yedukondalu et al. 2025 reproduction on MAT and STEW) |
| `synthesis/` | Shin 2018 + COG-BCI joined into one n-back feature table ([README](synthesis/README.md)) |
| `analysis/` | 70/20/10 split, EDA and load classifiers on the joined table ([README](analysis/README.md)) |
| `docs/` | paper draft |

## Reproducing every result

**Only Python needs to be installed beforehand** (3.8 or newer, from [python.org](https://www.python.org/downloads/) or the system's package manager). The script installs everything else itself, including Python 3.12 and the exact package versions in [requirements.txt](requirements.txt).

```bash
git clone https://github.com/roocxre/340W-project-code.git
cd 340W-project-code
python run_pipeline.py            # Windows: py run_pipeline.py   (Linux/macOS may need python3)
```

That single command downloads the datasets, joins them and runs every analysis stage of `eeg-load/`, `synthesis/` and `analysis/`, unattended. It takes about a day on an 8-core machine. Without git, use **Code → Download ZIP** on the GitHub page and run the same command in the unpacked folder; nothing depends on `.git`.

### The one manual step: STEW

STEW (used by `eeg-load/`) is served by IEEE DataPort to logged-in users only, so no script can download it:

1. Sign in (a free IEEE account is enough) at [the STEW dataset page](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset).
2. Download `STEW Dataset.zip` (52 MB) and leave it in your Downloads folder.

The script finds the zip there, or in `eeg-load/data/`, or wherever `--stew PATH` points. It checks for STEW before anything long starts, and stops with these instructions if the zip is missing. Every other dataset is downloaded automatically and verified (CRC-32 or SHA-256):

| Dataset | Source | Used by | Size |
|---|---|---|---|
| Shin 2018, n-back files only | TU Berlin | synthesis, analysis | 2.3 GB |
| COG-BCI v4, n-back files only | Zenodo record 7413650 | synthesis, analysis | 9.4 GB |
| MAT (eegmat 1.0.0) | PhysioNet | eeg-load | 175 MB |
| STEW | IEEE DataPort (manual, above) | eeg-load | 52 MB |

### Requirements

| | |
|---|---|
| Operating system | macOS 14+ on Apple Silicon, Linux x86_64 or Windows x86_64. Intel Macs cannot install the pinned PyTorch / numba. |
| Disk | about 30 GB free: 13 GB of raw EEG, 2 GB of harmonized windows, caches, and the environment |
| Memory | 16 GB recommended. Parallel workers are sized to the RAM, so less memory means slower, not failing. |
| Network | only for the download stages at the start (12 GB, typically 1-2 h; Zenodo can be slow) |
| Time | about 23 h on an 8-core Apple M1. `python run_pipeline.py --list` shows the time per stage. |

Keep the machine powered and awake. The script blocks idle sleep while it runs (caffeinate on macOS, SetThreadExecutionState on Windows, systemd-inhibit on Linux). A laptop that is closed or runs out of battery will still stop.

### What it runs, in order

1. **Preflight.** Checks the platform, free disk, RAM, every download server, and STEW.
2. **Environment.** Creates `.venv/` with Python 3.12 and `requirements.txt`, via [uv](https://docs.astral.sh/uv/), which is installed into `.pipeline/tools/`. Then checks that every package imports.
3. **Unit tests** of all three projects.
4. **All downloads:** MAT and STEW, the COG-BCI notebook, Shin, COG-BCI, and the COG-BCI event scan.
5. **eeg-load:** R-LMD features, overall and lead-wise BAO + optimized classifiers, the tuning sweep, figures, and `paper_vs_ours.md`.
6. **synthesis:** harmonization of both datasets, `nback_features.csv`, the viability checks, and the R-LMD pilot.
7. **analysis:** split, R-LMD features, the feature models, EEGNet, EDA, the Yedukondalu baselines (both protocols, three tasks), and the evaluation.

Each stage writes its output to the console and to the same log file as the committed one, for example `analysis/results/log_04_baselines.txt`. A run log of everything goes to `.pipeline/logs/`. At the end, the script checks that every result file under the three `results/` folders was rewritten, and lists any that were not.

### Interruptions and options

Finished stages are recorded in `.pipeline/state.json`. **If the run stops for any reason, run the same command again**: it resumes at the stage that did not finish, and most stages also resume within themselves (per subject or per model). Downloads retry and resume on their own.

| Command | Does |
|---|---|
| `python run_pipeline.py --list` | stages, done / to do, typical times |
| `python run_pipeline.py --setup-only` | preflight, environment and tests only (a 5-minute check of a new machine) |
| `python run_pipeline.py --from analysis:03_eda` | reruns that stage and everything after it |
| `python run_pipeline.py --only synthesis:04_viability` | reruns just that stage |
| `python run_pipeline.py --redo` | reruns everything |
| `python run_pipeline.py --cpus 4` | uses at most 4 cores |
| `python run_pipeline.py --stew ~/path/to/STEW\ Dataset.zip` | STEW from somewhere else |
| `python run_pipeline.py --dry-run` | prints the commands without running them |

### How close the numbers will be

Every random step is seeded and every package version is pinned, so a run on another machine reproduces the pipeline exactly. The numbers are close but not always identical in the last digits. ICA, BLAS and GPU kernels round differently across CPUs, and EEGNet trains on whatever is available: CUDA, Apple MPS, or the CPU.

### Not regenerated by the script

- **`docs/research-paper-draft.qmd` / `.pdf`.** The paper is written by hand; its tables are typed into the source. Rendering it needs R, knitr, kableExtra, Quarto and LaTeX, none of which the pipeline uses.
- **`eeg-analysis/`, `eeg-back/`, `gait-analysis/`.** These are the other parent-paper reproductions. Of them, the pipeline uses only `eeg-analysis/eegrepro`'s Shin loader.
- **`synthesis/scripts/shin_nback_signal.py`**, a dataset-selection check in [PROVENANCE.md](synthesis/PROVENANCE.md). It reads the AAR-cleaned Shin data that only `eeg-analysis/`'s own pipeline produces.
- **`analysis/scripts/run_rest.sh` and `analysis/results/log_run_rest.txt`.** This was the earlier shell driver for the analysis stages, superseded by `run_pipeline.py`. It runs the same commands.
