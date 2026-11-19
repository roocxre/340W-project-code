# Machine learning for working-memory load classification

| Directory | Contents |
|---|---|
| `gait-analysis/` | Parent Paper 1 codebase |
| `eeg-analysis/` | Parent Paper 2 codebase (Shin et al. 2018 reproduction) |
| `eeg-load/` | Parent Paper 3 codebase (Yedukondalu et al. 2025 reproduction on MAT and STEW); its `loadrepro` package provides the Yedukondalu baselines in `analysis/` |
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

That single command downloads the datasets, joins them and runs every stage of `synthesis/` and `analysis/`, unattended. It takes about a day on an 8-core machine. Without git, use **Code → Download ZIP** on the GitHub page and run the same command in the unpacked folder; nothing depends on `.git`.

### Data

Both datasets are downloaded automatically, only their n-back files, and every file is checked against its CRC-32 in the source archive:

| Dataset | Source | Used by | Size |
|---|---|---|---|
| Shin 2018, n-back files only | TU Berlin | synthesis, analysis | 2.3 GB |
| COG-BCI v4, n-back files only | Zenodo record 7413650 | synthesis, analysis | 9.4 GB |

### Requirements

| | |
|---|---|
| Operating system | macOS 14+ on Apple Silicon, Linux x86_64 or Windows x86_64. Intel Macs cannot install the pinned PyTorch / numba. |
| Disk | about 30 GB free: 13 GB of raw EEG, 2 GB of harmonized windows, caches, and the environment |
| Memory | 16 GB recommended. Parallel workers are sized to the RAM, so less memory means slower, not failing. |
| Network | only for the download stages at the start (12 GB, typically 1-2 h; Zenodo can be slow) |
| Time | about 20 h on an 8-core Apple M1. `python run_pipeline.py --list` shows the time per stage. |

Keep the machine powered and awake. The script blocks idle sleep while it runs (caffeinate on macOS, SetThreadExecutionState on Windows, systemd-inhibit on Linux). A laptop that is closed or runs out of battery will still stop.

### Windows notes

- **NVIDIA GPU.** PyPI's Windows PyTorch is CPU-only. When `nvidia-smi` is present and the driver supports CUDA 13.2+, the script swaps in the CUDA build of the same version (`torch 2.15.0+cu132`, about 3 GB) so EEGNet trains on the GPU. If that fails, it keeps the CPU build and carries on.
- **Certificates.** Windows adds root certificates only when its own browser or APIs need them, so a fresh PC may lack the root behind TU Berlin's server. The pipeline also trusts Mozilla's root list (`certifi`, inside `.venv`). On a network that intercepts HTTPS (a corporate proxy), set `SSL_CERT_FILE` to the proxy's CA bundle.
- **Use one kind of session.** Start every run either from a normal terminal or from an SSH session, not a mix. `.venv` made from a normal terminal does not start from an administrator SSH session: Windows refuses to follow uv's Python link created at the lower privilege level (`os error 448` / "uv trampoline failed to spawn Python"). Over SSH, which drops the run if the connection drops, start it with a scheduled task or keep the connection alive.

### What it runs, in order

1. **Preflight.** Checks the platform, free disk, RAM and both download servers.
2. **Environment.** Creates `.venv/` with Python 3.12 and `requirements.txt`, via [uv](https://docs.astral.sh/uv/), which is installed into `.pipeline/tools/`. Then checks that every package imports, swaps in CUDA PyTorch on Windows with an NVIDIA GPU, and checks verified HTTPS to both data servers.
3. **Unit tests** of synthesis, analysis and eeg-load's `loadrepro` (the R-LMD port, BAO and the optimizable classifiers).
4. **All downloads:** the COG-BCI notebook, Shin, COG-BCI, and the COG-BCI event scan.
5. **synthesis:** harmonization of both datasets, `nback_features.csv`, the viability checks, and the R-LMD pilot.
6. **analysis:** split, R-LMD features, the feature models, EEGNet, EDA, the Yedukondalu baselines (eeg-load's pipeline run on the joined data, both protocols, three tasks), and the evaluation.

Each stage writes its output to the console and to the same log file as the committed one, for example `analysis/results/log_04_baselines.txt`. A run log of everything goes to `.pipeline/logs/`. At the end, the script checks that every result file under `synthesis/results/` and `analysis/results/` was rewritten, and lists any that were not.

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
| `python run_pipeline.py --dry-run` | prints the commands without running them |

### How close the numbers will be

Every random step is seeded and every package version is pinned, so a run on another machine reproduces the pipeline exactly. The numbers are close but not always identical in the last digits. ICA, BLAS and GPU kernels round differently across CPUs, and EEGNet trains on whatever is available: CUDA, Apple MPS, or the CPU.

### Not regenerated by the script

- **`docs/research-paper-draft.qmd` / `.pdf`.** The paper is written by hand; its tables are typed into the source. Rendering it needs R, knitr, kableExtra, Quarto and LaTeX, none of which the pipeline uses.
- **`eeg-load/results/`, `eeg-analysis/`, `eeg-back/`, `gait-analysis/`.** These are the parent-paper reproductions, including eeg-load's on MAT and STEW. The pipeline uses only their code: `eeg-load/loadrepro` for the R-LMD features and Yedukondalu baselines, and `eeg-analysis/eegrepro` for the Shin loader.
- **`synthesis/scripts/shin_nback_signal.py`**, a dataset-selection check in [PROVENANCE.md](synthesis/PROVENANCE.md). It reads the AAR-cleaned Shin data that only `eeg-analysis/`'s own pipeline produces.
- **`analysis/scripts/run_rest.sh` and `analysis/results/log_run_rest.txt`.** This was the earlier shell driver for the analysis stages, superseded by `run_pipeline.py`. It runs the same commands.
