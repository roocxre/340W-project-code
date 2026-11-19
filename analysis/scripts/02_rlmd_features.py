"""Stage 2: Yedukondalu et al.'s R-LMD features for every window of the joined table.

    caffeinate -i ../synthesis/.venv/bin/python scripts/02_rlmd_features.py [--jobs 8]

Each 4 s window of each of the 26 channels is decomposed by R-LMD into 5 modes (4 product
functions + residual, the paper's level), and each mode gives 6 features: skewness,
kurtosis, mean PSD, fuzzy, Renyi and differential entropy. That is 26 x 5 x 6 = 780
features per window, from eeg-load's port (eeg-load/loadrepro, ported line by line from
the R-LMD authors' MATLAB code).

Input: synthesis/cache/windows/<subject>.npz (the harmonized windows behind the CSV).
Output: cache/rlmd/<subject>.npz with `f` of shape (n_windows, 26, 5, 6) in npz_index
order. One file per subject, so an interrupted run resumes where it stopped.
About 29 ms per channel-window, ~45 min on 8 cores.
"""
import argparse
import os
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "eeg-load"))
from loadrepro import features as lf  # noqa: E402

WINDOWS = REPO / "synthesis" / "cache" / "windows"
OUT = ROOT / "cache" / "rlmd"
FS, LEVEL = 200.0, 5


def subject(path):
    out = OUT / path.name
    if out.exists():
        return path.stem, 0.0
    t0 = time.time()
    x = np.load(path)["x"].astype(float)
    f = np.empty((len(x), x.shape[1], LEVEL, len(lf.FEATURES)), dtype=np.float32)
    for i, w in enumerate(x):
        for c, ch in enumerate(w):
            f[i, c] = lf.signal_features(ch, FS, levels=(LEVEL,))[LEVEL]
    tmp = out.with_name(out.stem + ".part.npz")
    np.savez(tmp, f=f)
    tmp.rename(out)
    return path.stem, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(WINDOWS.glob("*.npz"), key=lambda p: -p.stat().st_size)  # largest first
    print(f"{len(paths)} subjects, {args.jobs} jobs", flush=True)
    t0 = time.time()
    for name, dt in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(subject)(p) for p in paths):
        print(f"{name}: {'cached' if dt == 0 else f'{dt:.0f} s'} (elapsed {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
