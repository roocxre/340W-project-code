"""Stage 2: harmonize every session and cut task windows.

Writes cache/windows/<subject>.npz with
  x     (n_windows, 26, 800) float32, µV, 200 Hz, the 26 shared channels
  meta  JSON list of per-window metadata
  qa    JSON list of per-session preprocessing QA
COG-BCI subjects are streamed from Zenodo and their raw files deleted after use.

usage: 02_harmonize.py [shin|cog|all] [--jobs N] [--subjects ...]
"""
import argparse
import json
import os
import sys

# one BLAS thread per worker; parallelism comes from --jobs
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nbackjoin import features, loaders, preprocess, sources

OUT = sources.CACHE / "windows"


def process(dataset, subject):
    name = f"shin-{subject}" if dataset == "shin" else f"cog-{subject:02d}"
    dest = OUT / f"{name}.npz"
    if dest.exists():
        return name, "cached"
    sessions = loaders.shin_session_runs(subject) if dataset == "shin" else loaders.cog_subject(subject)
    interp_cz = dataset == "cog" and subject in sources.COG_NO_CZ
    xs, meta, qa = [], [], []
    for runs in sessions:
        clean, q = preprocess.harmonize(runs, interpolate_cz=interp_cz)
        qa.append(dict(dataset=dataset, subject=runs[0].subject, session=runs[0].session, **q))
        for run, c in zip(runs, clean):
            for x, m in features.windows(c, run):
                xs.append(x)
                meta.append(dict(dataset=dataset, subject=run.subject, session=run.session, run=run.label,
                                 n=run.blocks[m["block_in_run"] - 1].n, cz_interpolated=interp_cz,
                                 n_bad_channels=q["n_bad_channels"], n_ica_removed=q["n_ica_removed"], **m))
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.npz")
    np.savez(tmp, x=np.stack(xs), meta=json.dumps(meta), qa=json.dumps(qa))
    tmp.rename(dest)
    return name, f"{len(xs)} windows"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["shin", "cog", "all"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--subjects", nargs="*")
    a = ap.parse_args()
    todo = []
    if a.dataset in ("shin", "all"):
        todo += [("shin", s) for s in (a.subjects or sources.SHIN_SUBJECTS)]
    if a.dataset in ("cog", "all"):
        todo += [("cog", int(s)) for s in (a.subjects or sources.COG_SUBJECTS)]
    sources.CACHE.mkdir(exist_ok=True)
    failed = []
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(process, d, s): (d, s) for d, s in todo}
        for f in as_completed(futs):
            try:
                print(*f.result(), flush=True)
            except Exception:
                failed.append(futs[f])
                print("FAILED", futs[f], traceback.format_exc(), flush=True)
    print("failed:", failed)


if __name__ == "__main__":
    main()
