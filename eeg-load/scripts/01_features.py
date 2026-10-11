"""Stage 1: segmentation, R-LMD decomposition and feature extraction.

per subject: preprocess -> 4 s segments -> per lead R-LMD -> 6 features/mode
          -> cache/features_<dataset>.npz

The cache holds, for each decomposition level L, an array F<L> of shape
(n_segments, n_leads, L, 6), plus the labels (0 rest, 1 task) and subject
index of every segment. MAT is decomposed at L = 3, 5, 7, 9 (Fig. 9 of the
paper); STEW only at L = 5.
"""
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loadrepro import data, features  # noqa: E402

LEVELS = {"mat": (3, 5, 7, 9), "stew": (5,)}


def process(dataset, subject):
    cfg = data.DATASETS[dataset]
    segs, labels = data.load_segments(dataset, subject)
    levels = LEVELS[dataset]
    out = {L: np.zeros((len(segs), segs.shape[2], L, len(features.FEATURES))) for L in levels}
    for i, seg in enumerate(segs):
        for c in range(seg.shape[1]):
            f = features.signal_features(seg[:, c], cfg["fs"], levels)
            for L in levels:
                out[L][i, c] = f[L]
    return out, labels


def main(dataset):
    cfg = data.DATASETS[dataset]
    t0 = time.time()
    res = Parallel(n_jobs=-1, verbose=5)(
        delayed(process)(dataset, s) for s in cfg["subjects"])
    arrays = {f"F{L}": np.concatenate([r[0][L] for r in res]) for L in LEVELS[dataset]}
    y = np.concatenate([r[1] for r in res])
    subj = np.concatenate([np.full(len(r[1]), i) for i, r in enumerate(res)])
    data.CACHE_DIR.mkdir(exist_ok=True)
    np.savez(data.CACHE_DIR / f"features_{dataset}.npz", y=y, subj=subj, **arrays)
    f5 = arrays["F5"]
    print(f"{dataset}: {len(y)} segments ({(y == 0).sum()} rest, {(y == 1).sum()} task), "
          f"F5 {f5.shape}, non-finite {np.sum(~np.isfinite(f5))}, "
          f"empty modes {np.mean(np.all(f5 == 0, axis=-1)):.3f}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["mat", "stew"]):
        main(ds)
