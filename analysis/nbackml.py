"""Shared pieces of the modelling stages: tasks, feature sets, normalization, metrics.

Tasks (from docs/research-paper-draft.qmd):
    bin    0- vs 2-back, both datasets pooled (the levels they share): load present or not
    shin3  Shin 0 / 2 / 3-back: levels of load
    cog3   COG-BCI 0 / 1 / 2-back: levels of load

Every task uses clean windows only (artifact == False). Labels are 0 .. k-1 in level order.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import nbacksplit as ns

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
PRED = RESULTS / "predictions"
FIG = RESULTS / "figures"
SEED = 0

TASKS = {
    "bin": dict(datasets=("shin", "cog"), levels=(0, 2), title="0 vs 2-back, pooled"),
    "shin3": dict(datasets=("shin",), levels=(0, 2, 3), title="Shin 0 / 2 / 3-back"),
    "cog3": dict(datasets=("cog",), levels=(0, 1, 2), title="COG-BCI 0 / 1 / 2-back"),
}


def frame(split, task):
    """Clean windows of one split restricted to the task's datasets and levels."""
    t = TASKS[task]
    df = ns.load(split, clean=True)
    df = df[df.dataset.isin(t["datasets"]) & df.n.isin(t["levels"])]
    return df.sort_values(["subject", "npz_index"]).reset_index(drop=True)


def labels(df, task):
    return np.searchsorted(np.array(TASKS[task]["levels"]), df.n.to_numpy())


def bandpower(df):
    return df[ns.feature_columns(df)].to_numpy(float)


def rlmd(df):
    """R-LMD features (stage 2), (n, 780) in eeg-load's order: channel, mode, feature."""
    out = np.empty((len(df), 26 * 5 * 6))
    for subject, idx in df.groupby("subject").indices.items():
        f = np.load(CACHE / "rlmd" / f"{subject}.npz")["f"]
        out[idx] = f[df.npz_index.to_numpy()[idx]].reshape(len(idx), -1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def zscore_subject(X, groups):
    """Each subject's rows z-scored over that subject's rows. Uses no labels."""
    X = np.asarray(X, float)
    out = np.empty_like(X)
    for g in np.unique(groups):
        m = groups == g
        out[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-12)
    return out


def subject_folds(y, groups, k=10, seed=SEED):
    """Subject-wise stratified k-fold: no subject in both train and test."""
    return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))


def window_folds(y, k=10, seed=SEED):
    """Window-level stratified k-fold, as in Yedukondalu et al. (leaks subjects and recordings)."""
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def metrics(y, proba):
    """Accuracy, balanced accuracy, Cohen's kappa and AUC (macro one-vs-rest), in %."""
    pred = proba.argmax(1)
    k = proba.shape[1]
    auc = roc_auc_score(y, proba[:, 1]) if k == 2 else roc_auc_score(y, proba, multi_class="ovr")
    return dict(acc=100 * accuracy_score(y, pred), bacc=100 * balanced_accuracy_score(y, pred),
                kappa=100 * cohen_kappa_score(y, pred), auc=100 * auc)


def per_subject_acc(y, proba, groups):
    return pd.Series(proba.argmax(1) == y).groupby(np.asarray(groups)).mean() * 100


def smooth(proba, df, k=5):
    """Mean probability over each window and up to k - 1 preceding windows of the same block.

    k = 5 is a 20 s causal decision; it never crosses a block, run or subject boundary.
    """
    key = df.subject + "|" + df.run.astype(str) + "|" + df.block_in_run.astype(str)
    out = np.empty_like(proba)
    for _, idx in pd.Series(np.arange(len(df))).groupby(key.to_numpy()).indices.items():
        idx = idx[np.argsort(df.window.to_numpy()[idx])]
        c = np.cumsum(proba[idx], 0)
        c = np.vstack([np.zeros((1, proba.shape[1])), c])
        j = np.arange(1, len(idx) + 1)
        lo = np.maximum(j - k, 0)
        out[idx] = (c[j] - c[lo]) / (j - lo)[:, None]
    return out


def save_predictions(task, model, **arrays):
    PRED.mkdir(parents=True, exist_ok=True)
    np.savez(PRED / f"{task}__{model}.npz", **arrays)


def confusion(y, proba, k):
    return confusion_matrix(y, proba.argmax(1), labels=list(range(k)), normalize="true")


# ---------------------------------------------------------------- Riemannian features
BANDS_FB = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def _sumzero_basis(n=26):
    """Orthonormal basis (n, n - 1) of the space orthogonal to the all-ones vector.

    The windows are common-average referenced, so their covariance has rank n - 1;
    projecting onto this basis gives full-rank (n - 1) x (n - 1) matrices losslessly.
    """
    q, _ = np.linalg.qr(np.column_stack([np.ones(n), np.eye(n)[:, : n - 1]]))
    return q[:, 1:]


def covariances(subject):
    """Band-filtered covariance matrices of all of a subject's windows, cached.

    Returns (n_windows, 3, 25, 25) in npz_index order: theta, alpha, beta;
    4th-order zero-phase Butterworth per window, OAS shrinkage.
    """
    from pyriemann.estimation import Covariances
    from scipy.signal import butter, sosfiltfilt
    path = CACHE / "cov" / f"{subject}.npz"
    if path.exists():
        return np.load(path)["C"]
    x = np.load(ns.WINDOWS / f"{subject}.npz")["x"].astype(float)
    x = np.einsum("cd,ncs->nds", _sumzero_basis(x.shape[1]), x)
    C = np.empty((len(x), len(BANDS_FB), x.shape[1], x.shape[1]))
    for b, (lo, hi) in enumerate(BANDS_FB.values()):
        xf = sosfiltfilt(butter(4, [lo, hi], btype="band", fs=200.0, output="sos"), x, axis=-1)
        C[:, b] = Covariances("oas").transform(xf)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, C=C.astype(np.float32))
    return C


def tangent_recentred(df):
    """Filter-bank tangent-space features, each subject re-centred to its own Riemannian mean.

    For each subject and band, C -> M^-1/2 C M^-1/2 with M the subject's mean covariance
    (no labels), then the log map at the identity: 3 x 325 = 975 features.
    """
    from pyriemann.geometry.base import invsqrtm, logm
    from pyriemann.geometry.mean import mean_riemann
    from pyriemann.geometry.tangentspace import upper
    out = None
    for subject, idx in df.groupby("subject").indices.items():
        C = covariances(subject)[df.npz_index.to_numpy()[idx]].astype(float)
        feats = []
        for b in range(C.shape[1]):
            W = invsqrtm(mean_riemann(C[:, b]))
            feats.append(upper(logm(W @ C[:, b] @ W)))
        f = np.hstack(feats)
        if out is None:
            out = np.empty((len(df), f.shape[1]))
        out[idx] = f
    return out
