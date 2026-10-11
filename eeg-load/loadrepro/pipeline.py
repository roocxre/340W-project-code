"""Feature matrix assembly and the BAO -> optimized-classifier chain shared
by the overall, lead-wise and tuning stages."""
import numpy as np

from . import bao, classifiers, data, features

SEED = 0


def load_features(dataset, level=5, leads=None, standardize=True):
    """Feature matrix of a dataset, z-scored per column unless standardize
    is False.

    Returns (X, y, subj, names). X has one column per lead, mode and feature;
    leads restricts it to the given lead names.
    """
    cfg = data.DATASETS[dataset]
    z = np.load(data.CACHE_DIR / f"features_{dataset}.npz")
    F = z[f"F{level}"]
    clab = cfg["clab"]
    if leads is not None:
        F = F[:, [clab.index(c) for c in leads]]
        clab = list(leads)
    X = F.reshape(len(F), -1)
    if standardize:
        sd = X.std(axis=0)
        X = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1)
    return X, z["y"], z["subj"], features.feature_names(clab, level)


def select(X, y, n_iter=100, seed=SEED):
    """BAO feature selection. Returns (mask, fitness curve)."""
    fitness = bao.Fitness(X, y, seed=seed)
    return bao.bao(fitness, X.shape[1], n_iter=n_iter, seed=seed)


def tune(X, y, subj, family, seed=SEED, n_jobs=1):
    """Tune one classifier family (paper protocol: segment-level 10-fold CV)
    and re-score its best hyperparameters with subject-wise 10-fold CV."""
    res = classifiers.optimize(family, X, y, seed=seed, n_jobs=n_jobs)
    folds = classifiers.subject_kfold(X, y, subj, seed)
    pred, _, _ = classifiers.cross_validate(family, res["params"], X, y, folds, seed, n_jobs)
    res["subject_cv"] = classifiers.metrics(y, pred)
    return res
