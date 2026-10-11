"""The six "optimizable" classifiers of MATLAB's Classification Learner,
rebuilt on scikit-learn: tree, discriminant, naive Bayes, KNN, SVM and
ensemble. Each is tuned for 30 iterations against the 10-fold cross-validated
error, as the app does, with Optuna's TPE sampler standing in for MATLAB's
Gaussian-process Bayesian optimization. The search spaces follow the app's
defaults. Features are expected to be standardized.
"""
import warnings

import numpy as np
import optuna
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

FAMILIES = ["tree", "discriminant", "naive_bayes", "knn", "svm", "ensemble"]
LABELS = {"tree": "Tree", "discriminant": "Discriminant", "naive_bayes": "Naive Bayes",
          "knn": "KNN", "svm": "SVM", "ensemble": "EL"}
N_TRIALS = 30
N_FOLDS = 10


class DiagLinearDiscriminant(ClassifierMixin, BaseEstimator):
    """MATLAB 'diaglinear': Gaussian classes sharing one diagonal covariance."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        resid = X - self.means_[np.searchsorted(self.classes_, y)]
        self.var_ = resid.var(axis=0) + 1e-9
        self.log_prior_ = np.log(np.array([np.mean(y == c) for c in self.classes_]))
        return self

    def predict_proba(self, X):
        ll = np.array([-0.5 * np.sum((X - m) ** 2 / self.var_, axis=1) for m in self.means_]).T
        ll += self.log_prior_
        ll -= ll.max(axis=1, keepdims=True)
        p = np.exp(ll)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


_KERNELS = {
    "gaussian": lambda u: np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi),
    "box": lambda u: 0.5 * (np.abs(u) <= 1),
    "epanechnikov": lambda u: 0.75 * np.clip(1 - u * u, 0, None),
    "triangle": lambda u: np.clip(1 - np.abs(u), 0, None),
}


class KernelNaiveBayes(ClassifierMixin, BaseEstimator):
    """Naive Bayes with a kernel density per class and feature, bandwidth by
    the normal-reference rule (fitcnb 'DistributionNames','kernel')."""

    def __init__(self, kernel="gaussian"):
        self.kernel = kernel

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.X_ = [X[y == c] for c in self.classes_]
        self.h_ = [x.std(axis=0) * (4 / (3 * len(x))) ** 0.2 + 1e-9 for x in self.X_]
        self.log_prior_ = np.log(np.array([np.mean(y == c) for c in self.classes_]))
        return self

    def predict_proba(self, X):
        kern = _KERNELS[self.kernel]
        ll = np.tile(self.log_prior_, (len(X), 1))
        for ci, (xc, h) in enumerate(zip(self.X_, self.h_)):
            for f in range(X.shape[1]):
                u = (X[:, f, None] - xc[None, :, f]) / h[f]
                ll[:, ci] += np.log(kern(u).mean(axis=1) / h[f] + 1e-300)
        ll -= ll.max(axis=1, keepdims=True)
        p = np.exp(ll)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def squared_inverse(d):
    return 1.0 / (d ** 2 + 1e-12)


def suggest(family, trial, n_train, n_features):
    """Sample one hyperparameter set from the family's search space."""
    s = trial
    if family == "tree":
        return dict(max_splits=s.suggest_int("max_splits", 1, n_train - 1, log=True),
                    criterion=s.suggest_categorical("criterion", ["gini", "entropy"]))
    if family == "discriminant":
        return dict(kind=s.suggest_categorical(
            "kind", ["linear", "quadratic", "diaglinear", "diagquadratic"]))
    if family == "naive_bayes":
        dist = s.suggest_categorical("distribution", ["gaussian", "kernel"])
        p = dict(distribution=dist)
        if dist == "kernel":
            p["kernel"] = s.suggest_categorical("kernel", list(_KERNELS))
        return p
    if family == "knn":
        return dict(k=s.suggest_int("k", 1, n_train // 2, log=True),
                    metric=s.suggest_categorical("metric", ["euclidean", "cityblock", "chebyshev",
                                                            "cosine", "correlation", "cubic"]),
                    weight=s.suggest_categorical("weight", ["equal", "inverse", "squared_inverse"]))
    if family == "svm":
        return dict(kernel=s.suggest_categorical("kernel", ["linear", "quadratic", "cubic", "gaussian"]),
                    C=s.suggest_float("C", 1e-3, 1e3, log=True),
                    kernel_scale=s.suggest_float("kernel_scale", 1e-3, 1e3, log=True))
    if family == "ensemble":
        method = s.suggest_categorical("method", ["Bag", "AdaBoost", "GradBoost"])
        p = dict(method=method,
                 n_learners=s.suggest_int("n_learners", 10, 500, log=True),
                 max_splits=s.suggest_int("max_splits", 1, n_train - 1, log=True))
        if method == "Bag":
            p["n_predictors"] = s.suggest_int("n_predictors", 1, n_features)
        else:
            p["learning_rate"] = s.suggest_float("learning_rate", 1e-3, 1, log=True)
        return p
    raise ValueError(family)


def build(family, p, seed=0):
    """scikit-learn estimator for one hyperparameter set."""
    if family == "tree":
        return DecisionTreeClassifier(max_leaf_nodes=p["max_splits"] + 1,
                                      criterion=p["criterion"], random_state=seed)
    if family == "discriminant":
        return {"linear": LinearDiscriminantAnalysis,
                "quadratic": QuadraticDiscriminantAnalysis,
                "diaglinear": DiagLinearDiscriminant,
                "diagquadratic": GaussianNB}[p["kind"]]()
    if family == "naive_bayes":
        return GaussianNB() if p["distribution"] == "gaussian" else KernelNaiveBayes(p["kernel"])
    if family == "knn":
        metric, kw = p["metric"], {}
        if metric == "cubic":
            metric, kw = "minkowski", dict(p=3)
        weights = {"equal": "uniform", "inverse": "distance",
                   "squared_inverse": squared_inverse}[p["weight"]]
        return KNeighborsClassifier(n_neighbors=p["k"], metric=metric, weights=weights,
                                    algorithm="brute", **kw)
    if family == "svm":
        gamma = 1 / p["kernel_scale"] ** 2
        kernel, degree = {"linear": ("linear", 1), "quadratic": ("poly", 2),
                          "cubic": ("poly", 3), "gaussian": ("rbf", 1)}[p["kernel"]]
        return SVC(kernel=kernel, degree=degree, gamma=gamma, coef0=1, C=p["C"],
                   max_iter=20000, cache_size=500)
    if family == "ensemble":
        leaves = p["max_splits"] + 1
        if p["method"] == "Bag":
            return RandomForestClassifier(n_estimators=p["n_learners"], max_leaf_nodes=leaves,
                                          max_features=p["n_predictors"], random_state=seed)
        if p["method"] == "AdaBoost":
            return AdaBoostClassifier(DecisionTreeClassifier(max_leaf_nodes=leaves, random_state=seed),
                                      n_estimators=p["n_learners"],
                                      learning_rate=p["learning_rate"], random_state=seed)
        return HistGradientBoostingClassifier(max_iter=p["n_learners"], max_leaf_nodes=leaves,
                                              learning_rate=p["learning_rate"],
                                              min_samples_leaf=1, early_stopping=False,
                                              random_state=seed)
    raise ValueError(family)


def _fit_fold(family, p, X, y, tr, te, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build(family, p, seed).fit(X[tr], y[tr])
        pred = model.predict(X[te])
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X[te])[:, 1]
        else:
            score = model.decision_function(X[te])
    return pred, score


def cross_validate(family, p, X, y, folds, seed=0, n_jobs=1):
    """Out-of-fold predictions, scores for class 1 and per-fold accuracy."""
    out = Parallel(n_jobs=n_jobs)(
        delayed(_fit_fold)(family, p, X, y, tr, te, seed) for tr, te in folds)
    pred = np.empty(len(y), dtype=int)
    score = np.empty(len(y))
    fold_acc = []
    for (tr, te), (pr, sc) in zip(folds, out):
        pred[te], score[te] = pr, sc
        fold_acc.append(np.mean(pr == y[te]))
    return pred, np.nan_to_num(score), np.array(fold_acc)


def kfold(X, y, seed=0):
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y))


def subject_kfold(X, y, groups, seed=0):
    return list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y, groups))


def metrics(y, pred):
    """Accuracy, sensitivity (task recall) and specificity (rest recall), in %."""
    tp = np.sum((y == 1) & (pred == 1))
    tn = np.sum((y == 0) & (pred == 0))
    return dict(AC=100 * (tp + tn) / len(y), SE=100 * tp / np.sum(y == 1),
                SP=100 * tn / np.sum(y == 0))


def optimize(family, X, y, n_trials=N_TRIALS, seed=0, n_jobs=1):
    """Tune one classifier family on the 10-fold CV error.

    Returns a dict with the best hyperparameters, its out-of-fold predictions
    and scores, the per-fold accuracies and the running minimum error.
    """
    folds = kfold(X, y, seed)
    n_train = len(folds[0][0])
    best = {"error": np.inf}
    errors = []

    def objective(trial):
        p = suggest(family, trial, n_train, X.shape[1])
        try:
            pred, score, fold_acc = cross_validate(family, p, X, y, folds, seed, n_jobs)
            err = float(np.mean(pred != y))
        except Exception:  # e.g. a singular covariance in QDA
            errors.append(1.0)
            return 1.0
        errors.append(err)
        if err < best["error"]:
            best.update(error=err, params=p, pred=pred, score=score, fold_acc=fold_acc)
        return err

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best["min_error_curve"] = np.minimum.accumulate(errors)
    best.update(metrics(y, best["pred"]))
    return best
