"""Binary arithmetic optimization (BAO) wrapper feature selection.

Arithmetic Optimization Algorithm (Abualigah et al. 2021) with a Levy-flight
perturbation (paper Eq. 21-24) and an S-shaped transfer function (Eq. 17-18).
Each agent holds a binary feature mask. A candidate is built by applying the
AOA update to the best mask, read as a 0/1 position, and drawing a new mask
through the transfer function, so bits flip often early on and rarely near
the end. Fitness is Eq. 25 with a KNN wrapper.

The paper does not give the population size, the wrapper classifier or how
its error is estimated. Reverse-engineered choices, see the README:
- 30 agents, KNN with k = 5, weight 0.99 on the error.
- The error is 5-fold cross-validated. Fig. 3a of the paper suggests a 20 %
  hold-out instead (its steps equal one segment of a 216-segment hold-out),
  available as Fitness(holdout=0.2), but selection on a hold-out that small
  does not find the informative features in our feature space.
"""
from math import gamma, pi, sin

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

MOA_MIN, MOA_MAX = 0.2, 1.0
ALPHA = 5.0     # MOP sensitivity ("beta" in the paper's Eq. 15)
MU = 0.499      # AOA search-control parameter
LEVY_ETA = 1.5
LEVY_SCALE = 0.01
TRANSFER_SLOPE = 10.0


class Fitness:
    """Eq. 25: w * error + (1 - w) * |selected| / |all|, with the KNN error
    cross-validated over `folds` folds, or measured on one stratified
    hold-out when `holdout` is given."""

    def __init__(self, X, y, k=5, folds=5, holdout=None, w=0.99, seed=0):
        if holdout:
            cv = StratifiedShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
        else:
            cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
        self.splits = list(cv.split(X, y))
        self.X, self.y, self.k, self.w = X, y, k, w

    def error(self, mask):
        if not mask.any():
            return 1.0
        X = self.X[:, mask]
        wrong = n = 0
        for tr, te in self.splits:
            knn = KNeighborsClassifier(n_neighbors=self.k, algorithm="brute")
            wrong += np.sum(knn.fit(X[tr], self.y[tr]).predict(X[te]) != self.y[te])
            n += len(te)
        return wrong / n

    def __call__(self, mask):
        return self.w * self.error(mask) + (1 - self.w) * mask.sum() / len(mask)


def levy(rng, shape, eta=LEVY_ETA):
    """Mantegna's algorithm for Levy-stable steps (Eq. 22-24)."""
    sigma = (gamma(1 + eta) * sin(pi * eta / 2)
             / (gamma((1 + eta) / 2) * eta * 2 ** ((eta - 1) / 2))) ** (1 / eta)
    u = rng.normal(0, sigma, shape)
    v = rng.normal(0, 1, shape)
    return u / np.abs(v) ** (1 / eta)


def binarize(pos, rng):
    prob = 1 / (1 + np.exp(-TRANSFER_SLOPE * (pos - 0.5)))
    return rng.random(pos.shape) < prob


def bao(fitness, n_features, n_agents=30, n_iter=100, seed=0):
    """Run BAO. Returns (best_mask, curve) with curve[t] the best fitness
    after iteration t + 1."""
    rng = np.random.default_rng(seed)
    lb, ub = 0.0, 1.0
    masks = binarize(rng.random((n_agents, n_features)), rng)
    fit = np.array([fitness(m) for m in masks])
    b = int(np.argmin(fit))
    best_mask, best_fit = masks[b].copy(), fit[b]
    curve = np.empty(n_iter)
    scale = (ub - lb) * MU + lb
    eps = np.finfo(float).eps
    for t in range(1, n_iter + 1):
        moa = MOA_MIN + t * (MOA_MAX - MOA_MIN) / n_iter
        mop = 1 - t ** (1 / ALPHA) / n_iter ** (1 / ALPHA)
        best_pos = best_mask.astype(float)
        for a in range(n_agents):
            r1, r2, r3 = rng.random((3, n_features))
            explore = np.where(r2 < 0.5, best_pos / (mop + eps) * scale, best_pos * mop * scale)
            exploit = np.where(r3 < 0.5, best_pos - mop * scale, best_pos + mop * scale)
            new = np.where(r1 > moa, explore, exploit)
            step = rng.random(n_features) * np.sign(rng.random(n_features) - 0.5)
            new = np.clip(new + LEVY_SCALE * step * levy(rng, n_features), lb, ub)
            mask = binarize(new, rng)
            f = fitness(mask)
            if f < fit[a]:
                masks[a], fit[a] = mask, f
                if f < best_fit:
                    best_mask, best_fit = mask.copy(), f
        curve[t - 1] = best_fit
    return best_mask, curve
