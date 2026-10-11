"""Multi-domain features of R-LMD modes: skewness, kurtosis, PSD, fuzzy
entropy, Renyi entropy and differential entropy (6 per mode)."""
import numpy as np
from numba import njit
from scipy import signal

from . import rlmd

FEATURES = ["SN", "KT", "PSD", "FE", "RE", "DE"]


@njit(cache=True)
def fuzzy_entropy(x, m=2, r_factor=0.2, n=2.0):
    """Fuzzy entropy (Chen et al. 2007): baseline-removed templates, Chebyshev
    distance, similarity exp(-d^n / r) with r = r_factor * std(x)."""
    N = len(x)
    r = r_factor * np.std(x)
    if r == 0:
        return 0.0
    cnt = N - m
    phi_m = 0.0
    phi_m1 = 0.0
    for i in range(cnt):
        bi = 0.0
        for k in range(m):
            bi += x[i + k]
        bi1 = (bi + x[i + m]) / (m + 1)
        bi /= m
        for j in range(i + 1, cnt):
            bj = 0.0
            for k in range(m):
                bj += x[j + k]
            bj1 = (bj + x[j + m]) / (m + 1)
            bj /= m
            d = 0.0
            d1 = 0.0
            for k in range(m):
                v = abs((x[i + k] - bi) - (x[j + k] - bj))
                if v > d:
                    d = v
                v = abs((x[i + k] - bi1) - (x[j + k] - bj1))
                if v > d1:
                    d1 = v
            v = abs((x[i + m] - bi1) - (x[j + m] - bj1))
            if v > d1:
                d1 = v
            phi_m += np.exp(-(d ** n) / r)
            phi_m1 += np.exp(-(d1 ** n) / r)
    if phi_m <= 0 or phi_m1 <= 0:
        return 0.0
    return np.log(phi_m) - np.log(phi_m1)


@njit(cache=True)
def renyi_entropy(x):
    """Renyi quadratic entropy from the Parzen information potential (Eq. 9),
    Gaussian kernel with Silverman's bandwidth."""
    N = len(x)
    sigma = 1.06 * np.std(x) * N ** (-0.2)
    if sigma == 0:
        return 0.0
    s2 = 2.0 * sigma * sigma  # variance of the convolved kernel
    acc = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = x[i] - x[j]
            acc += np.exp(-d * d / (2.0 * s2))
    ip = (2.0 * acc + N) / (N * N * np.sqrt(2.0 * np.pi * s2))
    return -np.log(ip)


def mode_features(x, fs):
    """The 6 features of one mode; all zero for a constant (empty) mode."""
    x = np.ascontiguousarray(x, dtype=float)
    c = x - x.mean()
    var = np.mean(c ** 2)
    if var <= 1e-20:
        return np.zeros(len(FEATURES))
    sn = np.mean(c ** 3) / var ** 1.5
    kt = np.mean(c ** 4) / var ** 2
    # pwelch defaults: 8 Hamming segments with 50 % overlap.
    nperseg = int(len(x) / 4.5)
    _, pxx = signal.welch(x, fs=fs, window="hamming", nperseg=nperseg,
                          noverlap=nperseg // 2, detrend=False)
    psd = pxx.mean()
    fe = fuzzy_entropy(x)
    re = renyi_entropy(x)
    de = 0.5 * np.log(2 * np.pi * np.e * var)
    return np.array([sn, kt, psd, fe, re, de])


def signal_features(x, fs, levels=(5,)):
    """Features of the L-level decomposition of x for each L in levels.

    An L-level decomposition is L - 1 product functions plus the residual.
    Returns {L: array (L, 6)}. The product functions are computed once at the
    deepest level and shared, since PF k does not depend on how many follow.
    """
    pfs, _, _ = rlmd.rlmd(x, max_pfs=max(levels) - 1)
    pf, n_pf = pfs[:-1], len(pfs) - 1
    pf_feats = [None] * n_pf
    out = {}
    for L in levels:
        k = min(L - 1, n_pf)
        f = np.zeros((L, len(FEATURES)))
        for i in range(k):
            if pf_feats[i] is None:
                pf_feats[i] = mode_features(pf[i], fs)
            f[i] = pf_feats[i]
        f[k] = mode_features(x - pf[:k].sum(axis=0), fs)
        out[L] = f
    return out


def feature_names(clab, n_modes):
    return [f"{c}_M{m + 1}_{f}" for c in clab for m in range(n_modes) for f in FEATURES]
