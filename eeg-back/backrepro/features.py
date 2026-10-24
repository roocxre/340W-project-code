"""Segment features for the Table 3 baselines.

The paper names its classifiers (MLP, XGBoost) but not its features, window
length or validation, so these are standard choices: non-overlapping 2 s
windows, as in the released compute_topomap, and

- "bandpower": log10 absolute power in the released DEFAULT_BANDS (delta,
  theta, alpha, beta, gamma, 1-40 Hz) per channel, 80 features;
- "extended": bandpower plus relative band power and six time-domain
  descriptors per channel (log variance, skewness, kurtosis, Hjorth mobility
  and complexity, log line length), 256 features.
"""

import numpy as np
from scipy import signal, stats

from . import data
from .released import psd_analysis as pa

BANDS = [v[:2] for v in pa.DEFAULT_BANDS.values()]


def segment_features(x, fs=data.FS):
    """Features of one window, x: (channels, samples) in µV."""
    f, P = signal.welch(x, fs, nperseg=x.shape[1])
    bp = np.stack([np.trapezoid(P[:, (f >= lo) & (f <= hi)], f[(f >= lo) & (f <= hi)], axis=1) for lo, hi in BANDS], 1)
    total = np.trapezoid(P[:, (f >= 1) & (f <= 40)], f[(f >= 1) & (f <= 40)], axis=1)
    d1, d2 = np.diff(x, axis=1), np.diff(x, 2, axis=1)
    v0, v1, v2 = x.var(1), d1.var(1), d2.var(1)
    mob = np.sqrt(v1 / v0)
    td = np.stack([np.log10(v0), stats.skew(x, axis=1), stats.kurtosis(x, axis=1), mob,
                   np.sqrt(v2 / v1) / mob, np.log10(np.abs(d1).sum(1))], 1)
    return {"bandpower": np.log10(bp).ravel(),
            "extended": np.concatenate([np.log10(bp).ravel(), (bp / total[:, None]).ravel(), td.ravel()])}


def build(window_s=2.0):
    """Feature matrices over all recordings, with labels and subject ids."""
    n = int(window_s * data.FS)
    feats = {"bandpower": [], "extended": []}
    cond, subj = [], []
    for s, c in data.iter_files():
        x = data.load_array(s, c)
        for k in range(x.shape[1] // n):
            for name, v in segment_features(x[:, k * n:(k + 1) * n]).items():
                feats[name].append(v)
            cond.append(c)
            subj.append(s)
    return {k: np.array(v) for k, v in feats.items()}, np.array(cond), np.array(subj)
