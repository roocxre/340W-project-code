"""Windows within task blocks, and spectral features per window.

Windows are 4 s, non-overlapping, starting at a block's first trial onset and ending
before its last trial onset + SOA, so instructions and rest are excluded. Trial onsets
come from markers, which absorbs Shin's ~2.2 s and COG-BCI's 2.0 / 2.5 s SOAs.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from . import sources

WIN_S = 4.0
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
PTP_REJECT_UV = 150.0


def windows(clean, run):
    """Yield (x, meta) per window: x is (26, 800) float32 in µV."""
    fs = clean.info["sfreq"]
    native = run.raw.info["sfreq"]
    data = clean.get_data() * 1e6
    nwin = int(WIN_S * fs)
    for bi, b in enumerate(run.blocks):
        on = b.onsets / native
        t0, t_end = on[0], on[-1] + b.soa
        for w in range(int((t_end - t0) // WIN_S)):
            ws = t0 + w * WIN_S
            a = int(round(ws * fs))
            x = data[:, a:a + nwin]
            if x.shape[1] < nwin:
                break
            in_w = (on >= ws) & (on < ws + WIN_S)
            yield x.astype(np.float32), dict(
                block_in_run=bi + 1, window=w + 1, t_in_block_s=round(w * WIN_S, 1),
                # Shin's 0-back markers do not distinguish targets
                n_trials=int(in_w.sum()),
                n_targets=int(b.is_target[in_w].sum()) if b.n > 0 or run.dataset == "cog" else np.nan,
                soa_s=round(b.soa, 3), block_accuracy=round(b.accuracy, 4), block_mean_rt_s=round(b.mean_rt, 4),
                ptp_max_uv=round(float(np.ptp(x, axis=1).max()), 1))


def spectral(x, fs=sources.FS):
    """Log10 absolute band power (µV²) and relative band power (share of 1-40 Hz) per channel.

    x: (n_windows, n_channels, n_samples). Welch with 2 s Hann segments, 50 % overlap.
    Returns (n_windows, n_channels, n_bands) arrays: log_abs, rel.
    """
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), noverlap=int(fs), axis=-1)
    df = f[1] - f[0]
    bp = np.stack([p[..., (f >= lo) & (f < hi)].sum(-1) * df for lo, hi in BANDS.values()], -1)
    total = p[..., (f >= 1) & (f < 40)].sum(-1, keepdims=True) * df
    return np.log10(bp), bp / total


def feature_columns():
    abs_cols = [f"abs_{b}_{c}" for c in sources.CHANNELS for b in BANDS]
    rel_cols = [f"rel_{b}_{c}" for c in sources.CHANNELS for b in BANDS]
    return abs_cols, rel_cols
