"""Harmonization: one pipeline for both datasets, applied per recording session.

1. 1 Hz high-pass (removes COG-BCI's DC offsets), 50 Hz notch and harmonics below Nyquist.
2. Resample to 200 Hz (COG-BCI is recorded at 500 Hz; Shin is distributed at 200 Hz).
3. Broken channels only: robust spread < 0.1x or > 10x the median channel, or max |r| with
   any other channel < 0.2 (1-40 Hz, recording reference); spherical-spline interpolation.
4. Common average reference over the native EEG channels (ICLabel expects it).
5. ICA (extended Infomax via Picard) on the session, ICLabel, and removal of components
   labelled eye, muscle or heart with probability > 0.9 (COG-BCI's published criterion).
   No EOG channel is used, because COG-BCI has none.
6. 40 Hz low-pass.
7. Cz interpolated where it was not recorded (COG-BCI 1-9), the 26 shared channels
   picked, and common average reference over those 26.
"""
from __future__ import annotations

import mne
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel.iclabel import iclabel_label_components

from . import sources

ICLABEL_CLASSES = ["brain", "muscle", "eye", "heart", "line", "channel", "other"]
REMOVE = ("muscle", "eye", "heart")
P_REMOVE = 0.9
MONTAGE = mne.channels.make_standard_montage("standard_1005")


def _bad_channels(raws):
    """Channels that are broken rather than merely contaminated.

    On 1-40 Hz data in the recording reference, with robust spread = 1.4826 x MAD, a channel
    is bad if its spread is below 0.1x or above 10x the median channel, or if its largest |r|
    with any other channel is below 0.2.

    The thresholds are deliberately loose; contamination is left to ICA and the per-window
    artifact flag. Measured on the data:
    - ocular activity puts Shin's Fp1/Fp2 at up to 4.2x the median spread (so a 5x rule
      interpolated real frontal EEG in 6 subjects);
    - COG-BCI's Fp1 sits next to the Fpz reference: spread 0.4x and max |r| as low as 0.29;
    - re-referenced alternatives fail: an average reference spreads a bad channel into all
      others, and a median reference strips Cz (it tracks the per-sample median), which made
      Cz look decorrelated in 4 Shin subjects.
    """
    rr = [r.copy().pick("eeg").filter(None, 40.0) for r in raws]
    x = np.concatenate([r.get_data()[:, ::4] for r in rr], axis=1)
    names = rr[0].ch_names
    s = 1.4826 * np.median(np.abs(x - np.median(x, 1, keepdims=True)), 1)
    ratio = s / np.median(s)
    c = np.corrcoef(x)
    np.fill_diagonal(c, 0)
    max_r = np.abs(np.nan_to_num(c)).max(1)
    return [ch for ch, q, m in zip(names, ratio, max_r) if q < 0.1 or q > 10 or m < 0.2]


def harmonize(runs, interpolate_cz=False):
    """Clean one session's runs. Returns 26-channel 200 Hz raws (same order as ``runs``)
    and a QA dict."""
    raws = []
    for run in runs:
        raw = run.raw.copy()
        raw.set_montage(MONTAGE, on_missing="raise")
        raw.filter(1.0, None, picks="all", phase="zero")
        nyq = raw.info["sfreq"] / 2
        raw.notch_filter(np.arange(sources.LINE, nyq - 1, sources.LINE), picks="all")
        if raw.info["sfreq"] != sources.FS:
            raw.resample(sources.FS)
        raws.append(raw)

    bads = _bad_channels(raws)
    for raw in raws:
        raw.info["bads"] = list(bads)
        if bads:
            raw.interpolate_bads(reset_bads=True)
        raw.set_eeg_reference("average", projection=False)

    eog = "VEOG" in raws[0].ch_names
    fp_before = _fp_eog_corr(raws) if eog else np.nan

    fit = mne.concatenate_raws([r.copy().pick("eeg") for r in raws]) if len(raws) > 1 else raws[0].copy().pick("eeg")
    n_eeg = len(fit.ch_names)
    ica = ICA(n_components=n_eeg - 1 - len(bads), method="picard",
              fit_params=dict(ortho=False, extended=True), max_iter=1000, random_state=97)
    ica.fit(fit)
    proba = iclabel_label_components(fit, ica)
    top = proba.argmax(1)
    exclude = [i for i, p in enumerate(proba)
               if any(p[ICLABEL_CLASSES.index(c)] > P_REMOVE for c in REMOVE)]
    ica.exclude = exclude

    for raw in raws:
        ica.apply(raw)
    fp_after = _fp_eog_corr(raws) if eog else np.nan
    out = [raw.filter(None, 40.0, picks="eeg", phase="zero") for raw in raws]

    final = []
    for raw in out:
        raw = raw.pick("eeg")
        if interpolate_cz:
            cz = mne.io.RawArray(np.zeros((1, raw.n_times)),
                                 mne.create_info(["Cz"], raw.info["sfreq"], "eeg"), verbose=False)
            raw.add_channels([cz], force_update_info=True)
            raw.set_montage(MONTAGE)
            raw.info["bads"] = ["Cz"]
            raw.interpolate_bads(reset_bads=True)
        raw.pick(sources.CHANNELS)
        raw.set_eeg_reference("average", projection=False)
        final.append(raw)

    counts = {c: int(np.sum(top[exclude] == ICLABEL_CLASSES.index(c))) for c in REMOVE}
    qa = dict(n_bad_channels=len(bads), bad_channels=" ".join(bads), n_ica=ica.n_components_,
              n_ica_removed=len(exclude), **{f"n_ica_{c}": v for c, v in counts.items()},
              fp_veog_corr_before=fp_before, fp_veog_corr_after=fp_after)
    return final, qa


def _fp_eog_corr(raws):
    """Mean |r| between Fp1/Fp2 and VEOG (Shin only; a check that ocular activity was removed)."""
    x = np.concatenate([r.get_data(picks=["Fp1", "Fp2", "VEOG"]) for r in raws], axis=1)
    v = x[2] - x[2].mean()
    return float(np.mean([abs(np.corrcoef(x[i], v)[0, 1]) for i in (0, 1)]))
