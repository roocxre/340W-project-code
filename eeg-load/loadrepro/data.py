"""Dataset loading, preprocessing and 4 s segmentation for MAT and STEW.

Both loaders return a list of Recording (one per subject and condition) with
the signal in microvolts, shape (n_samples, n_channels).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

SEG_SEC = 4

# Channel order as stored in the EDF files (19 EEG leads; A2-A1 and ECG dropped).
MAT_CLAB = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "C3", "C4",
            "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz"]
MAT_FS = 500
MAT_SUBJECTS = [f"Subject{i:02d}" for i in range(36)]
# Fig. 4b of the paper has 540 segments per class = 36 subjects x 15 segments,
# so 60 s of each recording is used.
MAT_SEGS = 15

# Emotiv EPOC column order of the STEW text files.
STEW_CLAB = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8",
             "FC6", "F4", "F8", "AF4"]
STEW_FS = 128
STEW_SUBJECTS = [f"sub{i:02d}" for i in range(1, 49)]
# Fig. 5b has 3552 segments = 48 subjects x 2 x 37 (150 s -> 37 full segments).
STEW_SEGS = 37

DATASETS = {
    "mat": dict(clab=MAT_CLAB, fs=MAT_FS, subjects=MAT_SUBJECTS, segs=MAT_SEGS),
    "stew": dict(clab=STEW_CLAB, fs=STEW_FS, subjects=STEW_SUBJECTS, segs=STEW_SEGS),
}


@dataclass
class Recording:
    subject: str
    label: int  # 0 = rest / low load, 1 = task / high load
    x: np.ndarray  # (n_samples, n_channels), microvolts
    fs: int


def load_mat(subject):
    """Rest (_1) and task (_2) recordings of one MAT subject.

    The EDF files are already filtered by the dataset authors (0.5-45 Hz,
    50 Hz notch, ICA), which is the preprocessing the paper cites, so no
    further filtering is applied.
    """
    import mne
    out = []
    for suffix, label in (("1", 0), ("2", 1)):
        raw = mne.io.read_raw_edf(DATA_DIR / "eegmat" / f"{subject}_{suffix}.edf",
                                  preload=True, verbose="ERROR")
        x = raw.get_data(picks=[f"EEG {c}" for c in MAT_CLAB]).T * 1e6
        out.append(Recording(subject, label, x, MAT_FS))
    return out


def _patch_asrpy():
    """asrpy 0.0.3 uses a 1-element array (np.diff of two quantiles) as a
    scalar, which numpy >= 2.4 rejects. Give its asr_utils module a numpy
    whose diff() unwraps 1-element results."""
    import types

    import asrpy.asr_utils as au
    if getattr(au.np, "_loadrepro_patched", False):
        return

    def _diff(a, *args, **kw):
        r = np.diff(a, *args, **kw)
        return r.item() if r.size == 1 else r

    shim = types.ModuleType("numpy_shim")
    shim.__dict__.update(np.__dict__)
    shim.diff = _diff
    shim._loadrepro_patched = True
    au.np = shim


def _stew_preprocess(x, fs, asr=True):
    """STEW paper preprocessing: 1 Hz high-pass, line-noise removal, ASR,
    average reference."""
    sos = signal.butter(4, 1, "highpass", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, x, axis=0)
    b, a = signal.iirnotch(50, 30, fs=fs)
    x = signal.filtfilt(b, a, x, axis=0)
    if asr:
        import asrpy
        import mne
        _patch_asrpy()
        info = mne.create_info(STEW_CLAB, fs, "eeg", verbose="ERROR")
        raw = mne.io.RawArray(x.T * 1e-6, info, verbose="ERROR")
        cleaner = asrpy.ASR(sfreq=fs, cutoff=20)
        # asrpy's block covariance has an off-by-one for some calibration
        # lengths; calibrating on a slightly shorter stretch avoids it.
        for trim in range(0, 2 * fs, 8):
            try:
                cleaner.fit(raw.copy().crop(tmax=(len(x) - 1 - trim) / fs))
                break
            except ValueError:
                continue
        else:
            raise RuntimeError("ASR calibration failed")
        x = cleaner.transform(raw).get_data().T * 1e6
    return x - x.mean(axis=1, keepdims=True)


def load_stew(subject, asr=True):
    """Low (_lo, rest) and high (_hi, SIMKAP) recordings of one STEW subject."""
    out = []
    for suffix, label in (("lo", 0), ("hi", 1)):
        x = np.loadtxt(DATA_DIR / "STEW Dataset" / f"{subject}_{suffix}.txt")
        out.append(Recording(subject, label, _stew_preprocess(x, STEW_FS, asr), STEW_FS))
    return out


def segment(rec, n_segs, from_end=False):
    """Non-overlapping 4 s segments, shape (n_segs, seg_len, n_channels).

    from_end takes the last n_segs full segments of the recording instead of
    the first ones.
    """
    n = SEG_SEC * rec.fs
    total = n_segs * n
    x = rec.x[-total:] if from_end else rec.x[:total]
    return x.reshape(n_segs, n, -1)


def load_segments(dataset, subject):
    """(segments, labels) of one subject: (n, seg_len, n_ch) and (n,)."""
    cfg = DATASETS[dataset]
    recs = load_mat(subject) if dataset == "mat" else load_stew(subject)
    segs, labels = [], []
    for rec in recs:
        # MAT rest recordings are longer than 60 s; the last minute before
        # the task is used.
        s = segment(rec, cfg["segs"], from_end=(dataset == "mat" and rec.label == 0))
        segs.append(s)
        labels += [rec.label] * len(s)
    return np.concatenate(segs), np.array(labels)
