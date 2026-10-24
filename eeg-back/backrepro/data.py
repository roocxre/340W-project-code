"""Loading the iNCog-EEG recordings.

Each file holds 32 channels at 200 Hz: the 16 documented ear-referenced 10-20
leads (``Fp1-A1`` ... ``T6-A2``) and 16 undocumented ``Add_lead`` channels.
Only the 16 documented leads are used, as in the paper's figures.
"""

from pathlib import Path

import mne

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "iNCog-EEG"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

FS = 200.0
SUBJECTS = list(range(1, 41))
CLEAN = list(range(1, 31))
NOISY = list(range(31, 41))
CONDITIONS = ["nw", "lw", "mw", "hw"]
CONDITION_NAMES = {"nw": "No Workload", "lw": "Low Workload", "mw": "Moderate Workload", "hw": "High Workload"}
# Table 2 of the paper: the N-back component of each multitasking level. nw is rest, no task.
NBACK = {"nw": None, "lw": 2, "mw": 3, "hw": 4}

RAW_CHANNELS = ["Fp1-A1", "Fp2-A2", "F3-A1", "F4-A2", "C3-A1", "C4-A2", "P3-A1", "P4-A2",
                "O1-A1", "O2-A2", "F7-A1", "F8-A2", "T3-A1", "T4-A2", "T5-A1", "T6-A2"]
CHANNELS = [c.split("-")[0] for c in RAW_CHANNELS]
FRONTAL = ["Fp1", "Fp2", "F3", "F4", "F7", "F8"]

mne.set_log_level("ERROR")


def path(subject, cond):
    return DATA / f"sub{subject:02d}" / f"sub{subject:02d}_{cond}.edf"


def load_raw(subject, cond, leads="eeg16"):
    """Read one recording.

    leads: "all" keeps the 32 channels under their file names; "raw16" keeps the
    16 documented leads under their file names (``Fp1-A1``); "eeg16" also renames
    them to 10-20 labels and attaches the standard_1020 montage for topomaps.
    """
    raw = mne.io.read_raw_edf(path(subject, cond), preload=True)
    if leads == "all":
        return raw
    raw.pick(RAW_CHANNELS)
    if leads == "eeg16":
        raw.rename_channels(dict(zip(RAW_CHANNELS, CHANNELS)))
        raw.set_montage("standard_1020")
    return raw


def load_array(subject, cond):
    """The 16 documented leads in µV, shape (16, n_samples)."""
    return load_raw(subject, cond, "raw16").get_data() * 1e6


def header(subject, cond):
    """Raw EDF header fields that MNE does not expose directly."""
    b = path(subject, cond).read_bytes()[:256]
    return {
        "patient_id": b[8:88].decode().strip(),
        "recording_id": b[88:168].decode().strip(),
        "start": b[168:184].decode(),
        "n_records_declared": int(b[236:244]),
        "record_seconds": float(b[244:252]),
    }


def device_record(subject, cond):
    """The KT88's running record number, stored in the EDF patient field."""
    return int(header(subject, cond)["patient_id"].split()[-1])


def group(subject):
    return "clean" if subject in CLEAN else "noisy"


def iter_files(subjects=SUBJECTS, conds=CONDITIONS):
    for s in subjects:
        for c in conds:
            yield s, c
