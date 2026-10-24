"""Dataset locations, versions, channel sets and label maps (see PROVENANCE.md)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"

# Shin 2018, dataset A (n-back), MATLAB format. Archive ETag "59f7ff0b-16b10eedf", 6,091,239,135 bytes.
SHIN_URL = "https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/EEG/EEG_01-26_MATLAB.zip"
SHIN_DIR = REPO / "eeg-analysis" / "Cleaned Data"
SHIN_SUBJECTS = [f"VP{i:03d}" for i in range(1, 27)]

# COG-BCI version 4 (v1-v3 have data bugs in participants 17-24 and an electrode name mismatch).
COG_RECORD = 7413650
COG_URL = "https://zenodo.org/records/7413650/files/sub-{:02d}.zip?download=1"
COG_DIR = ROOT / "data" / "cogbci"     # n-back .set/.fdt only, ~13 GB, git-ignored
COG_SUBJECTS = list(range(1, 30))
COG_NO_CZ = set(range(1, 10))          # Cz not recorded for participants 1-9
COG_SESSIONS = ["ses-S1", "ses-S2", "ses-S3"]
COG_TASKS = {"zeroBACK": 0, "oneBACK": 1, "twoBACK": 2}

# The 26 channels both datasets share (Cz interpolated for COG-BCI 1-9).
CHANNELS = ["Fp1", "Fp2", "AFz", "F1", "F2", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
            "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "POz", "O1", "O2"]

FS = 200.0            # common sampling rate (Shin is distributed at 200 Hz)
LINE = 50.0           # both recorded on 50 Hz mains
