"""Stage 3: assemble the joined feature table, results/nback_features.csv.

One row per 4 s window. Metadata columns first, then 130 log10 absolute band powers
(abs_<band>_<channel>, µV²) and 130 relative band powers (rel_<band>_<channel>).
Row i of a subject's cache/windows/<subject>.npz is the row with that subject and
npz_index = i, for models that need the signals rather than features.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nbackjoin import features, sources

META = ["row_id", "dataset", "subject", "session", "run", "n", "block_in_run", "window", "t_in_block_s",
        "task_position", "interrupted", "soa_s", "n_trials", "n_targets", "block_accuracy",
        "block_mean_rt_s", "cz_interpolated", "n_bad_channels", "n_ica_removed", "ptp_max_uv",
        "artifact", "npz_index"]

abs_cols, rel_cols = features.feature_columns()
frames, qa = [], []
for f in sorted((sources.CACHE / "windows").glob("*.npz")):
    z = np.load(f)
    meta = pd.DataFrame(json.loads(str(z["meta"])))
    meta["npz_index"] = np.arange(len(meta))
    la, rel = features.spectral(z["x"])
    feats = pd.DataFrame(np.concatenate([la.reshape(len(meta), -1), rel.reshape(len(meta), -1)], 1),
                         columns=abs_cols + rel_cols)
    frames.append(pd.concat([meta, feats], axis=1))
    qa += json.loads(str(z["qa"]))
df = pd.concat(frames, ignore_index=True)

nb = pd.read_csv(sources.RESULTS / "cogbci_notebook.csv")[["subject", "session", "n", "task_position", "interrupted"]]
df = df.merge(nb, on=["subject", "session", "n"], how="left")
df["interrupted"] = df["interrupted"].fillna(False).astype(bool)
df["artifact"] = df["ptp_max_uv"] > features.PTP_REJECT_UV
df = df.sort_values(["dataset", "subject", "session", "run", "block_in_run", "window"], kind="stable")
df["row_id"] = np.arange(len(df))
df = df[META + abs_cols + rel_cols]

assert not df[abs_cols + rel_cols].isna().any().any()
df.to_csv(sources.RESULTS / "nback_features.csv", index=False, float_format="%.6g")
pd.DataFrame(qa).sort_values(["dataset", "subject", "session"]).to_csv(
    sources.RESULTS / "preprocessing_qa.csv", index=False, float_format="%.4g")
print(df.shape, "->", sources.RESULTS / "nback_features.csv")
