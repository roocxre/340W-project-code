"""Stage 5 (pilot): do Yedukondalu et al.'s R-LMD features add anything on the joined set?

Imports the R-LMD port and its features from eeg-load/loadrepro (needs numba):
    .venv/bin/python scripts/05_rlmd_pilot.py

0 vs 2 only (both datasets), 30 windows per level per subject (seeded), artifact windows
excluded. R-LMD (5-level, as in the paper, and 3-level) gives 6 features per mode and
channel: skewness, kurtosis, mean PSD, fuzzy, Renyi and differential entropy. The same
windows' band-power features come from the CSV. Same protocol as 04_viability: features
z-scored within subject, logistic regression, subject-wise 10-fold, plus cross-dataset.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "eeg-load"))
from loadrepro import features as lf  # noqa: E402

FS, PER_LEVEL, SEED = 200.0, 30, 0
OUT = ROOT / "cache" / "rlmd_pilot.npz"

df = pd.read_csv(ROOT / "results" / "nback_features.csv")
bp_cols = [c for c in df.columns if c.startswith(("abs_", "rel_"))]
d = df[~df.artifact & df.n.isin([0, 2])]
assert d.groupby(["subject", "n"]).size().min() >= PER_LEVEL
d = (d.groupby(["subject", "n"]).sample(n=PER_LEVEL, random_state=SEED)
      .sort_values(["subject", "npz_index"]).reset_index(drop=True))


def subject_features(subject, idx):
    x = np.load(ROOT / "cache" / "windows" / f"{subject}.npz")["x"][idx].astype(float)
    out = {3: [], 5: []}
    for w in x:
        per = [lf.signal_features(ch, FS, levels=(3, 5)) for ch in w]
        for L in (3, 5):
            out[L].append(np.concatenate([p[L].ravel() for p in per]))
    return subject, {L: np.array(v) for L, v in out.items()}


if OUT.exists():
    z = np.load(OUT)
    R = {3: z["r3"], 5: z["r5"]}
    assert np.array_equal(z["row_id"], d.row_id.to_numpy())
else:
    jobs = [(s, g.npz_index.to_numpy()) for s, g in d.groupby("subject", sort=True)]
    res = dict(Parallel(n_jobs=-1, verbose=5)(delayed(subject_features)(s, i) for s, i in jobs))
    R = {L: np.concatenate([res[s][L] for s, _ in jobs]) for L in (3, 5)}
    np.savez(OUT, r3=R[3], r5=R[5], row_id=d.row_id.to_numpy())


def zsub(X, groups):
    X = np.nan_to_num(X.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    out = np.empty_like(X)
    for g in np.unique(groups):
        m = groups == g
        out[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-12)
    return out


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.05))


y, g = d.n.to_numpy(), d.subject.to_numpy()
is_shin = d.dataset.eq("shin").to_numpy()
sets = {
    "band power (260)": d[bp_cols].to_numpy(),
    "R-LMD 3-level (468)": R[3],
    "R-LMD 5-level, paper (780)": R[5],
    "R-LMD 5-level, modes 1-2 entropies (156)": R[5].reshape(len(d), 26, 5, 6)[:, :, :2, 3:].reshape(len(d), -1),
    "band power + R-LMD 3-level (728)": np.hstack([d[bp_cols].to_numpy(), R[3]]),
}
rows = []
for name, X in sets.items():
    Xz = zsub(X, g)
    pred = np.empty_like(y)
    for tr, te in GroupKFold(10).split(Xz, y, g):
        pred[te] = clf().fit(Xz[tr], y[tr]).predict(Xz[te])
    acc = pd.Series(pred == y).groupby(g).mean()
    s2c = (clf().fit(Xz[is_shin], y[is_shin]).predict(Xz[~is_shin]) == y[~is_shin]).mean()
    c2s = (clf().fit(Xz[~is_shin], y[~is_shin]).predict(Xz[is_shin]) == y[is_shin]).mean()
    rows.append(dict(features=name, pooled_cv=round(100 * acc.mean(), 1), sd=round(100 * acc.std(), 1),
                     above_chance=f"{(acc > .5).sum()}/{len(acc)}",
                     shin_to_cog=round(100 * s2c, 1), cog_to_shin=round(100 * c2s, 1)))
print(f"windows {len(d)} ({d.subject.nunique()} subjects x 2 levels x up to {PER_LEVEL}); chance 50 %")
print(pd.DataFrame(rows).to_string(index=False))

# which R-LMD features carry the effect: per-feature standardized 2-vs-0 difference, averaged over channels
names = ["skew", "kurt", "psd", "fuzzy", "renyi", "diffent"]
Z = zsub(R[5], g).reshape(len(d), 26, 5, 6)
eff = Z[y == 2].mean(0) - Z[y == 0].mean(0)        # (26, 5, 6), in within-subject SD units
tab = pd.DataFrame(np.abs(eff).mean(0), index=[f"M{k + 1}" for k in range(5)], columns=names).round(2)
print("\nmean |2-back - 0-back| per R-LMD feature (within-subject SD units, averaged over channels)")
print(tab.to_string())
