"""Stage 4: characteristics of the joined table and baseline ML checks.

All classifiers: logistic regression on the 260 spectral features, z-scored within each
subject (removes amplifier/cap/reference offsets), artifact windows excluded, and
subjects never split across training and test. Accuracy is the mean over test subjects.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nbackjoin import features, sources

df = pd.read_csv(sources.RESULTS / "nback_features.csv")
qa = pd.read_csv(sources.RESULTS / "preprocessing_qa.csv")
abs_cols, rel_cols = features.feature_columns()
F = abs_cols + rel_cols


def section(t):
    print(f"\n## {t}\n")


section("Dimensions")
print(f"rows {len(df)}, columns {df.shape[1]} ({df.shape[1] - len(F)} metadata + {len(F)} features)")
print(f"subjects {df.subject.nunique()} (shin {df[df.dataset == 'shin'].subject.nunique()}, cog {df[df.dataset == 'cog'].subject.nunique()})")
print(f"file size {(sources.RESULTS / 'nback_features.csv').stat().st_size / 1e6:.0f} MB")
print("\nwindows by dataset x n")
print(pd.crosstab(df.dataset, df.n, margins=True).to_string())
print("\nsubjects by dataset x n")
print(df.groupby(["dataset", "n"]).subject.nunique().unstack().to_string())
print("\nminutes of task EEG by dataset x n")
print((pd.crosstab(df.dataset, df.n) * features.WIN_S / 60).round(0).to_string())
print("\nwindows per subject: median", df.groupby("subject").size().median(),
      "| shin", df[df.dataset == "shin"].groupby("subject").size().median(),
      "| cog", df[df.dataset == "cog"].groupby("subject").size().median())

section("Quality")
print("artifact windows (peak-to-peak > 150 µV):")
print(df.groupby("dataset").artifact.agg(["sum", "mean"]).round(3).to_string())
print("\nartifact rate by subject: max", df.groupby("subject").artifact.mean().max().round(3),
      "| subjects > 20 %:", ", ".join(df.groupby("subject").artifact.mean().loc[lambda s: s > .2].index))
print("\npreprocessing per session (mean):")
print(qa.groupby("dataset")[["n_bad_channels", "n_ica", "n_ica_removed", "n_ica_eye", "n_ica_muscle", "n_ica_heart"]]
      .mean().round(2).to_string())
s = qa[qa.dataset == "shin"]
print(f"\nShin |r(Fp1/Fp2, VEOG)| before ICA {s.fp_veog_corr_before.mean():.2f}, after {s.fp_veog_corr_after.mean():.2f}")
print("interrupted COG sessions in table:", df[df.interrupted].groupby(["subject", "session"]).ngroups,
      f"({df.interrupted.mean():.1%} of rows)")

section("Behaviour by n (block level)")
blk = df.drop_duplicates(["subject", "session", "run", "block_in_run"])
print(blk.groupby(["dataset", "n"])[["block_accuracy", "block_mean_rt_s"]].mean().round(3).to_string())

section("Spectral effect of n (per-subject means, Pz and frontal F1/F2/AFz)")
for ds, lo, hi in (("shin", 0, 2), ("shin", 0, 3), ("cog", 0, 1), ("cog", 0, 2)):
    d = df[(df.dataset == ds) & ~df.artifact]
    for band, chans in (("alpha", ["Pz", "P3", "P4"]), ("theta", ["F1", "F2", "AFz"])):
        cols = [f"abs_{band}_{c}" for c in chans]
        m = d.groupby(["subject", "n"])[cols].mean().mean(1).unstack()
        diff = m[hi] - m[lo]
        print(f"{ds} {band:5s} {hi} vs {lo}: mean Δlog10 {diff.mean():+.3f}, lower in {(diff < 0).sum()}/{len(diff)}, "
              f"Wilcoxon p = {wilcoxon(diff).pvalue:.2g}")


def zscore_within_subject(d):
    X = d[F].to_numpy()
    out = np.empty_like(X)
    for s, idx in d.groupby("subject").indices.items():
        x = X[idx]
        out[idx] = (x - x.mean(0)) / (x.std(0) + 1e-12)
    return out


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.05))


def per_subject_acc(y, pred, groups):
    t = pd.DataFrame(dict(ok=y == pred, g=groups))
    return t.groupby("g").ok.mean()


def grouped_cv(d, label, n_splits=10):
    X, y, g = zscore_within_subject(d), d[label].to_numpy(), d.subject.to_numpy()
    pred = np.empty_like(y)
    for tr, te in GroupKFold(n_splits).split(X, y, g):
        pred[te] = clf().fit(X[tr], y[tr]).predict(X[te])
    return per_subject_acc(y, pred, g)


def train_test(dtr, dte, label):
    m = clf().fit(zscore_within_subject(dtr), dtr[label])
    return per_subject_acc(dte[label].to_numpy(), m.predict(zscore_within_subject(dte)), dte.subject.to_numpy())


def report(name, acc, chance):
    print(f"{name:52s} {100 * acc.mean():5.1f} % ± {100 * acc.std():4.1f} (chance {100 * chance:.0f} %), "
          f"subjects above chance {(acc > chance).sum()}/{len(acc)}")


clean = df[~df.artifact].copy()
section("Baselines (subject-wise CV)")
report("Shin 0/2/3, 10-fold by subject", grouped_cv(clean[clean.dataset == "shin"], "n"), 1 / 3)
report("COG-BCI 0/1/2, 10-fold by subject", grouped_cv(clean[clean.dataset == "cog"], "n"), 1 / 3)
pool = clean[clean.n.isin([0, 2])]
acc = grouped_cv(pool, "n")
report("Pooled 0 vs 2, 10-fold by subject (all)", acc, .5)
for ds in ("shin", "cog"):
    report(f"  of which {ds} test subjects", acc[acc.index.str.startswith(ds)], .5)
report("Cross-dataset 0 vs 2: train Shin, test COG-BCI", train_test(pool[pool.dataset == "shin"], pool[pool.dataset == "cog"], "n"), .5)
report("Cross-dataset 0 vs 2: train COG-BCI, test Shin", train_test(pool[pool.dataset == "cog"], pool[pool.dataset == "shin"], "n"), .5)
low_high = clean[((clean.dataset == "shin") & clean.n.isin([0, 3])) | ((clean.dataset == "cog") & clean.n.isin([0, 2]))].copy()
low_high["high"] = (low_high.n > 0).astype(int)
report("Pooled low vs high (Shin 0/3, COG 0/2), 10-fold", grouped_cv(low_high, "high"), .5)
allv = clean.copy()
report("Pooled all levels 0/1/2/3 as 4 classes, 10-fold", grouped_cv(allv, "n"), allv.n.value_counts(normalize=True).max())

section("Dataset identity (domain shift)")
d = clean.copy()
d["is_cog"] = (d.dataset == "cog").astype(int)
X, y, g = d[F].to_numpy(), d.is_cog.to_numpy(), d.subject.to_numpy()
pred = np.empty_like(y)
for tr, te in GroupKFold(10).split(X, y, g):
    pred[te] = clf().fit(X[tr], y[tr]).predict(X[te])
print(f"dataset decodable from raw features (subject-wise CV): {100 * (pred == y).mean():.1f} % "
      f"(majority {100 * max(y.mean(), 1 - y.mean()):.1f} %)")
for name, sub in (("all levels", d), ("shared levels 0 and 2 only", d[d.n.isin([0, 2])])):
    X, y, g = zscore_within_subject(sub), sub.is_cog.to_numpy(), sub.subject.to_numpy()
    pred = np.empty_like(y)
    for tr, te in GroupKFold(10).split(X, y, g):
        pred[te] = clf().fit(X[tr], y[tr]).predict(X[te])
    print(f"dataset decodable after within-subject z-scoring, {name}: {100 * (pred == y).mean():.1f} % "
          f"(majority {100 * max(y.mean(), 1 - y.mean()):.1f} %)")
