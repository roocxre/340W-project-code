"""Stage 1: 70 / 20 / 10 train / test / validation split of the joined n-back table.

    ../synthesis/.venv/bin/python scripts/01_split.py      (from analysis/)

Compares five ways of splitting the 26,137 windows of synthesis/results/nback_features.csv,
each repeated over 20 seeds, then writes the chosen one:

  A  windows at random, stratified by dataset x n
  B  whole blocks at random, stratified by dataset
  C  whole sessions at random, stratified by dataset
  D  whole subjects at random
  E  whole subjects, stratified by dataset and Cz interpolation   <- chosen

Every technique allocates units (windows, blocks, sessions, subjects) to the three
splits per stratum by largest remainder, so the shares are as close to 70/20/10 as
whole units allow. Each split is scored on leakage (subjects in more than one split),
how close it lands to 70/20/10, whether each split keeps the pool's dataset mix and
all four levels, and a fixed probe: 0- vs 2-back logistic regression (as in
synthesis/scripts/04_viability.py) trained on train and scored on test, on features
z-scored within subject and on raw features.

Outputs: data/{train,test,val}.csv, data/split_subjects.csv, data/split_manifest.json,
results/split_comparison.csv. Print goes to results/log_01_split.txt.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "synthesis" / "results" / "nback_features.csv"
DATA, RESULTS = ROOT / "data", ROOT / "results"
SPLITS = ("train", "test", "val")
SHARES = (0.7, 0.2, 0.1)
SEED, N_SEEDS = 0, 20

TECHNIQUES = {
    "A window-level, stratified by dataset x n": (["row_id"], ["dataset", "n"]),
    "B block-level, stratified by dataset": (["subject", "run", "block_in_run"], ["dataset"]),
    "C session-level, stratified by dataset": (["subject", "session"], ["dataset"]),
    "D subject-level, unstratified": (["subject"], []),
    "E subject-level, stratified by dataset x Cz": (["subject"], ["dataset", "cz_interpolated"]),
}
CHOSEN = "E subject-level, stratified by dataset x Cz"


def allocate(n):
    """Whole-unit counts per split for n units: largest remainder on 70/20/10."""
    raw = np.array(SHARES) * n
    k = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - k), kind="stable")[: n - k.sum()]:
        k[i] += 1
    return k


def assign(df, unit, strata, seed):
    """Split label per row: units are shuffled within each stratum and cut 70/20/10."""
    rng = np.random.default_rng(seed)
    units = df[unit + strata].drop_duplicates(unit).sort_values(unit)
    assert not units.duplicated(unit).any(), "a unit spans two strata"
    label = {}
    for _, g in (units.groupby(strata, sort=True) if strata else [(None, units)]):
        keys = list(g[unit].itertuples(index=False, name=None))
        keys = [keys[i] for i in rng.permutation(len(keys))]
        edges = np.concatenate([[0], np.cumsum(allocate(len(keys)))])
        for name, a, b in zip(SPLITS, edges[:-1], edges[1:]):
            label.update({k: name for k in keys[a:b]})
    return pd.Series([label[k] for k in df[unit].itertuples(index=False, name=None)], index=df.index)


def zsub(X, groups):
    out = np.empty_like(X)
    for g in np.unique(groups):
        m = groups == g
        out[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-12)
    return out


def probe(df, split, feat):
    """0 vs 2, clean windows: fit on train, accuracy (%) on test; z-scored and raw features."""
    m = (~df.artifact & df.n.isin([0, 2])).to_numpy()
    X, y, s, g = df.loc[m, feat].to_numpy(float), df.n[m].to_numpy(), split[m].to_numpy(), df.subject[m].to_numpy()
    out = {}
    for name, Xf in (("z", zsub(X, g)), ("raw", X)):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.05))
        clf.fit(Xf[s == "train"], y[s == "train"])
        out[name] = 100 * (clf.predict(Xf[s == "test"]) == y[s == "test"]).mean()
    return out


def score(df, split, feat):
    n_split_per_subject = split.groupby(df.subject).nunique()
    share = split.value_counts(normalize=True).reindex(SPLITS).fillna(0) * 100
    shin_all = 100 * df.dataset.eq("shin").mean()
    shin = df.dataset.eq("shin").groupby(split).mean().reindex(SPLITS) * 100
    levels = pd.crosstab(split, df.n).reindex(SPLITS)
    p = probe(df, split, feat)
    return {
        "subjects_in_2plus_splits": int((n_split_per_subject > 1).sum()),
        "train_pct": share["train"], "test_pct": share["test"], "val_pct": share["val"],
        "max_share_error_pts": float(np.abs(share.to_numpy() - 100 * np.array(SHARES)).max()),
        "max_shin_share_error_pts": float(np.abs(shin - shin_all).max()),
        "min_windows_per_level_per_split": int(levels.min().min()),
        "probe_test_acc_z": p["z"], "probe_test_acc_raw": p["raw"],
    }


def main():
    df = pd.read_csv(SRC)
    feat = [c for c in df.columns if c.startswith(("abs_", "rel_"))]
    print(f"source {SRC.relative_to(ROOT.parent)}: {len(df)} rows x {df.shape[1]} columns, "
          f"{df.subject.nunique()} subjects")
    print(f"target shares train/test/val = {'/'.join(str(int(100 * s)) for s in SHARES)}; "
          f"{N_SEEDS} seeds per technique; probe = 0 vs 2, clean windows, logistic regression\n")

    rows = []
    for name, (unit, strata) in TECHNIQUES.items():
        for seed in range(N_SEEDS):
            rows.append({"technique": name, "seed": seed, **score(df, assign(df, unit, strata, seed), feat)})
    res = pd.DataFrame(rows)
    res.round(3).to_csv(RESULTS / "split_comparison.csv", index=False)

    agg = res.drop(columns="seed").groupby("technique", sort=False)
    summary = pd.DataFrame({
        "leaked subjects": agg.subjects_in_2plus_splits.mean().round(1),
        "train/test/val %": agg[["train_pct", "test_pct", "val_pct"]].mean().round(1)
                              .astype(str).agg("/".join, axis=1),
        "max share err": agg.max_share_error_pts.max().round(1),
        "max Shin-mix err": agg.max_shin_share_error_pts.max().round(1),
        "min windows/level/split": agg.min_windows_per_level_per_split.min(),
        "probe z (mean, min-max)": agg.probe_test_acc_z.agg(
            lambda v: f"{v.mean():.1f} ({v.min():.1f}-{v.max():.1f})"),
        "probe raw (mean, min-max)": agg.probe_test_acc_raw.agg(
            lambda v: f"{v.mean():.1f} ({v.min():.1f}-{v.max():.1f})"),
    })
    print("Techniques (worst case over seeds for the error columns; Shin is "
          f"{100 * df.dataset.eq('shin').mean():.1f} % of windows overall)")
    print(summary.to_string())

    # chosen split, fixed seed
    unit, strata = TECHNIQUES[CHOSEN]
    split = assign(df, unit, strata, SEED)
    DATA.mkdir(exist_ok=True)
    for name in SPLITS:
        df[split == name].to_csv(DATA / f"{name}.csv", index=False, float_format="%.6g")
    subj = (df.assign(split=split).groupby("subject")
              .agg(dataset=("dataset", "first"), cz_interpolated=("cz_interpolated", "first"),
                   split=("split", "first"), n_windows=("row_id", "size"),
                   levels=("n", lambda v: " ".join(map(str, sorted(v.unique()))))).reset_index())
    subj.to_csv(DATA / "split_subjects.csv", index=False)

    chosen = res[(res.technique == CHOSEN) & (res.seed == SEED)].iloc[0]
    manifest = {
        "source": str(SRC.relative_to(ROOT.parent)),
        "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
        "technique": CHOSEN, "unit": unit, "strata": strata, "seed": SEED,
        "target_shares": dict(zip(SPLITS, SHARES)),
        "splits": {s: {"subjects": sorted(subj.subject[subj.split == s]),
                       "rows": int((split == s).sum())} for s in SPLITS},
        "probe_test_acc_z": round(float(chosen.probe_test_acc_z), 2),
        "probe_test_acc_raw": round(float(chosen.probe_test_acc_raw), 2),
    }
    (DATA / "split_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    print(f"\nChosen: {CHOSEN}, seed {SEED}")
    print(pd.crosstab([split.rename("split"), df.dataset], df.n, margins=True).reindex(
        list(SPLITS) + ["All"], level=0).to_string())
    print("\nsubjects per split and dataset")
    print(pd.crosstab(subj.split, [subj.dataset, subj.cz_interpolated]).reindex(SPLITS).to_string())
    print("\nrows per split: " + ", ".join(f"{s} {(split == s).sum()} ({100 * (split == s).mean():.1f} %)"
                                           for s in SPLITS))
    print(f"artifact rate per split: " + ", ".join(
        f"{s} {100 * df.artifact[split == s].mean():.1f} %" for s in SPLITS))
    print(f"interrupted-session rows per split: " + ", ".join(
        f"{s} {int(df.interrupted[split == s].eq(True).sum())}" for s in SPLITS))
    for s in SPLITS:
        print(f"{s:5s} subjects: {' '.join(manifest['splits'][s]['subjects'])}")
    print(f"\nprobe on chosen split: test accuracy 0 vs 2 = {chosen.probe_test_acc_z:.1f} % (z-scored), "
          f"{chosen.probe_test_acc_raw:.1f} % (raw)")


if __name__ == "__main__":
    main()
