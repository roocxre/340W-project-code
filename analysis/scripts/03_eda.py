"""Stage 3: exploratory analysis of the joined set, in the forms the source papers used.

    ../synthesis/.venv/bin/python scripts/03_eda.py

Train split only (38 subjects), clean windows, so nothing here is informed by val or test.

    Shin 2018 / iNCog 2026   mean spectrum per level (Fig. eda_psd), topography of the change
                             from 0-back (Fig. eda_topomap, Shin's ERD maps / iNCog's BEAM maps)
    COG-BCI 2023             frontal theta and parieto-occipital alpha per level and subject, with a
                             within-subject test (Fig. eda_roi; the paper used a GLM per ROI)
    Yedukondalu 2025         one-way ANOVA F of every feature across levels (Fig. eda_anova; the
                             paper ranked R-LMD features this way)
    behavior                 block accuracy and RT per level
    PCA                      band power across vs within subjects: what each PC tracks (η² by subject
                             and by level), PC1 loadings, and whether top-k PCs help logistic
                             regression (Fig. eda_pca, eda_pca.csv, eda_pca_preprocessing.csv)

Outputs: results/figures/eda_*.png, results/eda_*.csv; print to results/log_03_eda.txt.
"""
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt  # noqa: E402

import nbackml as ml  # noqa: E402
import nbacksplit as ns  # noqa: E402
import vizstyle as vz  # noqa: E402

CH = ["Fp1", "Fp2", "AFz", "F1", "F2", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
      "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "POz", "O1", "O2"]
ROI = {"frontal": ["AFz", "F1", "F2", "FC1", "FC2"], "central": ["C3", "Cz", "C4", "CP1", "CP2"],
       "parieto-occipital": ["P3", "Pz", "P4", "POz", "O1", "O2"]}
DS_NAME = {"shin": "Shin 2018", "cog": "COG-BCI"}
LEVELS = {"shin": [0, 2, 3], "cog": [0, 1, 2]}
FIG = ml.FIG


def subject_level_means(df, cols):
    return df.groupby(["dataset", "subject", "n"])[cols].mean()


def psd(df):
    """Mean log10 PSD (1-40 Hz) per subject x level x ROI, from the windows."""
    rows = []
    for subject, g in df.groupby("subject"):
        x = np.load(ns.WINDOWS / f"{subject}.npz")["x"][g.npz_index.to_numpy()].astype(float)
        f, p = welch(x, fs=200.0, nperseg=400, noverlap=200, axis=-1)
        keep = (f >= 1) & (f <= 40)
        lp = np.log10(p[..., keep])
        for n in sorted(g.n.unique()):
            m = (g.n == n).to_numpy()
            for roi, chans in ROI.items():
                idx = [CH.index(c) for c in chans]
                rows.append(dict(dataset=g.dataset.iat[0], subject=subject, n=n, roi=roi,
                                 **{f"{fr:.1f}": v for fr, v in zip(f[keep], lp[m][:, idx].mean((0, 1)))}))
    return pd.DataFrame(rows)


def fig_psd(P):
    freqs = [c for c in P.columns if c[0].isdigit()]
    fv = np.array(freqs, float)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    for i, ds in enumerate(["shin", "cog"]):
        for j, roi in enumerate(["frontal", "parieto-occipital"]):
            ax = axes[i, j]
            q = P[(P.dataset == ds) & (P.roi == roi)]
            base = q[q.n == 0].set_index("subject")[freqs]
            for n in LEVELS[ds]:
                d = q[q.n == n].set_index("subject")[freqs]
                m = d.mean().to_numpy()
                ax.plot(fv, m, color=vz.LEVEL[n], label=f"{n}-back")
            ax.set_title(f"{DS_NAME[ds]}, {roi}", loc="left")
            ax.axvspan(8, 13, color=vz.GRID, alpha=0.5, lw=0)
            if i == 1:
                ax.set_xlabel("Frequency (Hz)")
            if j == 0:
                ax.set_ylabel("log10 power (µV²/Hz)")
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Mean spectrum by n-back level (train subjects; alpha band shaded)", x=0.02, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    vz.save(fig, FIG / "eda_psd.png")


def fig_topomap(M):
    info = mne.create_info(CH, 200.0, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    bands = ["theta", "alpha", "beta"]
    out = []
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    vmax = 0.0
    deltas = {}
    for ds in ["shin", "cog"]:
        top = max(LEVELS[ds])
        for b in bands:
            cols = [f"abs_{b}_{c}" for c in CH]
            m = M.loc[ds]
            d = (m.xs(top, level="n")[cols] - m.xs(0, level="n")[cols])
            p = np.array([stats.wilcoxon(d[c]).pvalue for c in cols])
            deltas[ds, b] = (d.mean().to_numpy(), p)
            vmax = max(vmax, np.abs(d.mean()).max())
            out += [dict(dataset=ds, band=b, channel=c, delta_log10=v, wilcoxon_p=pp, n_subjects=len(d))
                    for c, v, pp in zip(CH, d.mean().to_numpy(), p)]
    for i, ds in enumerate(["shin", "cog"]):
        for j, b in enumerate(bands):
            v, p = deltas[ds, b]
            im, _ = mne.viz.plot_topomap(v, info, axes=axes[i, j], cmap=vz.DIVERGING, vlim=(-vmax, vmax),
                                         mask=p < 0.05 / len(CH), show=False, contours=0,
                                         mask_params=dict(marker="o", markerfacecolor="white",
                                                          markeredgecolor=vz.INK, markersize=4))
            if i == 0:
                axes[i, j].set_title(b, fontsize=10)
            if j == 0:
                axes[i, j].text(-0.12, 0.5, f"{DS_NAME[ds]}\n{max(LEVELS[ds])} − 0-back",
                                transform=axes[i, j].transAxes, rotation=90, ha="center", va="center",
                                fontsize=9, fontweight="bold")
    cb = fig.colorbar(im, ax=axes, shrink=0.6, label="Δ log10 band power")
    cb.outline.set_visible(False)
    fig.suptitle("Change in band power from 0-back to the top level (white dots: p < 0.05, Bonferroni)",
                 x=0.02, ha="left", fontsize=10, fontweight="bold")
    vz.save(fig, FIG / "eda_topomap.png")
    return pd.DataFrame(out)


def roi_table(M):
    rows = []
    for ds in ["shin", "cog"]:
        m = M.loc[ds]
        for roi, band in [("frontal", "theta"), ("parieto-occipital", "alpha"), ("central", "beta")]:
            cols = [f"abs_{band}_{c}" for c in ROI[roi]]
            v = m[cols].mean(1).unstack("n")[LEVELS[ds]]
            fr = stats.friedmanchisquare(*[v[n] for n in LEVELS[ds]])
            w = stats.wilcoxon(v[max(LEVELS[ds])] - v[0])
            rows.append(dict(dataset=ds, roi=roi, band=band, n_subjects=len(v),
                             **{f"mean_n{n}": v[n].mean() for n in LEVELS[ds]},
                             top_minus_0=(v[max(LEVELS[ds])] - v[0]).mean(),
                             subjects_lower_at_top=int((v[max(LEVELS[ds])] < v[0]).sum()),
                             friedman_chi2=fr.statistic, friedman_p=fr.pvalue, wilcoxon_top_vs_0_p=w.pvalue))
    return pd.DataFrame(rows)


def fig_roi(M):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.8), sharey=False)
    k = 0
    for roi, band in [("frontal", "theta"), ("parieto-occipital", "alpha")]:
        for ds in ["shin", "cog"]:
            ax = axes[k]
            k += 1
            cols = [f"abs_{band}_{c}" for c in ROI[roi]]
            v = M.loc[ds][cols].mean(1).unstack("n")[LEVELS[ds]]
            v = v.sub(v.mean(1), axis=0)                     # within-subject: centre each subject
            xs = np.arange(len(LEVELS[ds]))
            for _, r in v.iterrows():
                ax.plot(xs, r.to_numpy(), color=vz.GRID, lw=0.8, zorder=1)
            ax.plot(xs, v.mean().to_numpy(), color=vz.DATASET[ds], lw=2, marker="o", zorder=3)
            ax.set_xticks(xs, [f"{n}" for n in LEVELS[ds]])
            ax.set_xlabel("n")
            ax.set_title(f"{DS_NAME[ds]}\n{roi} {band}", fontsize=8.5, loc="left")
            ax.axhline(0, color=vz.MUTED, lw=0.6)
    axes[0].set_ylabel("log10 power, subject-centred")
    fig.suptitle("ROI band power per level: grey = one subject, colour = mean", x=0.02, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    vz.save(fig, FIG / "eda_roi.png")


def anova(df, task):
    """One-way ANOVA F across the task's levels, per feature, on within-subject z-scores."""
    g = df.subject.to_numpy()
    y = ml.labels(df, task)
    out = {}
    for name, X in [("band power", ml.bandpower(df)), ("R-LMD", ml.rlmd(df))]:
        Z = ml.zscore_subject(X, g)
        out[name] = stats.f_oneway(*[Z[y == c] for c in np.unique(y)]).statistic
    return out


def fig_anova(F):
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    feats = ["SN", "KT", "PSD", "FE", "RE", "DE"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1]))
    vmax = max(np.nanmax(np.r_[f["band power"], f["R-LMD"]]) for f in F.values())
    for j, task in enumerate(ml.TASKS):
        bp = np.nan_to_num(F[task]["band power"]).reshape(2, 26, 5).mean(1)      # abs then rel; channel, band
        rl = np.nan_to_num(F[task]["R-LMD"]).reshape(26, 5, 6).mean(0)
        for i, (mat, xl, yl) in enumerate([(bp, bands, ["abs", "rel"]), (rl, feats, [f"M{m + 1}" for m in range(5)])]):
            ax = axes[i, j]
            ax.imshow(mat, cmap=mpl_seq(), vmin=0, aspect="auto")
            ax.set_xticks(range(len(xl)), xl, fontsize=7.5)
            ax.set_yticks(range(len(yl)), yl, fontsize=7.5)
            ax.grid(False)
            for (r, c), v in np.ndenumerate(mat):
                ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.55 * mat.max() else vz.INK)
            if i == 0:
                ax.set_title(ml.TASKS[task]["title"], fontsize=8.5, loc="left")
    axes[0, 0].set_ylabel("band power")
    axes[1, 0].set_ylabel("R-LMD mode")
    fig.suptitle("ANOVA F across levels, averaged over the 26 channels (within-subject z-scores)",
                 x=0.02, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    vz.save(fig, FIG / "eda_anova.png")


def rlmd_redundancy(task):
    """How much of the R-LMD features' variance band power predicts (ridge, within-subject z-scores,
    in-sample R² on train), overall and per R-LMD mode."""
    from sklearn.linear_model import RidgeCV
    df = ml.frame("train", task)
    g = df.subject.to_numpy()
    B, R = ml.zscore_subject(ml.bandpower(df), g), ml.zscore_subject(ml.rlmd(df), g)
    mode = (np.arange(R.shape[1]) % 30) // 6                     # columns: channel x mode x feature
    out = [dict(task=task, rlmd_part="all modes", r2=RidgeCV(alphas=[1, 10, 100]).fit(B, R).score(B, R))]
    for m in range(5):
        Rm = R[:, mode == m]
        out.append(dict(task=task, rlmd_part=f"mode {m + 1}", r2=RidgeCV(alphas=[1, 10, 100]).fit(B, Rm).score(B, Rm)))
    return out


def eta2(x, g):
    """Share of x's variance explained by the grouping g (one-way η²)."""
    m = x.mean()
    between = sum((x[g == v].mean() - m) ** 2 * (g == v).sum() for v in np.unique(g))
    return between / ((x - m) ** 2).sum()


NORMS = {"pooled": "z-scored across subjects", "subject": "z-scored within each subject"}


def pca_fit(df, task, norm, k=10):
    """PCA of band power under one normalization; PC signs fixed so each loads positively on
    whole-scalp alpha (abs + rel), which makes PC1 read as 'alpha power'."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    g = df.subject.to_numpy()
    X = StandardScaler().fit_transform(ml.bandpower(df)) if norm == "pooled" else ml.zscore_subject(ml.bandpower(df), g)
    p = PCA(k, random_state=ml.SEED).fit(X)
    alpha = np.array([("_alpha_" in c) for c in ns.feature_columns(df)], float)
    sign = np.sign(p.components_ @ alpha)
    sign[sign == 0] = 1
    W = p.components_ * sign[:, None]
    return X @ W.T, W, 100 * p.explained_variance_ratio_, np.cumsum(100 * PCA(random_state=ml.SEED).fit(X).explained_variance_ratio_)


def pca_table(task):
    df = ml.frame("train", task)
    y = ml.labels(df, task)
    rows = []
    for norm in NORMS:
        Z, _, var, _ = pca_fit(df, task, norm)
        for i in range(Z.shape[1]):
            rows.append(dict(task=task, normalization=norm, pc=i + 1, variance_pct=var[i],
                             eta2_subject=eta2(Z[:, i], df.subject.to_numpy()),
                             eta2_dataset=eta2(Z[:, i], df.dataset.to_numpy()),
                             eta2_level=eta2(Z[:, i], y), r_level=np.corrcoef(Z[:, i], y)[0, 1]))
    return pd.DataFrame(rows)


def pca_as_preprocessing(task="bin", ks=(None, 100, 50, 20, 10)):
    """Does keeping only the top PCs help the logistic-regression baseline? Subject-wise 5-fold CV
    inside train (PCA refitted in each fold)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    df = ml.frame("train", task)
    y, g = ml.labels(df, task), df.subject.to_numpy()
    X = ml.zscore_subject(ml.bandpower(df), g)
    folds = ml.subject_folds(y, g, k=5)
    rows = []
    for k in ks:
        pred = np.zeros_like(y)
        for tr, te in folds:
            steps = [StandardScaler()] + ([PCA(k, random_state=ml.SEED)] if k else []) + \
                    [LogisticRegression(C=0.005, max_iter=5000)]
            pred[te] = make_pipeline(*steps).fit(X[tr], y[tr]).predict(X[te])
        rows.append(dict(task=task, components="all 260" if k is None else k,
                         cv_bacc=100 * balanced_accuracy_score(y, pred)))
    return pd.DataFrame(rows)


def fig_pca(T, task="bin"):
    df = ml.frame("train", task)
    y = ml.labels(df, task)
    lv = ml.TASKS[task]["levels"]
    fits = {norm: pca_fit(df, task, norm) for norm in NORMS}
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.62, wspace=0.5)

    # a, b: PC1 score per level under each normalization (PC1 is z-scored so the two share an axis)
    edges = np.linspace(-3.5, 3.5, 43)
    for j, norm in enumerate(NORMS):
        ax = fig.add_subplot(gs[0, j], sharey=None)
        z = fits[norm][0][:, 0]
        z = (z - z.mean()) / z.std()
        for c, n in enumerate(lv):
            ax.hist(z[y == c], bins=edges, density=True, histtype="step", lw=1.6, color=vz.LEVEL[n],
                    label=f"{n}-back")
            ax.axvline(np.median(z[y == c]), color=vz.LEVEL[n], lw=0.9, ls=":")
        r = T[(T.task == task) & (T.normalization == norm) & (T.pc == 1)].iloc[0]
        ax.set(xlabel="PC1 score, alpha power (SD)", ylabel="density" if j == 0 else "")
        ax.set_title(f"{NORMS[norm].replace('z-scored', 'z')} (PC1: {r.variance_pct:.0f} %)\nPC1 η²: subject "
                     f"{r.eta2_subject:.2f}, level {r.eta2_level:.2f}", loc="left", fontsize=7.8)
        if j == 0:
            ax.legend(loc="upper right", fontsize=6.8, handlelength=1.4, borderaxespad=0.2)

    # c: what each PC tracks, subject identity vs load level
    ax = fig.add_subplot(gs[0, 2])
    q = T[T.task == task]
    for (norm, col, lab), color, ls in zip(
            [("pooled", "eta2_subject", "subject, across-subject z"), ("pooled", "eta2_level", "level, across-subject z"),
             ("subject", "eta2_level", "level, within-subject z")],
            [vz.MUTED, vz.SERIES[1], vz.SERIES[0]], ["-", "--", "-"]):
        r = q[q.normalization == norm]
        ax.plot(r.pc, r[col], color=color, ls=ls, lw=1.6, marker="o", ms=3.5, label=lab)
    ax.set(xlabel="principal component", ylabel="η², share of PC variance", ylim=(0, 1), xticks=[1, 2, 4, 6, 8, 10])
    ax.set_title("what each PC tracks", loc="left", fontsize=7.8)
    ax.legend(fontsize=6.3, loc="upper right", borderaxespad=0.2)

    # d: cumulative variance
    ax = fig.add_subplot(gs[1, 0])
    for norm, color in zip(NORMS, [vz.MUTED, vz.SERIES[0]]):
        cum = fits[norm][3]
        k90 = int(np.searchsorted(cum, 90) + 1)
        ax.plot(np.arange(1, len(cum) + 1), cum, color=color, lw=1.6,
                label=f"z {'across' if norm == 'pooled' else 'within'}: {k90} PCs")
        ax.plot(k90, cum[k90 - 1], "o", color=color, ms=4)
    ax.axhline(90, color=vz.GRID, lw=0.8)
    ax.set(xlabel="components (log scale)", ylabel="cumulative variance (%)", xscale="log", xlim=(1, 260), ylim=(0, 101))
    ax.set_title("PCs for 90 % of variance", loc="left", fontsize=8)
    ax.legend(fontsize=6.5, loc="lower right", handlelength=1.2, borderaxespad=0.3)
    ax.set_xticks([1, 10, 100], ["1", "10", "100"])

    # e: PC1 (within-subject) loadings by band, f: on the scalp (absolute alpha)
    W = fits["subject"][1][0].reshape(2, 26, 5)                  # abs/rel, channel, band
    bands = ["δ", "θ", "α", "β", "γ"]
    ax = fig.add_subplot(gs[1, 1])
    M = W.mean(1)
    v = np.abs(M).max()
    ax.imshow(M, cmap=vz.DIVERGING, vmin=-v, vmax=v, aspect="auto")
    for (r, c), val in np.ndenumerate(M):
        ax.text(c, r, f"{val:+.2f}", ha="center", va="center", fontsize=6,
                color="white" if abs(val) > 0.6 * v else vz.INK)
    ax.set_xticks(range(5), bands, fontsize=8.5)
    ax.set_yticks([0, 1], ["abs", "rel"], fontsize=7.5)
    ax.grid(False)
    ax.set_title("PC1 loading, mean over channels", loc="left", fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    info = mne.create_info(CH, 200.0, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    a = W[0, :, 2]
    im, _ = mne.viz.plot_topomap(a, info, axes=ax, cmap=vz.DIVERGING, vlim=(-np.abs(a).max(), np.abs(a).max()),
                                 show=False, contours=0, sensors=True)
    ax.set_title("PC1 loading on absolute alpha", loc="left", fontsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)

    fig.suptitle(f"PCA of band power, {ml.TASKS[task]['title']} (train subjects, {len(df):,} windows)",
                 x=0.02, y=0.99, ha="left", fontsize=10, fontweight="bold")
    vz.save(fig, FIG / "eda_pca.png")


def mpl_seq():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue_seq", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])


def main():
    vz.use()
    pd.set_option("display.width", 200)
    tr = ns.load("train", clean=True)
    print(f"train split, clean windows: {len(tr)} from {tr.subject.nunique()} subjects")
    print(pd.crosstab([tr.dataset, tr.n], "windows").T.to_string(), "\n")

    # class balance for every task and split
    bal = []
    for task in ml.TASKS:
        for s in ns.SPLITS:
            d = ml.frame(s, task)
            bal.append(dict(task=task, split=s, subjects=d.subject.nunique(), windows=len(d),
                            **{f"n{n}": int((d.n == n).sum()) for n in ml.TASKS[task]["levels"]}))
    bal = pd.DataFrame(bal)
    bal.to_csv(ml.RESULTS / "eda_class_balance.csv", index=False)
    print("class balance (clean windows)\n" + bal.to_string(index=False), "\n")

    # behavior, one row per block
    blocks = tr.drop_duplicates(["subject", "run", "block_in_run"])
    beh = blocks.groupby(["dataset", "n"]).agg(blocks=("subject", "size"), accuracy=("block_accuracy", "mean"),
                                               accuracy_sd=("block_accuracy", "std"), rt_s=("block_mean_rt_s", "mean"))
    beh.round(3).to_csv(ml.RESULTS / "eda_behavior.csv")
    print("behavior per block\n" + beh.round(3).to_string(), "\n")

    P = psd(tr)
    fig_psd(P)
    M = subject_level_means(tr, ns.feature_columns(tr))
    topo = fig_topomap(M)
    topo.round(5).to_csv(ml.RESULTS / "eda_topomap.csv", index=False)
    sig = topo[topo.wilcoxon_p < 0.05 / 26].groupby(["dataset", "band"]).channel.apply(lambda s: " ".join(s))
    print("channels changing at the top level vs 0-back (Wilcoxon, Bonferroni over 26 channels)")
    print(sig.to_string(), "\n")
    roi = roi_table(M)
    roi.to_csv(ml.RESULTS / "eda_roi.csv", index=False)
    print("ROI band power per level (log10 µV²), within-subject tests")
    print(roi.round(4).to_string(index=False), "\n")
    fig_roi(M)

    F = {task: anova(ml.frame("train", task), task) for task in ml.TASKS}
    rl_names = [f"{c}_M{m + 1}_{f}" for c in CH for m in range(5) for f in ["SN", "KT", "PSD", "FE", "RE", "DE"]]
    bp_names = ns.feature_columns(tr)
    rows = []
    for task, f in F.items():
        for fam, names in [("band power", bp_names), ("R-LMD", rl_names)]:
            rows += [dict(task=task, family=fam, feature=n, F=v) for n, v in zip(names, f[fam])]
    A = pd.DataFrame(rows)
    A.round(3).to_csv(ml.RESULTS / "eda_anova.csv", index=False)
    print("ANOVA F across levels: median and top features per task")
    for task in ml.TASKS:
        a = A[A.task == task]
        med = a.groupby("family").F.median().round(1).to_dict()
        top = a.nlargest(8, "F")
        print(f"  {task}: median F {med}; top: " + ", ".join(f"{r.feature} {r.F:.0f}" for r in top.itertuples()))
    fig_anova(F)
    red = pd.DataFrame([r for task in ml.TASKS for r in rlmd_redundancy(task)])
    red.round(3).to_csv(ml.RESULTS / "eda_rlmd_redundancy.csv", index=False)
    print("\nR-LMD variance predicted from band power (ridge R², train, within-subject z)")
    print(red.pivot(index="rlmd_part", columns="task", values="r2").round(2).to_string())

    T = pd.concat([pca_table(task) for task in ml.TASKS], ignore_index=True)
    T.round(4).to_csv(ml.RESULTS / "eda_pca.csv", index=False)
    print("\nPCA of band power: share of each PC's variance explained by subject and by level (η²)")
    for task in ml.TASKS:
        for norm in NORMS:
            r = T[(T.task == task) & (T.normalization == norm)].head(5)
            print(f"  {task:5s} {norm:7s} var % {np.round(r.variance_pct.to_numpy(), 1)} | subject "
                  f"{np.round(r.eta2_subject.to_numpy(), 2)} | level {np.round(r.eta2_level.to_numpy(), 3)}")
    pre = pca_as_preprocessing()
    pre.round(2).to_csv(ml.RESULTS / "eda_pca_preprocessing.csv", index=False)
    print("PCA as preprocessing for logistic regression (bin, subject-wise 5-fold CV in train)")
    print(pre.round(1).to_string(index=False))
    fig_pca(T)
    print("\nfigures: " + ", ".join(sorted(p.name for p in FIG.glob("eda_*.png"))))


if __name__ == "__main__":
    main()
