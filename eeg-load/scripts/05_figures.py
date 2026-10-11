"""Stage 5: figures in the paper's layout -> results/figures/.

fig2  R-LMD modes and their spectra for one rest and one task segment
fig3  BAO fitness curves
fig4/5  optimized ensemble: tuning curve, confusion matrix, ROC (MAT, STEW)
fig6/7  box plots of the top-3 leads (MAT, STEW)
fig8  topographic maps for the top-3 MAT leads, rest vs task
fig9  accuracy against decomposition level and BAO iterations
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loadrepro import data, paper, pipeline, rlmd  # noqa: E402

REST, TASK = "black", "tab:blue"


def read_csv(name):
    with open(data.RESULTS_DIR / name) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    data.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(data.FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig2():
    segs, labels = data.load_segments("mat", "Subject00")
    ch = data.MAT_CLAB.index("F3")
    sig = {"rest": segs[labels == 0][0][:, ch], "task": segs[labels == 1][0][:, ch]}
    modes = {k: rlmd.modes(v, 5) for k, v in sig.items()}
    fig, axes = plt.subplots(6, 2, figsize=(11, 10))
    gs = axes[0, 0].get_gridspec()
    for ax in axes[0]:
        ax.remove()
    top = fig.add_subplot(gs[0, :])
    for (k, v), c in zip(sig.items(), (REST, TASK)):
        top.plot(v, color=c, lw=0.8, label=k)
    top.set(ylabel="Amplitude (µV)", title="4 s segment, MAT Subject00, F3")
    top.legend(ncol=2, loc="upper right")
    freq = np.fft.rfftfreq(len(sig["rest"]), 1 / data.MAT_FS)
    for i in range(5):
        for k, c in zip(sig, (REST, TASK)):
            m = modes[k][i]
            axes[i + 1, 0].plot(m, color=c, lw=0.8)
            spec = np.abs(np.fft.rfft(m)) ** 2
            axes[i + 1, 1].plot(freq, spec / spec.max() if spec.max() > 0 else spec, color=c, lw=0.8)
        axes[i + 1, 0].set_ylabel(f"Mode {i + 1}")
        axes[i + 1, 1].set_xlim(0, 60)
    axes[5, 0].set_xlabel("Samples")
    axes[5, 1].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_title("Modes (4 product functions + residual)")
    axes[1, 1].set_title("Normalized power spectrum")
    fig.tight_layout()
    save(fig, "fig2_decomposition.png")


def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, ds in zip(axes, ("mat", "stew")):
        curve = np.load(data.CACHE_DIR / f"overall_{ds}.npz")["curve"]
        ax.plot(np.arange(1, len(curve) + 1), curve, drawstyle="steps-post")
        ax.set(title=f"BAO fitness, {ds.upper()}", xlabel="Iteration", ylabel="Fitness value")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fig3_fitness.png")


def fig_oel(ds, name):
    z = np.load(data.CACHE_DIR / f"overall_{ds}.npz")
    y, pred, score = z["y"], z["ensemble_pred"], z["ensemble_score"]
    err = z["ensemble_min_error_curve"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(np.arange(1, len(err) + 1), err, "o-")
    axes[0].set(xlabel="Tuning iteration", ylabel="Minimum 10-fold CV error",
                title="Classification error")
    axes[0].grid(alpha=0.3)
    cm = np.array([[np.sum((y == a) & (pred == b)) for b in (0, 1)] for a in (0, 1)])
    axes[1].imshow(cm, cmap="Greens", vmin=0, vmax=cm.max() * 1.5)
    for a in (0, 1):
        for b in (0, 1):
            axes[1].text(b, a, cm[a, b], ha="center", va="center", fontsize=14)
    axes[1].set(xticks=(0, 1), yticks=(0, 1), xticklabels=("rest", "task"),
                yticklabels=("rest", "task"), xlabel="Predicted class", ylabel="True class",
                title="Confusion matrix")
    fpr, tpr, _ = roc_curve(y, score)
    axes[2].plot(fpr, tpr)
    axes[2].fill_between(fpr, tpr, alpha=0.2)
    axes[2].plot(cm[0, 1] / cm[0].sum(), cm[1, 1] / cm[1].sum(), "o", color="tab:red")
    axes[2].set(xlabel="False positive rate", ylabel="True positive rate",
                title=f"ROC, AUC = {roc_auc_score(y, score):.3f}")
    axes[2].grid(alpha=0.3)
    fig.suptitle(f"Optimized ensemble, {ds.upper()}")
    fig.tight_layout()
    save(fig, name)


def top_leads(ds, n=3):
    rows = sorted(read_csv(f"leadwise_{ds}.csv"), key=lambda r: -float(r["AC"]))
    return rows[:n]


def best_feature(ds, row):
    """Of a lead's BAO-selected features, the one that separates the classes
    best (largest one-way ANOVA F). Returns (name, raw values, labels)."""
    X, y, _, names = pipeline.load_features(ds, leads=[row["lead"]], standardize=False)
    best = None
    for short in row["selected"].split():
        j = names.index(f"{row['lead']}_{short}")
        f = stats.f_oneway(X[y == 0, j], X[y == 1, j]).statistic
        if best is None or f > best[0]:
            best = (f, short, X[:, j])
    return best[1], best[2], y


def fig_box(ds, name):
    rows = top_leads(ds)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, row in zip(axes, rows):
        short, v, y = best_feature(ds, row)
        ax.boxplot([v[y == 0], v[y == 1]], notch=True, tick_labels=["Rest", "Task"],
                   flierprops=dict(marker="+", markersize=4))
        ax.set(title=f"{row['lead']} ({float(row['AC']):.1f} %)", ylabel=short)
    fig.suptitle(f"Top-3 leads, {ds.upper()}: best-separating selected feature")
    fig.tight_layout()
    save(fig, name)


def fig8():
    import mne
    rename = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    info = mne.create_info([rename.get(c, c) for c in data.MAT_CLAB], data.MAT_FS, "eeg")
    info.set_montage("standard_1020")
    X, y, _, names = pipeline.load_features("mat", standardize=False)
    rows = top_leads("mat")
    fig, axes = plt.subplots(3, 2, figsize=(7, 10))
    for r, row in enumerate(rows):
        short, _, _ = best_feature("mat", row)
        cols = [names.index(f"{c}_{short}") for c in data.MAT_CLAB]
        means = [X[y == k][:, cols].mean(axis=0) for k in (0, 1)]
        lim = (min(m.min() for m in means), max(m.max() for m in means))
        for k, state in enumerate(("Rest", "Task")):
            im, _ = mne.viz.plot_topomap(means[k], info, axes=axes[r, k], show=False,
                                         cmap="viridis", vlim=lim)
            axes[r, k].set_title(f"{state}, {short} (top lead {row['lead']})")
            fig.colorbar(im, ax=axes[r, k], shrink=0.7)
    save(fig, "fig8_topomaps.png")


def fig9():
    rows = read_csv("tuning_mat.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, scope, title in zip(axes, ("overall", "leadwise"), ("Overall", "Lead-wise (F3)")):
        rs = [r for r in rows if r["scope"] == scope]
        lev = sorted((int(r["level"]), float(r["AC"])) for r in rs if r["bao_iterations"] == "100")
        its = sorted((int(r["bao_iterations"]), float(r["AC"])) for r in rs if r["level"] == "5")
        ax.plot(*zip(*lev), "o-", color="tab:blue", label="ours")
        ax.plot(*zip(*sorted(paper.TUNING[scope]["level"].items())), "o:", color="tab:blue",
                alpha=0.5, label="paper")
        ax.set_xlabel("Decomposition level", color="tab:blue")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks([3, 5, 7, 9])
        top = ax.twiny()
        top.plot(*zip(*its), "s--", color="tab:red", label="ours")
        top.plot(*zip(*sorted(paper.TUNING[scope]["bao_iterations"].items())), "s:",
                 color="tab:red", alpha=0.5, label="paper")
        top.set_xlabel("BAO iterations", color="tab:red")
        top.set_xticks([25, 50, 75, 100])
        ax.legend(loc="center left", title="Level")
        top.legend(loc="center right", title="Iterations")
        ax.set_title(title, y=1.15)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fig9_tuning.png")


if __name__ == "__main__":
    fig2()
    fig3()
    fig_oel("mat", "fig4_mat_ensemble.png")
    fig_oel("stew", "fig5_stew_ensemble.png")
    fig_box("mat", "fig6_mat_boxplots.png")
    fig_box("stew", "fig7_stew_boxplots.png")
    fig8()
    fig9()
