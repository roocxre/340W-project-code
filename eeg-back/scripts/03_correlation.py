"""Stage 3: channel correlation (Fig. 7) -> results/correlation.json, results/figures/fig7_*.png.

The Fig. 7 heatmaps are read back from the paper's PDF (backrepro.digitize)
and compared with the 16-channel Pearson matrix (released compute_correlation)
of every recording. The paper calls the two panels clean and noisy
no-workload recordings; the match says which recordings they are.
The group-level question, whether the noisy subjects' no-workload correlations
are lower, is then answered over all subjects.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backrepro import data, digitize  # noqa: E402
from backrepro.released import correlation_analysis as ca  # noqa: E402

IU = np.triu_indices(16, 1)


def plot_fig7(left, right, t_left, t_right, t_diff, path):
    """Fig. 7 layout: two coolwarm matrices and a bwr difference (left - right).

    The released plot_correlation_with_difference picks its colormap from
    whether a title contains 'No', so with these titles it would colour the
    panels differently from the paper; the heatmap settings are otherwise its own.
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    for ax, m, t, cmap in [(axes[0], left, t_left, "coolwarm"), (axes[1], right, t_right, "coolwarm"),
                           (axes[2], left - right, t_diff, "bwr")]:
        sns.heatmap(m, ax=ax, cmap=cmap, center=0, square=True, linewidths=0.4,
                    cbar=ax is axes[2], cbar_kws={"label": "Diff"} if ax is axes[2] else None)
        ax.set_title(t, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    out = {}
    corr = {k: ca.compute_correlation(data.load_raw(*k)) for k in data.iter_files()}

    dig, err = digitize.fig7_matrices()
    print(f"Fig. 7 digitised: mean colour error {err['clean']:.4f} (clean), {err['noisy']:.4f} (noisy) on a 0-1.7 RGB scale")
    out["fig7"] = {}
    for panel in ["clean", "noisy"]:
        d = dig[panel][IU]
        scores = sorted((float(np.sqrt(((corr[k].values[IU] - d) ** 2).mean())),
                         float(np.corrcoef(corr[k].values[IU], d)[0, 1]), k) for k in corr)
        best = scores[0]
        print(f"  '{panel}' panel: best sub{best[2][0]:02d}_{best[2][1]} (RMSE {best[0]:.3f}, r {best[1]:.3f}); "
              f"runner-up sub{scores[1][2][0]:02d}_{scores[1][2][1]} (RMSE {scores[1][0]:.3f}, r {scores[1][1]:.3f})")
        out["fig7"][panel] = dict(recording=f"sub{best[2][0]:02d}_{best[2][1]}", group=data.group(best[2][0]),
                                  rmse=best[0], r=best[1], runner_up_rmse=scores[1][0], runner_up_r=scores[1][1])
    a, b = (out["fig7"][p]["recording"] for p in ["clean", "noisy"])
    print(f"  -> 'clean' = {a}, 'noisy' = {b}: both from clean subject {a[3:5]}; "
          "the figure is rest vs high workload, not clean vs noisy")

    # Fig. 7 rebuilt from the identified recordings, with the paper's titles corrected
    plot_fig7(corr[(7, "nw")], corr[(7, "hw")], f"No Workload ({a})", f"High Workload ({b})",
              "Difference (No - High)", data.FIGURES / "fig7_correlation_sub07.png")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, panel in zip(axes, ["clean", "noisy"]):
        rec = out["fig7"][panel]["recording"]
        k = (int(rec[3:5]), rec[-2:])
        ax.scatter(dig[panel][IU], corr[k].values[IU], s=8)
        ax.plot([-0.2, 1], [-0.2, 1], color="0.6", lw=0.8)
        ax.set_title(f"Fig. 7 '{panel}' vs {rec}"); ax.set_xlabel("read from the paper"); ax.set_ylabel("computed")
    fig.tight_layout(); fig.savefig(data.FIGURES / "fig7_match.png", dpi=150); plt.close(fig)

    # Group level: no-workload correlation, clean vs noisy subjects
    mean_r = pd.DataFrame([dict(subject=s, cond=c, group=data.group(s), mean_r=corr[(s, c)].values[IU].mean())
                           for s, c in data.iter_files()])
    nw = mean_r[mean_r.cond == "nw"]
    a_, b_ = nw[nw.group == "clean"].mean_r, nw[nw.group == "noisy"].mean_r
    p = stats.mannwhitneyu(a_, b_).pvalue
    print(f"no-workload mean off-diagonal r: clean subjects median {a_.median():.3f}, noisy {b_.median():.3f}, "
          f"Mann-Whitney p = {p:.2f}")
    gm = {g: np.mean([corr[(s, "nw")].values for s in S], 0) for g, S in [("clean", data.CLEAN), ("noisy", data.NOISY)]}
    diff = gm["clean"] - gm["noisy"]
    print(f"  group-mean matrices, clean - noisy: off-diagonal range {diff[IU].min():+.3f} to {diff[IU].max():+.3f} "
          f"(paper's example pair: about -0.27 to +0.53)")
    wide = mean_r.pivot(index="subject", columns="cond", values="mean_r")
    print("  mean r by condition, all 40 subjects: " + ", ".join(f"{c} {wide[c].median():.3f}" for c in data.CONDITIONS))
    out["group"] = dict(nw_clean_median=float(a_.median()), nw_noisy_median=float(b_.median()), p=float(p),
                        diff_min=float(diff[IU].min()), diff_max=float(diff[IU].max()))

    names = data.CHANNELS
    plot_fig7(pd.DataFrame(gm["clean"], names, names), pd.DataFrame(gm["noisy"], names, names),
              "No Workload, mean of clean subjects", "No Workload, mean of noisy subjects",
              "Difference (clean - noisy)", data.FIGURES / "fig7_correlation_groups.png")

    with open(data.RESULTS / "correlation.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
