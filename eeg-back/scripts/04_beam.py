"""Stage 4: BEAM topographic maps (Fig. 8) -> results/beam.json, results/figures/fig8_*.png.

Band power per channel comes from the released compute_topomap (FIR band-pass,
2 s epochs, Welch n_fft 256, mean over epochs and bins), in µV²/Hz.

Fig. 8 is not MNE output: filled ellipses, a 16-step colour scale and no head
outline, which looks like the acquisition software's own BEAM display. Its
recordings are searched for anyway by sampling each map at the electrodes and
rank-correlating with every recording. The paper's two claims, frontal theta
higher under workload and posterior alpha higher at rest, are then tested
across subjects.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backrepro import data, digitize, paper  # noqa: E402
from backrepro.released import beam_topography as bt  # noqa: E402

FRONTAL_MIDLINE = ["Fp1", "Fp2", "F3", "F4"]   # the montage has no Fz
POSTERIOR = ["P3", "P4", "O1", "O2"]
WORK = ["lw", "mw", "hw"]


def band_power():
    out = {}
    for s, c in data.iter_files():
        raw = data.load_raw(s, c)
        for band, (lo, hi) in bt.BEAM_BANDS.items():
            p, info = bt.compute_topomap(raw, lo, hi)
            out[(s, c, band)] = p * 1e12
    return out, info


def plot_fig8(maps, info, path):
    """Fig. 8 layout: rows clean / noisy, columns theta W, NW, alpha W, NW.

    As in the paper, W and NW of one row and band share a colour scale. The
    released plot_beam_grid scales every map from 0 to its own maximum and has
    no colorbar, which makes maps incomparable, so it is not used.
    """
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    for i, g in enumerate(["clean", "noisy"]):
        for j, (b, w) in enumerate([(b, w) for b in bt.BEAM_BANDS for w in ("W", "NW")]):
            lo = min(maps[(g, b, "W")].min(), maps[(g, b, "NW")].min())
            hi = max(maps[(g, b, "W")].max(), maps[(g, b, "NW")].max())
            im, _ = mne.viz.plot_topomap(maps[(g, b, w)], info, axes=axes[i, j], show=False, contours=0,
                                         cmap=bt.DEFAULT_CMAP, vlim=(lo, hi))
            axes[i, j].set_title(f"{g}, {b}\n{'Workload' if w == 'W' else 'No Workload'}", fontsize=10)
            if w == "NW":
                fig.colorbar(im, ax=axes[i, j - 1:j + 1], shrink=0.7, label="µV²/Hz")
    fig.suptitle("Group-mean BEAM maps, released compute_topomap; workload = lw, mw, hw pooled")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def region(v, chans):
    return v[[data.CHANNELS.index(c) for c in chans]].mean()


def main():
    P, info = band_power()
    out = {}

    # Group-mean maps in the paper's layout
    maps = {(g, b, w): np.mean([P[(s, c, b)] for s in S for c in (WORK if w == "W" else ["nw"])], 0)
            for g, S in [("clean", data.CLEAN), ("noisy", data.NOISY)] for b in bt.BEAM_BANDS for w in ("W", "NW")}
    plot_fig8(maps, info, data.FIGURES / "fig8_beam_groups.png")
    for (g, b, w), v in maps.items():
        print(f"{g} {b} {w:2s}: {v.min():.1f}-{v.max():.1f} µV²/Hz, max at {data.CHANNELS[v.argmax()]}")
    print("paper colorbars: " + ", ".join(f"{g} {b} {lo}-{hi}" for (g, b), (lo, hi) in paper.FIG8_COLORBAR.items()))

    # Search for Fig. 8's recordings
    dig = digitize.fig8_levels(info)
    rng = np.random.default_rng(0)
    out["fig8_search"] = {}
    print("Fig. 8 maps vs every recording, Spearman rho at 16 electrodes (null: best of 160 shuffled maps):")
    for key, lev in dig.items():
        g, b, w = key
        rhos = sorted(((stats.spearmanr(P[(s, c, b)], lev)[0], f"sub{s:02d}_{c}") for s, c in data.iter_files()),
                      reverse=True)
        null = [max(stats.spearmanr(P[(s, c, b)], rng.permutation(lev))[0] for s, c in data.iter_files())
                for _ in range(50)]
        consistent = (rhos[0][1][-2:] == "nw") == (w == "NW") and (int(rhos[0][1][3:5]) in data.CLEAN) == (g == "clean")
        print(f"  {g} {b} {w:2s}: best {rhos[0][1]} rho {rhos[0][0]:.2f}, next {rhos[1][1]} {rhos[1][0]:.2f}; "
              f"null best median {np.median(null):.2f}; best match is {'' if consistent else 'not '}"
              f"the panel's own group and condition")
        out["fig8_search"]["_".join(key)] = dict(best=rhos[0][1], rho=float(rhos[0][0]), null_median=float(np.median(null)))

    # The paper's claims, per subject
    out["claims"] = {}
    for g, S in [("clean", data.CLEAN), ("noisy", data.NOISY), ("all", data.SUBJECTS)]:
        th_w = np.array([np.mean([region(P[(s, c, "Theta")], FRONTAL_MIDLINE) for c in WORK]) for s in S])
        th_n = np.array([region(P[(s, "nw", "Theta")], FRONTAL_MIDLINE) for s in S])
        al_w = np.array([np.mean([region(P[(s, c, "Alpha")], POSTERIOR) for c in WORK]) for s in S])
        al_n = np.array([region(P[(s, "nw", "Alpha")], POSTERIOR) for s in S])
        pt = stats.wilcoxon(th_w, th_n).pvalue
        pa = stats.wilcoxon(al_n, al_w).pvalue
        print(f"{g:5s} (n={len(S)}): frontal theta W > NW in {(th_w > th_n).sum()}/{len(S)}, "
              f"median W/NW {np.median(th_w / th_n):.2f}, Wilcoxon p = {pt:.3f} | "
              f"posterior alpha NW > W in {(al_n > al_w).sum()}/{len(S)}, median NW/W {np.median(al_n / al_w):.2f}, p = {pa:.3f}")
        out["claims"][g] = dict(theta_up=int((th_w > th_n).sum()), theta_p=float(pt), theta_ratio=float(np.median(th_w / th_n)),
                                alpha_down=int((al_n > al_w).sum()), alpha_p=float(pa), alpha_ratio=float(np.median(al_n / al_w)), n=len(S))

    with open(data.RESULTS / "beam.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
