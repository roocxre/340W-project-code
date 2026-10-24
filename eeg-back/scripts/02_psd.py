"""Stage 2: PSD analysis (Figs 5 and 6) -> results/psd.json, results/figures/fig5_*.png, fig6_*.png.

Fig. 5 is one subject. Its seven numbers (four peaks, three power ratios)
identify it: sub02, Fp1-A1, the released compute_psd_mne (MNE Welch with a
Hamming window, n_fft 2048). The released compute_psd_welch (scipy, Hann)
does not give them. The same statistics are then computed for every clean
subject to see whether the subject is representative.

Fig. 6 compares one clean and one noisy recording at n_fft 4096. No ordered
pair of released recordings gives its two peaks and its 4.01 power ratio.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backrepro import data, paper  # noqa: E402
from backrepro.released import psd_analysis as pa  # noqa: E402

CH = "Fp1-A1"
COLORS = {"nw": "#2E8B57", "lw": "#4472C4", "mw": "#FF8C00", "hw": "#E74C3C"}


def fp1_psd(subject, cond, n_fft, method="mne"):
    raw = data.load_raw(subject, cond, "raw16")
    if method == "mne":
        sp = pa.compute_psd_mne(raw, fmin=1.0, fmax=40.0, n_fft=n_fft, picks=[CH])
        return sp.freqs, sp.get_data()[0]
    f, p = pa.compute_psd_welch(raw.get_data(picks=[CH])[0], raw.info["sfreq"], nperseg=n_fft)
    m = (f >= 1) & (f <= 40)
    return f[m], p[m]


def fig5_stats(subject, method="mne"):
    out, base = {}, None
    for c in data.CONDITIONS:
        f, p = fp1_psd(subject, c, 2048, method)
        base = p.mean() if base is None else base
        out[c] = dict(peak=float(f[p.argmax()]), ratio=float(p.mean() / base),
                      bands=pa.analyze_bands(f, p), freqs=f, psd=p)
    return out


def matches_fig5(st):
    return all(abs(st[c]["peak"] - paper.FIG5[c][0]) < 0.01 and abs(st[c]["ratio"] - paper.FIG5[c][1]) < 0.006
               for c in data.CONDITIONS)


def plot_fig5(st, title, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, c in zip(axes.ravel(), data.CONDITIONS):
        name = data.CONDITION_NAMES[c].replace("Workload", "Cognitive Workload")
        ax.semilogy(st[c]["freqs"], st[c]["psd"], color=COLORS[c], lw=1.5, label=name)
        ax.set_title(f"{name}\npeak {st[c]['peak']:.2f} Hz, mean power x{st[c]['ratio']:.2f}", fontweight="bold")
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (V²/Hz)")
        ax.set_xlim(1, 40); ax.grid(alpha=0.3, ls="--")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig6_search(n_fft=4096):
    """Best ordered (clean, noisy) pair over all files for Fig. 6's peaks and ratio."""
    P = {(s, c): fp1_psd(s, c, n_fft) for s, c in data.iter_files()}
    st = {k: (f[p.argmax()], p.mean()) for k, (f, p) in P.items()}
    t = paper.FIG6
    exact = [(a, b) for a in st for b in st if a != b
             and abs(st[a][0] - t["clean_peak"]) < 0.01 and abs(st[b][0] - t["noisy_peak"]) < 0.01
             and abs(st[b][1] / st[a][1] - t["ratio"]) < 0.006]
    near = sorted((abs(np.log(st[b][1] / st[a][1] / t["ratio"])), a, b, st[b][1] / st[a][1])
                  for a in st for b in st if a != b
                  and abs(st[a][0] - t["clean_peak"]) < 0.01 and abs(st[b][0] - t["noisy_peak"]) < 0.01)
    return P, exact, near


def main():
    out = {}

    # Fig. 5: which subject, and which released function
    hits = {m: [s for s in data.SUBJECTS if matches_fig5(fig5_stats(s, m))] for m in ("mne", "welch")}
    print(f"Fig. 5 search over 40 subjects: compute_psd_mne matches {hits['mne']}, "
          f"compute_psd_welch matches {hits['welch']}")
    st = fig5_stats(2, "mne")
    print("sub02, Fp1-A1, compute_psd_mne n_fft 2048 (paper in brackets):")
    for c in data.CONDITIONS:
        print(f"  {c}: peak {st[c]['peak']:.2f} Hz [{paper.FIG5[c][0]}], ratio {st[c]['ratio']:.2f} [{paper.FIG5[c][1]}]")
    plot_fig5(st, "Fig. 5 reproduced: sub02, Fp1-A1, released compute_psd_mne (n_fft 2048)",
              data.FIGURES / "fig5_psd_sub02.png")
    out["fig5"] = dict(match_mne=hits["mne"], match_welch=hits["welch"],
                       sub02={c: dict(peak=st[c]["peak"], ratio=st[c]["ratio"]) for c in data.CONDITIONS})

    # Is sub02 representative? Same statistics for each clean subject.
    rows = {s: fig5_stats(s) for s in data.CLEAN}
    summ = {}
    for c in ["lw", "mw", "hw"]:
        r = np.array([rows[s][c]["ratio"] for s in data.CLEAN])
        summ[c] = dict(median_ratio=float(np.median(r)), q1=float(np.percentile(r, 25)), q3=float(np.percentile(r, 75)),
                       below_1=int((r < 1).sum()))
    alpha_nw = sum(8 <= rows[s]["nw"]["peak"] <= 13 for s in data.CLEAN)
    order = sum(rows[s]["lw"]["ratio"] < rows[s]["mw"]["ratio"] < rows[s]["hw"]["ratio"] for s in data.CLEAN)
    print("all 30 clean subjects, power relative to own nw (median [IQR], subjects below 1):")
    for c, v in summ.items():
        print(f"  {c}: {v['median_ratio']:.2f} [{v['q1']:.2f}-{v['q3']:.2f}], {v['below_1']}/30 below 1")
    print(f"  nw peak in the alpha band (8-13 Hz) in {alpha_nw}/30; lw < mw < hw ordering in {order}/30 "
          f"(1/6 = 5 expected by chance)")
    out["fig5_clean_subjects"] = dict(ratios=summ, nw_alpha_peak=alpha_nw, ordered=order)

    grand = {c: np.mean([rows[s][c]["psd"] for s in data.CLEAN], 0) for c in data.CONDITIONS}
    f = rows[1]["nw"]["freqs"]
    gst = {c: dict(freqs=f, psd=grand[c], peak=float(f[grand[c].argmax()]),
                   ratio=float(grand[c].mean() / grand["nw"].mean())) for c in data.CONDITIONS}
    plot_fig5(gst, "Same analysis, mean PSD of the 30 clean subjects", data.FIGURES / "fig5_psd_clean_mean.png")
    print("  grand mean of 30 clean subjects: " +
          ", ".join(f"{c} peak {gst[c]['peak']:.2f} Hz ratio {gst[c]['ratio']:.2f}" for c in data.CONDITIONS))
    out["fig5_grand_mean"] = {c: dict(peak=gst[c]["peak"], ratio=gst[c]["ratio"]) for c in data.CONDITIONS}

    # Fig. 6
    P, exact, near = fig6_search()
    print(f"Fig. 6 search, n_fft 4096, all {len(P) * (len(P) - 1)} ordered pairs: {len(exact)} exact matches")
    for e, a, b, r in near[:3]:
        print(f"  nearest with both peaks right: clean sub{a[0]:02d}_{a[1]} / noisy sub{b[0]:02d}_{b[1]}: ratio {r:.2f}")
    rec276 = (31, "lw")
    print(f"  record 276 (docstring 'Clean_276.EDF') = sub31_lw: peak {P[rec276][0][P[rec276][1].argmax()]:.2f} Hz "
          f"[Fig. 6 clean {paper.FIG6['clean_peak']}]; record 256 ('Noisy_256.EDF') is not released")
    gc = np.mean([P[k][1] for k in P if k[0] in data.CLEAN], 0)
    gn = np.mean([P[k][1] for k in P if k[0] in data.NOISY], 0)
    ratio = float(gn.mean() / gc.mean())
    print(f"  group means, all conditions: noisy/clean mean power {ratio:.2f} [paper's example pair: {paper.FIG6['ratio']}]")
    out["fig6"] = dict(exact_matches=len(exact), nearest=[(f"sub{a[0]:02d}_{a[1]}", f"sub{b[0]:02d}_{b[1]}", r) for _, a, b, r in near[:3]],
                       group_ratio=ratio)

    fig, ax = plt.subplots(figsize=(9, 5))
    fq = P[rec276][0]
    ax.semilogy(fq, P[rec276][1], color="#2E86C1", lw=1.2, label="sub31_lw = record 276 ('Clean_276')")
    ax.semilogy(fq, gc, color="#1B4F72", lw=2, label="mean, clean subjects 01-30")
    ax.semilogy(fq, gn, color="#E74C3C", lw=2, label="mean, noisy subjects 31-40")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (V²/Hz), Fp1-A1"); ax.set_xlim(1, 40)
    ax.set_title(f"Fig. 6 counterpart (n_fft 4096): group noisy/clean ratio {ratio:.2f}, paper's pair {paper.FIG6['ratio']}")
    ax.legend(frameon=False); ax.grid(alpha=0.3, ls="--")
    fig.tight_layout(); fig.savefig(data.FIGURES / "fig6_clean_vs_noisy.png", dpi=150); plt.close(fig)

    with open(data.RESULTS / "psd.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
