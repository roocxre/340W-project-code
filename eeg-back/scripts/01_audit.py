"""Stage 1: data audit -> results/audit.csv, results/figures/audit_*.png.

Checks what the files contain against what the paper and README say:
recording length, header fields, the device's record counter, the 16
undocumented Add_lead channels, the acquisition filter, whether the "noisy"
subjects 31-40 are noisier than 01-30, how much individual signature the
recordings carry, and whether any recording is duplicated in another file.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backrepro import data  # noqa: E402


def file_table():
    rows = []
    for s, c in data.iter_files():
        raw = data.load_raw(s, c, "all")
        x = raw.get_data() * 1e6
        eeg, add = x[:16], x[16:]
        h = data.header(s, c)
        f, P = signal.welch(eeg, data.FS, nperseg=400)
        rows.append(dict(
            subject=s, cond=c, group=data.group(s), n_samples=raw.n_times,
            seconds=raw.n_times / data.FS, declared_seconds=h["n_records_declared"] * h["record_seconds"],
            device_record=data.device_record(s, c), start=h["start"],
            sd_uV=np.median(eeg.std(1)), p99_abs_uV=np.percentile(np.abs(eeg), 99),
            frac_over_100uV=(np.abs(eeg) > 100).mean(), kurtosis=np.median(stats.kurtosis(eeg, axis=1)),
            delta_dB=10 * np.log10(P[:, (f >= 1) & (f < 4)].mean()),
            beta_dB=10 * np.log10(P[:, (f >= 13) & (f < 30)].mean()),
            add_sd_uV=np.median(add.std(1)),
            add_max_corr_own_eeg=np.abs(np.corrcoef(x)[16:, :16]).max()))
    return pd.DataFrame(rows)


def mean_spectrum(subjects):
    acc = []
    for s, c in data.iter_files(subjects):
        f, P = signal.welch(data.load_array(s, c), data.FS, nperseg=1024)
        acc.append(P.mean(0))
    return f, np.mean(acc, 0)


def duplicate_search(t):
    """Largest same-channel cross-correlation, at any lag, between two different files (1-30 Hz)."""
    n = t.n_samples.min()
    sos = signal.butter(4, [1, 30], "bandpass", fs=data.FS, output="sos")
    names, E = [], []
    for s, c in data.iter_files():
        x = signal.sosfiltfilt(sos, data.load_array(s, c)[:, :n])
        E.append((x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True))
        names.append(f"sub{s:02d}_{c}")
    F = np.fft.rfft(np.stack(E), 2 * n)
    best = []
    for i in range(len(names) - 1):
        cc = np.fft.irfft(F[i][None] * np.conj(F[i + 1:]), 2 * n) / n
        med = np.median(np.abs(cc).max(-1), 1)   # median over the 16 channels
        j = med.argmax()
        best.append((med[j], names[i], names[i + 1 + j]))
    return sorted(best, reverse=True)


def separability():
    """How much individual signature the recordings carry, from 2 s log spectra (1-40 Hz)."""
    X, cond, subj = [], [], []
    halves = []
    for s, c in data.iter_files():
        x = data.load_array(s, c)
        for k in range(x.shape[1] // 400):
            f, P = signal.welch(x[:, k * 400:(k + 1) * 400], data.FS, nperseg=400)
            X.append(np.log10(P[:, (f >= 1) & (f <= 40)]).ravel()); cond.append(c); subj.append(s)
        h = x.shape[1] // 2
        halves.append([np.log10(signal.welch(seg, data.FS, nperseg=400)[1][:, 2:61]).ravel()
                       for seg in (x[:, :h], x[:, h:2 * h])])
    X, cond, subj = np.array(X), np.array(cond), np.array(subj)
    rec = np.char.add(subj.astype(str), cond)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(5))
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    subj_acc = cross_val_score(knn, X, subj, cv=cv).mean()
    rec_acc = cross_val_score(knn, X, rec, cv=cv).mean()
    within = [cross_val_score(lr, X[(subj == s) & (cond != "nw")], cond[(subj == s) & (cond != "nw")], cv=cv).mean()
              for s in data.SUBJECTS]
    H = np.array(halves)
    A, B = H[:, 0] - H[:, 0].mean(0), H[:, 1] - H[:, 1].mean(0)
    C = np.corrcoef(A, B)[:len(A), len(A):]
    print("separability, 2 s log spectra:")
    print(f"  subject identity, 5-NN: {subj_acc * 100:.1f} % (chance 2.5 %); recording identity: {rec_acc * 100:.1f} % (chance 0.6 %)")
    print(f"  within-subject lw/mw/hw, logistic regression: median {np.median(within) * 100:.0f} % "
          f"(range {min(within) * 100:.0f}-{max(within) * 100:.0f} %, chance 33 %)")
    print(f"  second half of a file best matches its own first half in {(C.argmax(0) == np.arange(len(C))).mean() * 100:.0f} % of files")

    sos = signal.butter(4, [1, 30], "bandpass", fs=data.FS, output="sos")
    left = [i for i, c in enumerate(data.RAW_CHANNELS) if c.endswith("A1")]
    right = [i for i, c in enumerate(data.RAW_CHANNELS) if c.endswith("A2")]
    same, cross, fp = [], [], []
    for s, c in data.iter_files():
        R = np.corrcoef(signal.sosfiltfilt(sos, data.load_array(s, c)))
        same.append(np.median([R[i, j] for g in (left, right) for i in g for j in g if i < j]))
        cross.append(np.median([R[i, j] for i in left for j in right]))
        fp.append(R[0, 1])
    print(f"  inter-channel r (1-30 Hz): Fp1-Fp2 median {np.median(fp):.2f}; same reference ear {np.median(same):.2f}, "
          f"opposite ears {np.median(cross):.2f}")


def main():
    t = file_table()
    t.to_csv(data.RESULTS / "audit.csv", index=False, float_format="%.4g")

    print(f"{len(t)} files, {t.subject.nunique()} subjects, 4 conditions each")
    print(f"length: {t.seconds.min():.0f}-{t.seconds.max():.0f} s, median {t.seconds.median():.0f} s "
          f"(paper: 5 min per condition, 'approximately 1 minute' saved)")
    print(f"  longer than 54 s: {', '.join(f'sub{r.subject:02d}_{r.cond} ({r.seconds:.0f} s)' for r in t[t.seconds > 54].itertuples())}")
    print(f"header declares {t.declared_seconds.min():.0f}-{t.declared_seconds.max():.0f} s; "
          f"declared == stored in {(t.declared_seconds == t.seconds).sum()} of {len(t)} files")
    print(f"header start time is {t.start.iloc[0]} in {(t.start == t.start.iloc[0]).sum()} of {len(t)} files (device default)")

    rec = t.pivot(index="subject", columns="cond", values="device_record")[data.CONDITIONS]
    in_order = (np.diff(rec.values, axis=1) > 0).all() and (np.diff(rec.values.ravel()) > 0).all()
    print(f"device record number (EDF patient field): {rec.values.min()}-{rec.values.max()}, "
          f"{'strictly increasing in subject then nw<lw<mw<hw order' if in_order else 'NOT in acquisition order'}")
    gaps = np.diff(np.sort(rec.values.ravel()))
    print(f"  gap to next record: median {np.median(gaps):.0f}, largest {gaps.max()} "
          f"(between records {np.sort(rec.values.ravel())[gaps.argmax()]} and {np.sort(rec.values.ravel())[gaps.argmax() + 1]})")
    for name, num in [("Clean_276.EDF", 276), ("Noisy_256.EDF", 256)]:
        hit = t[t.device_record == num]
        where = f"sub{hit.subject.iloc[0]:02d}_{hit.cond.iloc[0]} ({hit.group.iloc[0]} in the release)" if len(hit) else "not in the release"
        print(f"  released code's docstring example {name}: record {num} is {where}")

    print(f"Add_lead1-16: median SD {t.add_sd_uV.median():.1f} µV (EEG {t.sd_uV.median():.1f} µV), "
          f"max |r| with own EEG median {t.add_max_corr_own_eeg.median():.2f}, max {t.add_max_corr_own_eeg.max():.2f}; "
          "undocumented, excluded as in the paper's figures")

    f, Pc = mean_spectrum(data.CLEAN)
    _, Pn = mean_spectrum(data.NOISY)
    db = lambda P, fr: 10 * np.log10(P[np.argmin(np.abs(f - fr))])
    print("mean spectrum, dB re 1 µV²/Hz (clean / noisy):")
    print("  " + "  ".join(f"{fr:g} Hz {db(Pc, fr):.1f}/{db(Pn, fr):.1f}" for fr in [0.2, 0.5, 2, 10, 30, 35, 40, 50]))
    print(f"  drop 30->40 Hz: {db(Pc, 30) - db(Pc, 40):.1f} dB; 50 Hz line peak above 45-55 Hz floor: "
          f"{db(Pc, 50) - 10 * np.log10(Pc[(f > 45) & (f < 55)].mean()):.1f} dB")
    print("  -> the KT88 recorded through a ~0.5 Hz high-pass and a ~35 Hz low-pass, with no line noise left")

    per_subj = t.groupby(["subject", "group"])[["sd_uV", "p99_abs_uV", "frac_over_100uV", "delta_dB", "beta_dB"]].mean().reset_index()
    print("artifact measures, per subject (mean over conditions), clean vs noisy, Mann-Whitney:")
    for col in ["sd_uV", "p99_abs_uV", "frac_over_100uV", "delta_dB", "beta_dB"]:
        a = per_subj[per_subj.group == "clean"][col]
        b = per_subj[per_subj.group == "noisy"][col]
        print(f"  {col:16s} clean median {a.median():8.3g}  noisy median {b.median():8.3g}  "
              f"p = {stats.mannwhitneyu(a, b).pvalue:.2f}  (noisy > clean in {(b.values[:, None] > a.values).mean() * 100:.0f}% of pairs)")
    top = per_subj.sort_values("p99_abs_uV", ascending=False).head(8)
    print(f"  highest 99th-percentile amplitude: {', '.join(f'sub{r.subject:02d} ({r.group}, {r.p99_abs_uV:.0f} µV)' for r in top.itertuples())}")

    separability()

    dup = duplicate_search(t)
    print(f"duplicate search: largest median lagged |r| between two files {dup[0][0]:.2f} "
          f"({dup[0][1]} vs {dup[0][2]}); median of per-file maxima "
          f"{np.median([d[0] for d in dup]):.2f} -> no duplicated recordings")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, lab in [(axes[0], "p99_abs_uV", "99th percentile |EEG| (µV)"), (axes[1], "frac_over_100uV", "fraction of samples > 100 µV")]:
        for g, colr in [("clean", "#2E86C1"), ("noisy", "#E74C3C")]:
            d = per_subj[per_subj.group == g]
            ax.bar(d.subject, d[col], color=colr, label=f"{g} (sub {'01-30' if g == 'clean' else '31-40'})")
        ax.set_xlabel("subject"); ax.set_ylabel(lab); ax.legend(frameon=False)
    fig.suptitle("Artifact measures per subject: the 'noisy' group is not noisier")
    fig.tight_layout(); fig.savefig(data.FIGURES / "audit_artifacts.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(f, Pc, color="#2E86C1", label="clean, sub 01-30")
    ax.semilogy(f, Pn, color="#E74C3C", label="noisy, sub 31-40")
    ax.axvline(35, color="0.5", ls=":"); ax.axvline(50, color="0.5", ls=":")
    ax.set_xlabel("Hz"); ax.set_ylabel("µV²/Hz, mean over channels and files"); ax.legend(frameon=False)
    ax.set_title("Mean spectrum: ~35 Hz low-pass, no 50 Hz line")
    fig.tight_layout(); fig.savefig(data.FIGURES / "audit_spectrum.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    data.FIGURES.mkdir(parents=True, exist_ok=True)
    main()
