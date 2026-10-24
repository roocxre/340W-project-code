"""Is n decodable in Shin? Log band power per 40 s series (2 s windows averaged), LOSO."""
import sys
import numpy as np
from scipy.signal import welch
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "eeg-analysis"))
from eegrepro.data import load_clean, SUBJECTS, NBACK_SERIES, EEG_CLAB

SHARED3 = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2", "T7", "T8", "P7", "P8"]
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
X, y, g, seq = [], [], [], []
for s in SUBJECTS:
    r = load_clean(s, "nback")
    idx = r.channels(SHARED3)
    smp = r.mrk_sample
    ser = [i for i, c in enumerate(r.mrk_code) if c in NBACK_SERIES]
    for k, i in enumerate(ser):
        a = smp[i + 1]; b = smp[i + 20] + int(2 * r.fs)
        f, p = welch(r.x[a:b, idx].T, fs=r.fs, nperseg=int(2 * r.fs))
        feats = [np.log(p[:, (f >= lo) & (f < hi)].mean(1)) for lo, hi in BANDS.values()]
        X.append(np.concatenate(feats)); y.append(NBACK_SERIES[r.mrk_code[i]]); g.append(s); seq.append(k)
X, y, g, seq = map(np.asarray, (X, y, g, seq))
# per-subject z-scoring (subject normalisation, as a joined pipeline would do)
Xz = X.copy()
for s in SUBJECTS:
    m = g == s; Xz[m] = (X[m] - X[m].mean(0)) / X[m].std(0)
nch = len(SHARED3)
for bname, bi in zip(BANDS, range(3)):
    sub = {n: np.array([X[(g == s) & (y == n), bi*nch:(bi+1)*nch].mean() for s in SUBJECTS]) for n in (0, 2, 3)}
    d2, d3 = sub[2] - sub[0], sub[3] - sub[0]
    print(f"{bname:5s} mean over shared ch: 2-back vs 0: {np.mean(d2):+.3f} ({(d2<0).sum()}/26 lower, p={wilcoxon(d2).pvalue:.3g}); "
          f"3 vs 0: {np.mean(d3):+.3f} ({(d3<0).sum()}/26 lower, p={wilcoxon(d3).pvalue:.3g})")
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
for name, mask, chance in (("0/2/3", np.ones(len(y), bool), 1/3), ("0 vs 2", np.isin(y, [0, 2]), .5), ("2 vs 3", np.isin(y, [2, 3]), .5)):
    acc = []
    for s in SUBJECTS:
        tr = mask & (g != s); te = mask & (g == s)
        acc.append((clf.fit(Xz[tr], y[tr]).predict(Xz[te]) == y[te]).mean())
    print(f"LOSO {name}: {100*np.mean(acc):.1f}% (chance {100*chance:.0f}%), subjects above chance {sum(a > chance for a in acc)}/26")
# series position confound check: correlation of features with time in recording within subject
print("mean series index by n:", {n: round(seq[y == n].mean(), 1) for n in (0, 2, 3)})
