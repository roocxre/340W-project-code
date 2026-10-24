"""Stage 5: repeated-measures ANOVA on frontal theta -> results/anova.csv.

Paper: 30 clean subjects, lw / mw / hw, F(2, 58) = 4.21, p = 0.02, post-hoc
lw-hw p = 0.02, lw-mw 0.12, mw-hw 0.19, theta rising monotonically. The
frontal electrodes, the power estimate and the post-hoc test are not given,
so every combination of the choices below is run. The primary row uses the
paper's own BEAM theta (released compute_topomap) over the six frontal leads.
Post-hoc tests are uncorrected paired t-tests.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backrepro import data, paper  # noqa: E402
from backrepro.released import beam_topography as bt, psd_analysis as pa  # noqa: E402

CHANNEL_SETS = {"Fp1 Fp2 F3 F4 F7 F8": data.FRONTAL, "Fp1 Fp2 F3 F4": ["Fp1", "Fp2", "F3", "F4"],
                "F3 F4 F7 F8": ["F3", "F4", "F7", "F8"], "F3 F4": ["F3", "F4"], "Fp1 Fp2": ["Fp1", "Fp2"]}
MEASURES = {"beam": "compute_topomap theta (µV²/Hz)", "abs": "compute_psd_mne + analyze_bands theta power",
            "rel": "theta power / 1-40 Hz power"}
WORK = ["lw", "mw", "hw"]


def theta_per_channel():
    rows = []
    for s in data.CLEAN:
        for c in WORK:
            raw = data.load_raw(s, c)
            beam = bt.compute_topomap(raw, 4, 8)[0] * 1e12
            sp = pa.compute_psd_mne(raw, fmin=1.0, fmax=40.0, n_fft=2048)
            bands = [pa.analyze_bands(sp.freqs, p) for p in sp.get_data()]
            rows.append(dict(subject=s, cond=c, beam=beam,
                             abs=np.array([b["Theta"]["power"] for b in bands]) * 1e12,
                             rel=np.array([b["Theta"]["relative_power"] for b in bands])))
    return pd.DataFrame(rows)


def run(df, chans, measure, log):
    idx = [data.CHANNELS.index(c) for c in chans]
    d = df[["subject", "cond"]].copy()
    d["y"] = [np.log10(v[idx].mean()) if log else v[idx].mean() for v in df[measure]]
    a = AnovaRM(d, "y", "subject", within=["cond"]).fit().anova_table.iloc[0]
    w = {c: d[d.cond == c].sort_values("subject").y.values for c in WORK}
    post = {f"p_{x}_{y}": stats.ttest_rel(w[x], w[y]).pvalue for x, y in [("lw", "mw"), ("lw", "hw"), ("mw", "hw")]}
    m = [w[c].mean() for c in WORK]
    return dict(channels=" ".join(chans), measure=measure, scale="log10" if log else "linear",
                F=a["F Value"], df1=int(a["Num DF"]), df2=int(a["Den DF"]), p=a["Pr > F"], **post,
                mean_lw=m[0], mean_mw=m[1], mean_hw=m[2],
                order="lw<mw<hw" if m[0] < m[1] < m[2] else "".join("<" if m[i] < m[i + 1] else ">" for i in range(2)))


def main():
    df = theta_per_channel()
    res = pd.DataFrame([run(df, chans, meas, log) for (_, chans), meas, log
                        in itertools.product(CHANNEL_SETS.items(), MEASURES, [False, True])])
    res.to_csv(data.RESULTS / "anova.csv", index=False, float_format="%.4g")
    t = paper.ANOVA
    print(f"paper: F({t['df'][0]}, {t['df'][1]}) = {t['F']}, p = {t['p']}; post-hoc lw-mw {t['posthoc'][('lw', 'mw')]}, "
          f"lw-hw {t['posthoc'][('lw', 'hw')]}, mw-hw {t['posthoc'][('mw', 'hw')]}; lw < mw < hw")
    p0 = res.iloc[0]
    print(f"primary (BEAM theta, six frontal leads, linear): F({p0.df1}, {p0.df2}) = {p0.F:.2f}, p = {p0.p:.2f}; "
          f"post-hoc lw-mw {p0.p_lw_mw:.2f}, lw-hw {p0.p_lw_hw:.2f}, mw-hw {p0.p_mw_hw:.2f}; "
          f"means {p0.mean_lw:.2f} / {p0.mean_mw:.2f} / {p0.mean_hw:.2f} µV²/Hz ({p0.order})")
    print(f"all {len(res)} variants: F {res.F.min():.2f}-{res.F.max():.2f}, smallest p {res.p.min():.3f}, "
          f"{(res.p < 0.05).sum()} with p < 0.05, {(res.order == 'lw<mw<hw').sum()} monotonic increasing; "
          f"order counts {res.order.value_counts().to_dict()}")
    print(res.sort_values("F", ascending=False).head(5)[["channels", "measure", "scale", "F", "p", "p_lw_mw", "p_lw_hw", "p_mw_hw", "order"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
