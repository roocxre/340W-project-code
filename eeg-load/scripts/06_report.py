"""Stage 6: paper-vs-ours tables -> results/paper_vs_ours.md."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loadrepro import classifiers, data, paper, pipeline  # noqa: E402

NAMES = {"mat": "MAT", "stew": "STEW"}


def read_csv(name):
    with open(data.RESULTS_DIR / name) as f:
        return list(csv.DictReader(f))


def triple(v):
    return f"{v[0]:.1f} / {v[1]:.0f} / {v[2]:.0f}"


def ours(r):
    return f"{float(r['AC']):.1f} / {float(r['SE']):.0f} / {float(r['SP']):.0f}"


def overall(ds, out):
    rows = read_csv(f"overall_{ds}.csv")
    z = np.load(data.CACHE_DIR / f"overall_{ds}.npz")
    sel = json.load(open(data.RESULTS_DIR / f"overall_{ds}_selection.json"))
    table = "1" if ds == "mat" else "2"
    out += [f"## {NAMES[ds]} overall (Table {table})", "",
            "AC / SE / SP in %. Subject-wise AC re-scores the same tuned model with no "
            "subject in both training and test folds.", "",
            "| Classifier | Paper | Ours | Ours, subject-wise AC |", "|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['classifier']} | {triple(paper.OVERALL[ds][r['classifier']])} | "
                   f"{ours(r)} | {float(r['subject_cv_AC']):.1f} |")
    curve, mask = z["curve"], z["mask"]
    p0, p1 = paper.FITNESS[ds]
    out += ["", f"- BAO kept {mask.sum()} of {len(mask)} features. Fitness went from "
            f"{curve[0]:.4f} to {curve[-1]:.4f} over 100 iterations (paper, Fig. 3: about {p0} to {p1}).",
            f"- Ensemble: {sel['params']['ensemble']}."]

    # One-way ANOVA, rest vs task, on each selected feature.
    X, y, _, _ = pipeline.load_features(ds)
    pvals = np.array([stats.f_oneway(X[y == 0, j], X[y == 1, j]).pvalue
                      for j in np.flatnonzero(mask)])
    out.append(f"- ANOVA, rest vs task: {np.sum(pvals < 0.01)} of {len(pvals)} selected "
               f"features differ at p < 0.01.")
    # One-way ANOVA across the six classifiers' per-fold accuracies.
    folds = [z[f"{f}_fold_acc"] for f in classifiers.FAMILIES]
    p = stats.f_oneway(*folds).pvalue
    el = folds[-1]
    beaten = [classifiers.LABELS[f] for f, acc in zip(classifiers.FAMILIES[:-1], folds[:-1])
              if stats.ttest_rel(el, acc).pvalue < 0.01 and el.mean() > acc.mean()]
    out += [f"- ANOVA across the six classifiers' fold accuracies: p = {p:.2g}. The ensemble "
            f"is better at p < 0.01 (paired t-test over folds) than: "
            f"{', '.join(beaten) if beaten else 'none'}.", ""]


def leadwise(ds, out):
    rows = {r["lead"]: r for r in read_csv(f"leadwise_{ds}.csv")}
    ref = paper.LEADWISE[ds]
    table = "3" if ds == "mat" else "4"
    out += [f"## {NAMES[ds]} lead-wise, optimized ensemble (Table {table})", "",
            "| Lead | Paper | Ours | Ours, subject-wise AC | Features kept |", "|---|---|---|---|---|"]
    for lead in data.DATASETS[ds]["clab"]:
        r = rows[lead]
        out.append(f"| {lead} | {triple(ref[lead])} | {ours(r)} | "
                   f"{float(r['subject_cv_AC']):.1f} | {r['n_selected']} |")
    leads = list(ref)
    a = [ref[c][0] for c in leads]
    b = [float(rows[c]["AC"]) for c in leads]
    rho = stats.spearmanr(a, b)
    rank = sorted(leads, key=lambda c: -float(rows[c]["AC"]))
    out += ["", f"- Best lead: {rank[0]} at {float(rows[rank[0]]['AC']):.1f} % "
            f"(paper: F3 at {ref['F3'][0]} %). Our top three: {', '.join(rank[:3])}.",
            f"- F3 ranks {rank.index('F3') + 1} of {len(rank)} here.",
            f"- Lead accuracies range {min(b):.1f}-{max(b):.1f} % (paper {min(a):.1f}-{max(a):.1f} %). "
            f"Spearman correlation with the paper's lead ranking: {rho.statistic:.2f} "
            f"(p = {rho.pvalue:.2g}).", ""]


def tuning(out):
    rows = read_csv("tuning_mat.csv")
    out += ["## Tuning parameters, MAT (Fig. 9)", "",
            "Ensemble accuracy in %. Paper values are read off the plot.", "",
            "| Setting | Paper, overall | Ours, overall | Paper, F3 | Ours, F3 |", "|---|---|---|---|---|"]

    def get(scope, level, it):
        return next(float(r["AC"]) for r in rows if r["scope"] == scope
                    and int(r["level"]) == level and int(r["bao_iterations"]) == it)

    for L in (3, 5, 7, 9):
        out.append(f"| L = {L}, 100 iterations | {paper.TUNING['overall']['level'][L]} | "
                   f"{get('overall', L, 100):.1f} | {paper.TUNING['leadwise']['level'][L]} | "
                   f"{get('leadwise', L, 100):.1f} |")
    for it in (25, 50, 75):
        out.append(f"| L = 5, {it} iterations | {paper.TUNING['overall']['bao_iterations'][it]} | "
                   f"{get('overall', 5, it):.1f} | {paper.TUNING['leadwise']['bao_iterations'][it]} | "
                   f"{get('leadwise', 5, it):.1f} |")
    out.append("")


def main():
    out = ["# Paper vs ours", "",
           "Yedukondalu et al. 2025, Sci. Rep. 15:842. Generated by `scripts/06_report.py`.", ""]
    for ds in ("mat", "stew"):
        overall(ds, out)
    for ds in ("mat", "stew"):
        leadwise(ds, out)
    tuning(out)
    (data.RESULTS_DIR / "paper_vs_ours.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
