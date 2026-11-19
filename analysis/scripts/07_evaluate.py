"""Stage 7: score every model once on the held-out test subjects; tables and figures.

    ../synthesis/.venv/bin/python scripts/07_evaluate.py

Reads results/predictions/<task>__<model>.npz from stages 4-6.

Baselines (what the feature-based candidates must beat):
    y-leaky       Yedukondalu pipeline, tuned and estimated with window-level CV (as published)
    y-corrected   the same pipeline with subject-wise folds in BAO and in tuning
    lr            logistic regression on band power (the synthesis viability baseline)
    eegnet        EEGNet on the raw windows (the deep end-to-end reference)
Candidates: svm, mlp, rf, xgb, riemann, and "vote".

Selection never looks at test:
- the Yedukondalu entry per protocol is the paper's proposed model, the optimized ensemble
  (all six families are reported in the leakage table);
- the vote averages the probabilities of the candidates that pass a validation gate:
  every class's recall and precision above chance (1/k and the class's prevalence), and the
  lower 95 % bound of AUC, bootstrapped over validation subjects, above 50 %;
- the "best candidate" is the candidate (or vote) with the best validation balanced accuracy.

Test metrics: accuracy, balanced accuracy (95 % bootstrap CI over test subjects), Cohen's kappa,
AUC (macro one-vs-rest), per-class precision and recall, per-subject accuracy, and a 20 s causal
smoothing (mean probability over the window and the 4 before it in the same block). Each
candidate is compared with y-corrected and with eegnet by a paired subject bootstrap.

Outputs: results/eval_*.csv, results/figures/eval_*.png; print to results/log_07_evaluate.txt.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt  # noqa: E402

import nbackml as ml  # noqa: E402
import vizstyle as vz  # noqa: E402

FAMILIES = ["tree", "discriminant", "naive_bayes", "knn", "svm", "ensemble"]
FAM_LABEL = {"tree": "Tree", "discriminant": "Discriminant", "naive_bayes": "Naive Bayes", "knn": "KNN",
             "svm": "SVM", "ensemble": "Ensemble (EL)"}
BASELINES = ["y-leaky", "y-corrected", "lr", "eegnet"]
CANDIDATES = ["svm", "mlp", "rf", "xgb", "riemann"]
ORDER = BASELINES + CANDIDATES + ["vote"]
LABEL = {"y-leaky": "Yedukondalu ensemble, leaky CV", "y-corrected": "Yedukondalu ensemble, corrected CV",
         "lr": "Logistic regression", "eegnet": "EEGNet (raw EEG)", "svm": "RBF SVM", "mlp": "MLP",
         "rf": "Random forest", "xgb": "XGBoost", "riemann": "Riemannian tangent space", "vote": "Gated soft vote"}
# for small panels (confusion matrices)
SHORT = {**LABEL, "y-leaky": "Yedukondalu, leaky", "y-corrected": "Yedukondalu, corrected",
         "eegnet": "EEGNet", "riemann": "Riemannian TS"}
# one colour per model across every figure: categorical slots in fixed order, plus two neutrals
COLOR = {"y-corrected": vz.SERIES[0], "eegnet": vz.SERIES[1], "lr": vz.SERIES[2], "svm": vz.SERIES[3],
         "mlp": vz.SERIES[4], "rf": vz.SERIES[5], "xgb": vz.SERIES[6], "riemann": vz.SERIES[7],
         "y-leaky": vz.MUTED, "vote": vz.INK}
N_BOOT = 2000


def pred_file(task, model):
    p = ml.PRED / f"{task}__{model}.npz"
    return dict(np.load(p)) if p.exists() else None


def dev_json(task, name):
    if name.startswith("yedukondalu-"):
        _, protocol, norm, family = name.split("-", 3)
        f = ml.CACHE / "baselines" / f"{task}_{protocol}_{norm}" / f"{family}.json"
    else:
        f = ml.CACHE / "models" / f"{task}_{name}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def pick_yedukondalu(task, protocol):
    """The paper's proposed model, the optimized ensemble (bagging / AdaBoost / boosting over
    BAO-selected R-LMD features, Tables 1-2). The other five families are in the leakage table."""
    return "ensemble" if dev_json(task, f"yedukondalu-{protocol}-pooled-ensemble") else None


class Split:
    """One model's predictions on one split, aligned with that split's task frame."""

    def __init__(self, proba, y, df):
        self.proba, self.y, self.df = proba, y, df
        self.g = df.subject.to_numpy()

    @classmethod
    def from_npz(cls, npz, split, df, task):
        pos = pd.Series(np.arange(len(npz[f"{split}_row"])), index=npz[f"{split}_row"])
        idx = pos.reindex(df.row_id.to_numpy()).to_numpy()
        assert not np.isnan(idx).any(), "predictions do not cover the split"
        return cls(npz[f"{split}_proba"][idx.astype(int)], ml.labels(df, task), df)

    def counts(self, proba=None):
        proba = self.proba if proba is None else proba
        subs = np.unique(self.g)
        k = proba.shape[1]
        hit = proba.argmax(1) == self.y
        corr, tot = np.zeros((len(subs), k)), np.zeros((len(subs), k))
        for i, s in enumerate(subs):
            m = self.g == s
            for c in range(k):
                mc = m & (self.y == c)
                tot[i, c], corr[i, c] = mc.sum(), hit[mc].sum()
        return corr, tot


def boot_bacc(corr, tot, draws):
    c, t = corr[draws].sum(1), tot[draws].sum(1)
    return 100 * np.nanmean(np.where(t > 0, c / np.maximum(t, 1), np.nan), axis=1)


def auc_ci(s, n_boot=500, seed=0):
    """95 % bootstrap interval of AUC over subjects."""
    rng = np.random.default_rng(seed)
    subs = np.unique(s.g)
    by = {u: np.flatnonzero(s.g == u) for u in subs}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([by[u] for u in rng.choice(subs, len(subs))])
        if len(np.unique(s.y[idx])) < s.proba.shape[1]:
            continue
        vals.append(ml.metrics(s.y[idx], s.proba[idx])["auc"])
    return np.percentile(vals, [2.5, 97.5])


def gate(s):
    """Validation gate for the vote: no class below chance on recall or precision, AUC CI above 50."""
    k = s.proba.shape[1]
    prec, rec, _, sup = precision_recall_fscore_support(s.y, s.proba.argmax(1), labels=list(range(k)),
                                                        zero_division=0)
    prev = sup / sup.sum()
    lo, _ = auc_ci(s)
    ok = bool((rec > 1 / k).all() and (prec > prev).all() and lo > 50)
    return dict(passes=ok, min_recall=100 * rec.min(), min_precision_over_prevalence=100 * (prec - prev).min(),
                auc_ci_lo=lo)


def evaluate_task(task):
    frames = {s: ml.frame(s, task) for s in ("val", "test")}
    k = len(ml.TASKS[task]["levels"])
    names = {}
    for protocol in ("leaky", "corrected"):
        fam = pick_yedukondalu(task, protocol)
        if fam:
            names[f"y-{protocol}"] = f"yedukondalu-{protocol}-pooled-{fam}"
    for m in ["lr", "eegnet"] + CANDIDATES:
        if pred_file(task, m) is not None:
            names[m] = m
    P = {}
    for e, n in names.items():
        npz = pred_file(task, n)
        P[e] = {s: Split.from_npz(npz, s, frames[s], task) for s in ("val", "test")}

    gates = {c: gate(P[c]["val"]) for c in CANDIDATES if c in P}
    members = [c for c, gt in gates.items() if gt["passes"]]
    if members:
        P["vote"] = {s: Split(np.mean([P[c][s].proba for c in members], 0), P[members[0]][s].y, frames[s])
                     for s in ("val", "test")}
        names["vote"] = "vote"

    test0 = next(iter(P.values()))["test"]
    subs = np.unique(test0.g)
    draws = np.random.default_rng(0).integers(0, len(subs), (N_BOOT, len(subs)))
    boots = {e: boot_bacc(*p["test"].counts(), draws) for e, p in P.items()}
    rows, per_class = [], []
    for e in [m for m in ORDER if m in P]:
        t, v = P[e]["test"], P[e]["val"]
        m = ml.metrics(t.y, t.proba)
        ms = ml.metrics(t.y, ml.smooth(t.proba, t.df))
        sub_acc = ml.per_subject_acc(t.y, t.proba, t.g)
        j = dev_json(task, names[e]) if e != "vote" else {}
        fam = names[e].rsplit("-", 1)[1] if e.startswith("y-") else ""
        row = dict(task=task, model=e, group="baseline" if e in BASELINES else "candidate",
                   label=LABEL[e] + (f" [{FAM_LABEL[fam]}]" if fam else ""),
                   dev_cv_acc=j.get("cv", {}).get("acc"),
                   dev_cv=("window 10-fold" if e == "y-leaky" else "subject 10-fold" if e == "y-corrected"
                           else "subject 5-fold" if j.get("cv") else ""),
                   val_bacc=ml.metrics(v.y, v.proba)["bacc"],
                   vote_member=e in members, gate=gates.get(e, {}).get("passes"),
                   test_acc=m["acc"], test_bacc=m["bacc"], bacc_lo=np.percentile(boots[e], 2.5),
                   bacc_hi=np.percentile(boots[e], 97.5), test_kappa=m["kappa"], test_auc=m["auc"],
                   subj_acc_sd=sub_acc.std(ddof=1), subj_above_chance=f"{int((sub_acc > 100 / k).sum())}/{len(sub_acc)}",
                   smooth20_bacc=ms["bacc"], smooth20_auc=ms["auc"])
        for ref in ("y-corrected", "eegnet"):
            if ref in boots and e != ref and e not in ("y-leaky",):
                d = boots[e] - boots[ref]
                row[f"diff_vs_{ref}"] = m["bacc"] - ml.metrics(P[ref]["test"].y, P[ref]["test"].proba)["bacc"]
                row[f"diff_vs_{ref}_lo"], row[f"diff_vs_{ref}_hi"] = np.percentile(d, [2.5, 97.5])
        rows.append(row)
        prec, rec, f1, sup = precision_recall_fscore_support(t.y, t.proba.argmax(1), labels=list(range(k)),
                                                             zero_division=0)
        per_class += [dict(task=task, model=e, n=ml.TASKS[task]["levels"][c], precision=100 * prec[c],
                           recall=100 * rec[c], f1=100 * f1[c], support=int(sup[c])) for c in range(k)]
    S = pd.DataFrame(rows)
    cand = S[S.model.isin(CANDIDATES + ["vote"])]
    S["best_by_val"] = S.model.eq(cand.loc[cand.val_bacc.idxmax(), "model"]) if len(cand) else False
    G = pd.DataFrame([dict(task=task, model=c, **g) for c, g in gates.items()])
    agree = None
    if len(members) > 1:   # how often the vote's members agree on validation
        pv = {c: P[c]["val"].proba.argmax(1) for c in members}
        agree = pd.DataFrame([[100 * np.mean(pv[a] == pv[b]) for b in members] for a in members],
                             index=members, columns=members)
    return S, pd.DataFrame(per_class), G, agree, P


def leakage_table():
    rows = []
    for task in ml.TASKS:
        for fam in FAMILIES:
            for protocol in ("leaky", "corrected"):
                j = dev_json(task, f"yedukondalu-{protocol}-pooled-{fam}")
                if j:
                    rows.append(dict(task=task, family=FAM_LABEL[fam], protocol=protocol, dev_cv_acc=j["cv"]["acc"],
                                     subject_rescore_acc=j.get("cv_subject_rescore", {}).get("acc"),
                                     val_acc=j["val"]["acc"], test_acc=j["test"]["acc"], test_bacc=j["test"]["bacc"],
                                     test_auc=j["test"]["auc"], inflation=j["cv"]["acc"] - j["test"]["acc"],
                                     params=json.dumps(j["params"])))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue_seq", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])


def fig_leakage(L):
    tasks = [t for t in ml.TASKS if t in set(L.task)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.2, 3.1), sharey=True, squeeze=False)
    for ax, task in zip(axes[0], tasks):
        q = L[L.task == task]
        fams = [FAM_LABEL[f] for f in FAMILIES if FAM_LABEL[f] in set(q.family)]
        ys = np.arange(len(fams))
        for off, protocol, color in [(-0.16, "leaky", COLOR["y-leaky"]), (0.16, "corrected", COLOR["y-corrected"])]:
            r = q[q.protocol == protocol].set_index("family").reindex(fams)
            ax.hlines(ys + off, r.test_acc, r.dev_cv_acc, color=color, lw=1.4)
            ax.scatter(r.dev_cv_acc, ys + off, color=color, s=26, zorder=3, label=f"{protocol}: CV estimate")
            ax.scatter(r.test_acc, ys + off, facecolor=vz.SURFACE, edgecolor=color, s=26, lw=1.4, zorder=3,
                       label=f"{protocol}: test subjects")
        ax.axvline(100 / len(ml.TASKS[task]["levels"]), color=vz.MUTED, lw=0.8, ls="--")
        ax.set_yticks(ys, fams)
        ax.set_title(ml.TASKS[task]["title"], loc="left", fontsize=8.5)
        ax.set_xlabel("accuracy (%)")
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(0, -0.22), ncol=4, fontsize=7.5)
    fig.suptitle("Yedukondalu pipeline: development CV estimate (filled) vs held-out test subjects (open)",
                 x=0.02, y=1.04, ha="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    vz.save(fig, ml.FIG / "eval_leakage.png")


def fig_summary(S):
    tasks = [t for t in ml.TASKS if t in set(S.task)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.2, 3.6), sharey=True, squeeze=False)
    for ax, task in zip(axes[0], tasks):
        q = S[S.task == task].set_index("model").reindex([m for m in ORDER if m in set(S[S.task == task].model)])
        ys = np.arange(len(q))[::-1]
        for y, (m, r) in zip(ys, q.iterrows()):
            ax.hlines(y, r.bacc_lo, r.bacc_hi, color=COLOR[m], lw=1.4)
            ax.scatter(r.test_bacc, y, color=COLOR[m], s=34 if r.best_by_val else 24, zorder=3,
                       marker="D" if r.best_by_val else "o")
            ax.text(r.bacc_hi + 0.8, y, f"{r.test_bacc:.1f}", va="center", fontsize=7, color=vz.INK2)
        nb = int(q.group.eq("baseline").sum())
        ax.axhline(ys[nb - 1] - 0.5, color=vz.GRID, lw=1.0)
        ax.axvline(100 / len(ml.TASKS[task]["levels"]), color=vz.MUTED, lw=0.8, ls="--")
        ax.set_yticks(ys, [LABEL[m] for m in q.index], fontsize=7.5)
        ax.set_title(ml.TASKS[task]["title"], loc="left", fontsize=8.5)
    fig.supxlabel("test balanced accuracy (%)", fontsize=9, color=vz.INK2)
    fig.suptitle("Held-out test subjects: balanced accuracy and 95 % subject-bootstrap CI\n"
                 "baselines above the line, candidates below; ◆ = best candidate by validation; dashed = chance",
                 x=0.02, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    vz.save(fig, ml.FIG / "eval_summary.png")


def fig_confusion(task, S, P):
    models = [m for m in ORDER if m in P]
    cols = 5
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 1.75 * rows + 0.5), squeeze=False)
    lv = ml.TASKS[task]["levels"]
    q = S[S.task == task].set_index("model")
    for ax, m in zip(axes.ravel(), models):
        t = P[m]["test"]
        cm = ml.confusion(t.y, t.proba, len(lv)) * 100
        ax.imshow(cm, cmap=seq_cmap(), vmin=0, vmax=100)
        ax.set_xticks(range(len(lv)), lv, fontsize=7)
        ax.set_yticks(range(len(lv)), lv, fontsize=7)
        ax.grid(False)
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 55 else vz.INK)
        ax.set_title(f"{SHORT[m]}\nbacc {q.loc[m, 'test_bacc']:.1f}", fontsize=7, loc="left")
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("true n", fontsize=7.5)
    for ax in axes[-1, :]:
        ax.set_xlabel("predicted n", fontsize=7.5)
    fig.suptitle(f"{ml.TASKS[task]['title']}: confusion matrices on test subjects (% of each true level)",
                 x=0.02, ha="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    vz.save(fig, ml.FIG / f"eval_confusion_{task}.png")


def macro_roc(y, proba):
    grid = np.linspace(0, 1, 201)
    if proba.shape[1] == 2:
        f, t, _ = roc_curve(y, proba[:, 1])
        return grid, np.interp(grid, f, t)
    return grid, np.mean([np.interp(grid, *roc_curve(y == c, proba[:, c])[:2]) for c in range(proba.shape[1])], 0)


def fig_roc(S, Pall):
    tasks = [t for t in ml.TASKS if t in Pall]
    fig, axes = plt.subplots(2, len(tasks), figsize=(7.2, 5.4), squeeze=False)
    for j, task in enumerate(tasks):
        q = S[S.task == task].set_index("model")
        for i, group in enumerate([BASELINES, CANDIDATES + ["vote"]]):
            ax = axes[i, j]
            if i == 1 and "y-corrected" in Pall[task]:      # reference line in the candidates panel
                t = Pall[task]["y-corrected"]["test"]
                ax.plot(*macro_roc(t.y, t.proba), color=COLOR["y-corrected"], lw=1.0, ls=":",
                        label=f"{SHORT['y-corrected']} ({q.loc['y-corrected', 'test_auc']:.1f})")
            for m in group:
                if m not in Pall[task]:
                    continue
                t = Pall[task][m]["test"]
                ax.plot(*macro_roc(t.y, t.proba), color=COLOR[m], lw=2.0 if m == "vote" else 1.4,
                        ls="--" if m == "y-leaky" else "-", label=f"{SHORT[m]} ({q.loc[m, 'test_auc']:.1f})")
            ax.plot([0, 1], [0, 1], color=vz.GRID, lw=0.8)
            ax.set_aspect("equal")
            ax.legend(loc="lower right", fontsize=5.6, handlelength=1.2, borderaxespad=0.2, labelspacing=0.25,
                      frameon=True, framealpha=0.92, facecolor=vz.SURFACE, edgecolor="none")
            if i == 0:
                ax.set_title(ml.TASKS[task]["title"] + ("" if len(ml.TASKS[task]["levels"]) == 2 else "\n(macro one-vs-rest)"),
                             loc="left", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(("baselines" if i == 0 else "candidates") + "\ntrue positive rate")
            if i == 1:
                ax.set_xlabel("false positive rate")
    fig.suptitle("ROC curves on test subjects (legend: AUC %)", x=0.02, ha="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    vz.save(fig, ml.FIG / "eval_roc.png")


def main():
    vz.use()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    Ss, PCs, Gs, Pall = [], [], [], {}
    for task in ml.TASKS:
        if not list(ml.PRED.glob(f"{task}__*.npz")):
            continue
        S, PC, G, agree, P = evaluate_task(task)
        Ss.append(S)
        PCs.append(PC)
        Gs.append(G)
        Pall[task] = P
        print(f"\n== {ml.TASKS[task]['title']}  ({S.subj_above_chance.iat[0].split('/')[1]} test subjects)")
        print("validation gate for the vote:")
        print(G.drop(columns="task").round(1).to_string(index=False))
        if agree is not None:
            print("prediction agreement between vote members on validation (%):")
            print(agree.round(0).to_string())
        cols = ["model", "dev_cv", "dev_cv_acc", "val_bacc", "test_acc", "test_bacc", "bacc_lo", "bacc_hi",
                "test_kappa", "test_auc", "subj_above_chance", "smooth20_bacc", "diff_vs_y-corrected",
                "diff_vs_y-corrected_lo", "diff_vs_y-corrected_hi", "diff_vs_eegnet", "diff_vs_eegnet_lo",
                "diff_vs_eegnet_hi", "best_by_val"]
        print(S[[c for c in cols if c in S]].round(1).to_string(index=False))
        fig_confusion(task, S, P)
    S, PC, G = pd.concat(Ss, ignore_index=True), pd.concat(PCs, ignore_index=True), pd.concat(Gs, ignore_index=True)
    S.round(2).to_csv(ml.RESULTS / "eval_summary.csv", index=False)
    PC.round(2).to_csv(ml.RESULTS / "eval_per_class.csv", index=False)
    G.round(2).to_csv(ml.RESULTS / "eval_vote_gate.csv", index=False)
    print("\n== per-class precision / recall on test (%)")
    print(PC.pivot_table(index=["task", "model"], columns="n", values=["precision", "recall"]).round(0).to_string())

    L = leakage_table()
    if len(L):
        L.round(2).to_csv(ml.RESULTS / "eval_leakage.csv", index=False)
        print("\n== Yedukondalu pipeline: development CV vs test, per family")
        print(L.drop(columns="params").round(1).to_string(index=False))
        fig_leakage(L)

    rq2 = []
    for task in Pall:
        k = len(ml.TASKS[task]["levels"])
        q = S[S.task == task]
        for _, r in q[q.model.isin(["y-corrected", "eegnet"]) | q.best_by_val].iterrows():
            rq2.append(dict(task=task, classes=k, model=r.model, chance=100 / k, test_bacc=r.test_bacc,
                            bacc_normalized=100 * (r.test_bacc - 100 / k) / (100 - 100 / k),
                            kappa=r.test_kappa, auc=r.test_auc, smooth20_bacc=r.smooth20_bacc))
    rq2 = pd.DataFrame(rq2)
    rq2.round(2).to_csv(ml.RESULTS / "eval_rq2.csv", index=False)
    print("\n== RQ2: chance-corrected performance, presence of load (binary) vs levels (ternary)")
    print(rq2.round(1).to_string(index=False))

    fig_summary(S)
    fig_roc(S, Pall)
    print("\nfigures: " + ", ".join(sorted(p.name for p in ml.FIG.glob("eval_*.png"))))


if __name__ == "__main__":
    main()
