"""Stage 4: Yedukondalu et al.'s pipeline as the baseline, with and without the leak.

    caffeinate -i ../synthesis/.venv/bin/python scripts/04_baselines.py --task bin --protocol leaky

Pipeline (eeg-load's reproduction, imported unchanged from eeg-load/loadrepro):
    R-LMD features (stage 2, 780 per window), z-scored per column over the training pool
    -> BAO feature selection (30 agents, 100 iterations, KNN k = 5 wrapper, 5-fold error)
    -> six "optimizable" classifiers (tree, discriminant, naive Bayes, KNN, SVM, ensemble),
       each tuned for 30 Optuna trials on the 10-fold CV error.

Protocols differ only in what a fold is, both inside BAO's fitness and in the tuning CV:
    leaky      windows at random (the paper's segment-level 10-fold): neighbouring windows
               of the same block and subject fall on both sides
    corrected  whole subjects (stratified group k-fold), so every score is on unseen people

Development uses the train split only (38 subjects). Each tuned model is then refitted on all
of train and scored on val and test, which neither protocol ever saw. The leaky protocol's
tuned models are also re-scored with subject-wise CV, as eeg-load did for MAT and STEW.

--norm subject replaces the pooled z-score with a within-subject z-score (the synthesis
normalization), to separate the effect of normalization from that of the model.

Outputs: cache/baselines/<task>_<protocol>_<norm>/{bao.json, <family>.json},
results/predictions/<task>__yedukondalu-<protocol>-<norm>-<family>.npz.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "eeg-load"))
import nbackml as ml  # noqa: E402
from loadrepro import bao, classifiers  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS, BAO_ITER = classifiers.N_TRIALS, 100


def scores(model, X):
    """Class probabilities, or a softmax of decision values when the model has none."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    d = model.decision_function(X)
    if d.ndim == 1:
        d = np.column_stack([-d, d]) / 2
    e = np.exp(d - d.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def fit_fold(family, p, X, y, tr, te, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = classifiers.build(family, p, seed).fit(X[tr], y[tr])
        return scores(model, X[te])


def cross_val(family, p, X, y, folds, n_jobs):
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=n_jobs)(delayed(fit_fold)(family, p, X, y, tr, te, ml.SEED) for tr, te in folds)
    proba = np.zeros((len(y), len(np.unique(y))))
    for (_, te), pr in zip(folds, out):
        proba[te] = pr
    return np.nan_to_num(proba, nan=1.0 / proba.shape[1])


def tune(family, X, y, folds, n_jobs):
    best = {"err": np.inf}
    curve = []

    def objective(trial):
        p = classifiers.suggest(family, trial, len(folds[0][0]), X.shape[1])
        try:
            proba = cross_val(family, p, X, y, folds, n_jobs)
            err = float(np.mean(proba.argmax(1) != y))
        except Exception:  # e.g. a singular covariance in QDA
            err, proba = 1.0, None
        curve.append(err)
        if err < best["err"]:
            best.update(err=err, params=p, proba=proba)
        return err

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=ml.SEED))
    study.optimize(objective, n_trials=N_TRIALS)
    best["curve"] = np.minimum.accumulate(curve).tolist()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(ml.TASKS), required=True)
    ap.add_argument("--protocol", choices=["leaky", "corrected"], required=True)
    ap.add_argument("--norm", choices=["pooled", "subject"], default="pooled")
    ap.add_argument("--families", nargs="*", default=classifiers.FAMILIES)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    tag = f"{a.task}_{a.protocol}_{a.norm}"
    out = ml.CACHE / "baselines" / tag
    out.mkdir(parents=True, exist_ok=True)

    d = {s: ml.frame(s, a.task) for s in ("train", "val", "test")}
    y = {s: ml.labels(d[s], a.task) for s in d}
    g = {s: d[s].subject.to_numpy() for s in d}
    R = {s: ml.rlmd(d[s]) for s in d}
    if a.norm == "pooled":
        mu, sd = R["train"].mean(0), R["train"].std(0)
        X = {s: (R[s] - mu) / np.where(sd > 0, sd, 1) for s in d}
    else:
        X = {s: ml.zscore_subject(R[s], g[s]) for s in d}
    Xtr, ytr, gtr = X["train"].astype(np.float32), y["train"], g["train"]
    print(f"{tag}: train {len(ytr)} windows / {len(set(gtr))} subjects, val {len(y['val'])}, "
          f"test {len(y['test'])}, {Xtr.shape[1]} features, classes {np.bincount(ytr).tolist()}", flush=True)

    # BAO, with the protocol's folds inside its KNN fitness
    bao_file = out / "bao.json"
    if bao_file.exists():
        sel = json.loads(bao_file.read_text())
    else:
        t0 = time.time()
        fit = bao.Fitness(Xtr, ytr, k=5, folds=5, seed=ml.SEED)
        if a.protocol == "corrected":
            fit.splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=ml.SEED).split(Xtr, ytr, gtr))
        else:
            fit.splits = list(StratifiedKFold(5, shuffle=True, random_state=ml.SEED).split(Xtr, ytr))
        mask, curve = bao.bao(fit, Xtr.shape[1], n_iter=BAO_ITER, seed=ml.SEED)
        sel = dict(mask=mask.astype(int).tolist(), curve=curve.tolist(), seconds=round(time.time() - t0))
        bao_file.write_text(json.dumps(sel))
    mask = np.array(sel["mask"], bool)
    print(f"  BAO kept {mask.sum()}/{len(mask)}, fitness {sel['curve'][0]:.4f} -> {sel['curve'][-1]:.4f} "
          f"({sel['seconds']} s)", flush=True)

    Xs = {s: X[s][:, mask] for s in X}
    folds = ml.window_folds(ytr) if a.protocol == "leaky" else ml.subject_folds(ytr, gtr)
    for family in a.families:
        f_json = out / f"{family}.json"
        if f_json.exists():
            r = json.loads(f_json.read_text())
        else:
            t0 = time.time()
            best = tune(family, Xs["train"], ytr, folds, a.jobs)
            r = dict(family=family, params=best["params"], curve=best["curve"],
                     cv=ml.metrics(ytr, best["proba"]))
            if a.protocol == "leaky":   # eeg-load's correction: same model, subject-wise folds
                p_sub = cross_val(family, best["params"], Xs["train"], ytr, ml.subject_folds(ytr, gtr), a.jobs)
                r["cv_subject_rescore"] = ml.metrics(ytr, p_sub)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = classifiers.build(family, best["params"], ml.SEED).fit(Xs["train"], ytr)
            P = {s: scores(model, Xs[s]) for s in ("val", "test")}
            r.update(val=ml.metrics(y["val"], P["val"]), test=ml.metrics(y["test"], P["test"]),
                     seconds=round(time.time() - t0))
            ml.save_predictions(a.task, f"yedukondalu-{a.protocol}-{a.norm}-{family}",
                                cv_proba=best["proba"], cv_y=ytr, cv_row=d["train"].row_id.to_numpy(),
                                **{f"{s}_proba": P[s] for s in P}, **{f"{s}_y": y[s] for s in P},
                                **{f"{s}_row": d[s].row_id.to_numpy() for s in P})
            f_json.write_text(json.dumps(r, indent=1))
        extra = f" | subject re-score {r['cv_subject_rescore']['acc']:.1f}" if "cv_subject_rescore" in r else ""
        print(f"  {classifiers.LABELS[family]:13s} CV acc {r['cv']['acc']:.1f}{extra} | val {r['val']['acc']:.1f} "
              f"| test acc {r['test']['acc']:.1f} bacc {r['test']['bacc']:.1f} auc {r['test']['auc']:.1f} "
              f"| {r['params']} ({r.get('seconds', 0)} s)", flush=True)


if __name__ == "__main__":
    main()
