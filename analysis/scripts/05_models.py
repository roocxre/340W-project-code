"""Stage 5: feature-based classifiers, and the logistic-regression baseline.

    caffeinate -i ../synthesis/.venv/bin/python scripts/05_models.py [--task bin] [--models ...]

All candidates normalize within subject without labels: features are z-scored over each
subject's windows, and covariance matrices are re-centred to each subject's Riemannian mean.
Hyperparameters are chosen on subject-wise 5-fold CV inside the train split (balanced
accuracy). The chosen configuration is refitted on all of train and scored on val and test.

    lr        logistic regression on band power (the synthesis baseline, 260 features)
    svm       RBF support vector machine on band power
    mlp       two-layer perceptron on band power
    rf        random forest on band power + R-LMD (1,040 features)
    xgb       XGBoost on band power + R-LMD
    riemann   filter-bank tangent space (theta, alpha, beta covariances, subject re-centred)
              + logistic regression (975 features)

Outputs: cache/models/<task>_<model>.json, results/predictions/<task>__<model>.npz.
"""
import argparse
import itertools
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nbackml as ml  # noqa: E402


def grid(**axes):
    return [dict(zip(axes, v)) for v in itertools.product(*axes.values())]


MODELS = {
    "lr": dict(features="bp", grid=grid(C=[0.005, 0.02, 0.05, 0.2, 1.0]),
               make=lambda p: make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=5000))),
    "svm": dict(features="bp", grid=grid(C=[0.3, 1.0, 3.0], gamma=["scale", 0.001]),
                make=lambda p: make_pipeline(StandardScaler(), SVC(C=p["C"], gamma=p["gamma"], cache_size=1000))),
    "mlp": dict(features="bp", grid=grid(alpha=[1e-3, 1e-1, 1.0], hidden=[(256, 128)]),
                make=lambda p: make_pipeline(StandardScaler(), MLPClassifier(
                    hidden_layer_sizes=p["hidden"], alpha=p["alpha"], max_iter=60, batch_size=256,
                    learning_rate_init=1e-3, random_state=ml.SEED))),
    "rf": dict(features="bp+rlmd", grid=grid(max_features=["sqrt", 0.1], min_samples_leaf=[1, 10]),
               make=lambda p: RandomForestClassifier(
                   n_estimators=500, max_features=p["max_features"], min_samples_leaf=p["min_samples_leaf"],
                   n_jobs=2, random_state=ml.SEED)),
    "xgb": dict(features="bp+rlmd", grid=grid(max_depth=[3, 6], learning_rate=[0.05, 0.1]),
                make=lambda p: XGBClassifier(
                    n_estimators=400, max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                    subsample=0.8, colsample_bytree=0.5, min_child_weight=5, reg_lambda=1.0,
                    tree_method="hist", n_jobs=2, random_state=ml.SEED, verbosity=0)),
    "riemann": dict(features="riemann", grid=grid(C=[0.005, 0.02, 0.1, 0.5]),
                    make=lambda p: make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=5000))),
}


def features(df, kind):
    g = df.subject.to_numpy()
    if kind == "bp":
        return ml.zscore_subject(ml.bandpower(df), g)
    if kind == "bp+rlmd":
        return ml.zscore_subject(np.hstack([ml.bandpower(df), ml.rlmd(df)]), g)
    if kind == "riemann":
        return ml.zscore_subject(ml.tangent_recentred(df), g)
    raise ValueError(kind)


def proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    d = model.decision_function(X)
    d = np.column_stack([-d, d]) / 2 if d.ndim == 1 else d
    e = np.exp(d - d.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def fit_predict(est, X, y, tr, te):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return te, proba(clone(est).fit(X[tr], y[tr]), X[te])


def run(task, name, jobs):
    out = ml.CACHE / "models" / f"{task}_{name}.json"
    if out.exists():
        return json.loads(out.read_text())
    spec = MODELS[name]
    d = {s: ml.frame(s, task) for s in ("train", "val", "test")}
    y = {s: ml.labels(d[s], task) for s in d}
    t0 = time.time()
    X = {s: features(d[s], spec["features"]) for s in d}
    t_feat = time.time() - t0
    folds = ml.subject_folds(y["train"], d["train"].subject.to_numpy(), k=5)
    k = len(ml.TASKS[task]["levels"])
    jobs_list = [(i, tr, te) for i in range(len(spec["grid"])) for tr, te in folds]
    res = Parallel(n_jobs=jobs)(delayed(fit_predict)(spec["make"](spec["grid"][i]), X["train"], y["train"], tr, te)
                                for i, tr, te in jobs_list)
    oof = np.zeros((len(spec["grid"]), len(y["train"]), k))
    for (i, _, _), (te, p) in zip(jobs_list, res):
        oof[i, te] = p
    cv_bacc = [100 * balanced_accuracy_score(y["train"], o.argmax(1)) for o in oof]
    best = int(np.argmax(cv_bacc))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = spec["make"](spec["grid"][best]).fit(X["train"], y["train"])
    P = {s: proba(model, X[s]) for s in ("val", "test")}
    r = dict(model=name, task=task, features=spec["features"], n_features=X["train"].shape[1],
             params=spec["grid"][best], cv_bacc_grid=[round(v, 2) for v in cv_bacc],
             cv=ml.metrics(y["train"], oof[best]), val=ml.metrics(y["val"], P["val"]),
             test=ml.metrics(y["test"], P["test"]), seconds=round(time.time() - t0), feature_seconds=round(t_feat))
    ml.save_predictions(task, name, cv_proba=oof[best], cv_y=y["train"], cv_row=d["train"].row_id.to_numpy(),
                        **{f"{s}_proba": P[s] for s in P}, **{f"{s}_y": y[s] for s in P},
                        **{f"{s}_row": d[s].row_id.to_numpy() for s in P})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1, default=str))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", nargs="*", default=list(ml.TASKS))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    for task in a.task:
        for name in a.models:
            r = run(task, name, a.jobs)
            print(f"{task:6s} {name:8s} CV bacc {r['cv']['bacc']:.1f} | val acc {r['val']['acc']:.1f} | "
                  f"test acc {r['test']['acc']:.1f} bacc {r['test']['bacc']:.1f} kappa {r['test']['kappa']:.1f} "
                  f"auc {r['test']['auc']:.1f} | {r['params']} ({r['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
