"""Stage 6: baseline workload classification (Table 3) -> results/classification.csv.

Binary: nw vs (lw, mw, hw). Ternary: lw vs mw vs hw, i.e. 2- vs 3- vs 4-back
inside the multitask. Clean = subjects 01-30, noisy = 31-40, as in the paper.

Two validation schemes:
- "segment": stratified 10-fold over 2 s windows pooled across subjects, which
  is what a table without subject-wise splitting usually means;
- "subject": leave-one-subject-out, so no subject is in both training and test.

Metrics follow Table 3: accuracy and support-weighted precision, recall and
F1 (weighted recall equals accuracy, as in the paper's table). The majority
baseline is reported next to them because the binary task is 3:1.

Control: the noisy group has a third of the clean group's subjects. Random
10-subject subsets of the clean group show what the smaller sample alone does.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backrepro import data, features, paper  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
SEED = 0
MODELS = {
    "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000,
                                                                 early_stopping=True, random_state=SEED)),
    "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
                                     colsample_bytree=0.8, random_state=SEED, n_jobs=1),
}


def task_data(X, cond, subj, task, subjects):
    m = np.isin(subj, subjects) & ((cond != "nw") if task == "ternary" else True)
    y = (cond[m] != "nw").astype(int) if task == "binary" else np.array([["lw", "mw", "hw"].index(c) for c in cond[m]])
    return X[m], y, subj[m]


def evaluate(X, y, groups, model, scheme):
    cv = (StratifiedKFold(10, shuffle=True, random_state=SEED) if scheme == "segment" else LeaveOneGroupOut())
    pred = cross_val_predict(MODELS[model](), X, y, cv=cv, groups=groups, n_jobs=-1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    return dict(accuracy=100 * accuracy_score(y, pred), precision=100 * p, recall=100 * r, f1=100 * f,
                balanced_accuracy=100 * balanced_accuracy_score(y, pred),
                majority=100 * np.bincount(y).max() / len(y), n_segments=len(y))


def main():
    F, cond, subj = features.build(2.0)
    print(f"{len(cond)} windows of 2 s; {F['bandpower'].shape[1]} bandpower, {F['extended'].shape[1]} extended features")
    rows = []
    for fs_name, X in F.items():
        for task in ["binary", "ternary"]:
            for grp, S in [("clean", data.CLEAN), ("noisy", data.NOISY)]:
                Xt, yt, gt = task_data(X, cond, subj, task, S)
                for model in MODELS:
                    for scheme in ["segment", "subject"]:
                        r = dict(features=fs_name, task=task, group=grp, model=model, cv=scheme,
                                 **evaluate(Xt, yt, gt, model, scheme))
                        rows.append(r)
                        ref = paper.TABLE3[(task, grp, model)][0]
                        print(f"{fs_name:9s} {task:7s} {grp:5s} {model:7s} {scheme:7s} acc {r['accuracy']:5.1f} "
                              f"(paper {ref:5.2f}, majority {r['majority']:4.1f}, balanced {r['balanced_accuracy']:4.1f}) "
                              f"P {r['precision']:4.1f} R {r['recall']:4.1f} F1 {r['f1']:4.1f}", flush=True)

    # Sample-size control: 10-subject subsets of the clean group, segment-level CV, bandpower
    rng = np.random.default_rng(SEED)
    X = F["bandpower"]
    for task in ["binary", "ternary"]:
        for model in MODELS:
            accs = []
            for _ in range(10):
                S = rng.choice(data.CLEAN, 10, replace=False)
                accs.append(evaluate(*task_data(X, cond, subj, task, S), model, "segment")["accuracy"])
            rows.append(dict(features="bandpower", task=task, group="clean, 10 random subjects", model=model,
                             cv="segment", accuracy=float(np.mean(accs)), accuracy_sd=float(np.std(accs))))
            print(f"control   {task:7s} 10 clean subjects {model:7s} segment acc {np.mean(accs):5.1f} ± {np.std(accs):.1f} "
                  f"over 10 draws", flush=True)

    pd.DataFrame(rows).to_csv(data.RESULTS / "classification.csv", index=False, float_format="%.2f")


if __name__ == "__main__":
    main()
