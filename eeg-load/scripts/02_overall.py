"""Stage 2: overall analysis (Tables 1 and 2, Figs 3-5).

all leads x 5 modes x 6 features -> BAO feature selection
    -> six optimized classifiers, 10-fold CV
    -> results/overall_<dataset>.csv, cache/overall_<dataset>.npz

Each classifier's best hyperparameters are also re-scored with subject-wise
10-fold CV (no subject in both train and test), which the paper does not do.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loadrepro import classifiers, data, pipeline  # noqa: E402


def main(dataset):
    X, y, subj, names = pipeline.load_features(dataset)
    t0 = time.time()
    mask, curve = pipeline.select(X, y)
    print(f"{dataset}: BAO kept {mask.sum()}/{len(mask)} features, "
          f"fitness {curve[0]:.4f} -> {curve[-1]:.4f} ({time.time() - t0:.0f} s)", flush=True)
    Xs = X[:, mask]
    rows, params, arrays = [], {}, dict(mask=mask, curve=curve, y=y)
    for family in classifiers.FAMILIES:
        t0 = time.time()
        res = pipeline.tune(Xs, y, subj, family, n_jobs=-1)
        rows.append(dict(classifier=classifiers.LABELS[family],
                         AC=res["AC"], SE=res["SE"], SP=res["SP"],
                         fold_sd=100 * res["fold_acc"].std(ddof=1),
                         subject_cv_AC=res["subject_cv"]["AC"],
                         subject_cv_SE=res["subject_cv"]["SE"],
                         subject_cv_SP=res["subject_cv"]["SP"]))
        params[family] = res["params"]
        for k in ("pred", "score", "fold_acc", "min_error_curve"):
            arrays[f"{family}_{k}"] = res[k]
        r = rows[-1]
        print(f"  {r['classifier']:13s} AC {r['AC']:.1f} SE {r['SE']:.1f} SP {r['SP']:.1f} | "
              f"subject-wise AC {r['subject_cv_AC']:.1f} | {res['params']} "
              f"({time.time() - t0:.0f} s)", flush=True)
    with open(data.RESULTS_DIR / f"overall_{dataset}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()}
                    for r in rows)
    with open(data.RESULTS_DIR / f"overall_{dataset}_selection.json", "w") as f:
        json.dump(dict(selected=[n for n, m in zip(names, mask) if m], params=params), f, indent=1)
    np.savez(data.CACHE_DIR / f"overall_{dataset}.npz", **arrays)


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["mat", "stew"]):
        main(ds)
