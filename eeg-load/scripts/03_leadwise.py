"""Stage 3: lead-wise analysis (Tables 3 and 4).

per lead: 5 modes x 6 features -> BAO -> optimized ensemble, 10-fold CV
    -> results/leadwise_<dataset>.csv, cache/leadwise_<dataset>.npz
"""
import csv
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loadrepro import data, pipeline  # noqa: E402


def run_lead(dataset, lead):
    X, y, subj, names = pipeline.load_features(dataset, leads=[lead])
    mask, curve = pipeline.select(X, y)
    res = pipeline.tune(X[:, mask], y, subj, "ensemble")
    row = dict(lead=lead, AC=res["AC"], SE=res["SE"], SP=res["SP"],
               subject_cv_AC=res["subject_cv"]["AC"], n_selected=int(mask.sum()),
               selected=" ".join(n.split("_", 1)[1] for n, m in zip(names, mask) if m),
               method=res["params"]["method"])
    return row, mask, curve


def main(dataset):
    clab = data.DATASETS[dataset]["clab"]
    out = Parallel(n_jobs=-1)(delayed(run_lead)(dataset, c) for c in clab)
    rows = [o[0] for o in out]
    with open(data.RESULTS_DIR / f"leadwise_{dataset}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()}
                    for r in rows)
    np.savez(data.CACHE_DIR / f"leadwise_{dataset}.npz",
             masks=np.array([o[1] for o in out]), curves=np.array([o[2] for o in out]))
    for r in sorted(rows, key=lambda r: -r["AC"]):
        print(f"{dataset} {r['lead']:4s} AC {r['AC']:.1f} SE {r['SE']:.1f} SP {r['SP']:.1f} | "
              f"subject-wise AC {r['subject_cv_AC']:.1f} | {r['n_selected']} features")


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["mat", "stew"]):
        main(ds)
