"""Stage 4: tuning-parameter sweep on MAT (Fig. 9).

Ensemble accuracy against the R-LMD decomposition level (L = 3, 5, 7, 9 at
100 BAO iterations) and the number of BAO iterations (25, 50, 75, 100 at
L = 5), for all leads together and for the F3 lead alone.
    -> results/tuning_mat.csv
"""
import csv
import sys
from pathlib import Path

from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loadrepro import data, pipeline  # noqa: E402

LEVELS = (3, 5, 7, 9)
ITERS = (25, 50, 75, 100)
LEAD = "F3"


def run(scope, level, n_iter):
    leads = None if scope == "overall" else [LEAD]
    X, y, subj, _ = pipeline.load_features("mat", level=level, leads=leads)
    mask, _ = pipeline.select(X, y, n_iter=n_iter)
    res = pipeline.tune(X[:, mask], y, subj, "ensemble")
    return dict(scope=scope, level=level, bao_iterations=n_iter, n_selected=int(mask.sum()),
                AC=f"{res['AC']:.2f}", SE=f"{res['SE']:.2f}", SP=f"{res['SP']:.2f}",
                subject_cv_AC=f"{res['subject_cv']['AC']:.2f}")


def main():
    configs = sorted({(L, 100) for L in LEVELS} | {(5, i) for i in ITERS})
    jobs = [(s, L, i) for s in ("overall", "leadwise") for L, i in configs]
    rows = Parallel(n_jobs=-1)(delayed(run)(*j) for j in jobs)
    with open(data.RESULTS_DIR / "tuning_mat.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
