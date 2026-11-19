import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nbackml as ml  # noqa: E402


def test_labels_follow_level_order():
    df = pd.DataFrame({"n": [0, 2, 3, 2, 0]})
    assert ml.labels(df, "shin3").tolist() == [0, 1, 2, 1, 0]
    assert ml.labels(pd.DataFrame({"n": [0, 2]}), "bin").tolist() == [0, 1]


def test_subject_folds_never_share_a_subject():
    g = np.repeat(np.arange(30), 20)
    y = np.tile([0, 1], 300)
    for tr, te in ml.subject_folds(y, g, k=5):
        assert not set(g[tr]) & set(g[te])


def test_smooth_is_causal_and_stays_in_block():
    df = pd.DataFrame({"subject": ["a"] * 6, "run": ["r"] * 6, "block_in_run": [1, 1, 1, 2, 2, 2],
                       "window": [0, 1, 2, 0, 1, 2]})
    p = np.array([[1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1]], float)
    s = ml.smooth(p, df, k=2)
    assert np.allclose(s[:, 1], [0, 0.5, 1, 0, 0, 0.5])          # block 2 does not see block 1
    shuffled = df.sample(frac=1, random_state=1)
    s2 = ml.smooth(p[shuffled.index], shuffled.reset_index(drop=True), k=2)
    assert np.allclose(s2, s[shuffled.index])                    # row order does not matter


def test_metrics_perfect_and_chance():
    y = np.array([0, 1, 2, 0, 1, 2])
    m = ml.metrics(y, np.eye(3)[y])
    assert m["acc"] == m["bacc"] == m["kappa"] == m["auc"] == 100
    assert abs(ml.metrics(y, np.full((6, 3), 1 / 3))["auc"] - 50) < 1e-9


def test_sumzero_basis_keeps_average_referenced_signal():
    B = ml._sumzero_basis(26)
    assert np.allclose(B.T @ B, np.eye(25)) and np.allclose(B.sum(0), 0)
    x = np.random.default_rng(0).normal(size=(26, 100))
    x -= x.mean(0)                                               # common average reference
    assert np.allclose(B @ (B.T @ x), x)                         # projection is lossless


def test_zscore_subject_uses_each_subject_alone():
    X = np.array([[1.0], [3.0], [100.0], [300.0]])
    z = ml.zscore_subject(X, np.array(["a", "a", "b", "b"]))
    assert np.allclose(z.ravel(), [-1, 1, -1, 1])
