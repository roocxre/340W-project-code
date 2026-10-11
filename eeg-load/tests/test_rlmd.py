import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loadrepro import features, rlmd  # noqa: E402


def demo_signal():
    # lmd_demo.m
    fs, n = 10000, 30000
    t = np.arange(1, n + 1) / fs
    x1 = (2 + np.cos(2 * np.pi * 0.5 * t)) * np.cos(2 * np.pi * 5 * t + 15 * t ** 2)
    x2 = np.cos(2 * np.pi * 2 * t)
    return x1, x2


def test_smooth_matches_matlab():
    # smooth([1 2 4 8 16 32 64], 5)
    y = np.array([1, 2, 4, 8, 16, 32, 64], float)
    expect = [1, 7 / 3, 31 / 5, 62 / 5, 124 / 5, 112 / 3, 64]
    assert np.allclose(rlmd.smooth(y, 5), expect)
    assert np.allclose(rlmd.smooth(y, 4), [1, 7 / 3, 14 / 3, 28 / 3, 56 / 3, 112 / 3, 64])
    assert np.allclose(rlmd.smooth(y, 1), y)


def test_extr():
    x = np.array([0, 1, 0, -1, 0, 2, 1], float)
    indmin, indmax = rlmd.extr(x)
    assert list(indmin) == [4] and list(indmax) == [2, 6]


def test_reconstruction():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(2000))
    pfs, ams, fms = rlmd.rlmd(x, max_pfs=4)
    assert np.allclose(pfs.sum(axis=0), x)
    assert len(ams) == len(fms) == len(pfs) - 1
    assert np.allclose(rlmd.modes(x, 5).sum(axis=0), x)


def test_demo_separation():
    x1, x2 = demo_signal()
    pfs, _, _ = rlmd.rlmd(x1 + x2)
    core = slice(3000, -3000)
    assert np.corrcoef(pfs[0][core], x1[core])[0, 1] > 0.98
    assert np.corrcoef(pfs[1][core], x2[core])[0, 1] > 0.95


def test_modes_zero_padded():
    x = np.sin(np.linspace(0, 20 * np.pi, 1000))
    m = rlmd.modes(x, 5)
    assert m.shape == (5, 1000)
    assert np.allclose(m.sum(axis=0), x)


def test_entropies():
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(1000)
    sine = np.sin(np.linspace(0, 40 * np.pi, 1000))
    assert features.fuzzy_entropy(noise) > features.fuzzy_entropy(sine)
    # Renyi quadratic entropy of N(0, s^2) is log(2 s sqrt(pi))
    big = rng.standard_normal(4000)
    assert abs(features.renyi_entropy(big) - np.log(2 * np.sqrt(np.pi))) < 0.1
    f = features.mode_features(big, 500)
    assert abs(f[5] - 0.5 * np.log(2 * np.pi * np.e)) < 0.05
    assert np.all(features.mode_features(np.zeros(100), 500) == 0)


def test_signal_features_levels_share_pfs():
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.standard_normal(1000))
    multi = features.signal_features(x, 500, levels=(3, 5))
    for L in (3, 5):
        m = rlmd.modes(x, L)
        direct = np.array([features.mode_features(r, 500) for r in m])
        assert np.allclose(multi[L], direct)
