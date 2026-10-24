import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backrepro import data, digitize, paper  # noqa: E402
from backrepro.released import psd_analysis  # noqa: E402

needs_data = pytest.mark.skipif(not data.DATA.exists(), reason="dataset not in data/iNCog-EEG")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_released_band_analysis_on_a_sine():
    fs = 200.0
    t = np.arange(0, 60, 1 / fs)
    f, p = psd_analysis.compute_psd_welch(np.sin(2 * np.pi * 10.0 * t), fs)
    m = (f >= 1) & (f <= 40)
    bands = psd_analysis.analyze_bands(f[m], p[m])
    assert abs(bands["Alpha"]["dominant_freq"] - 10.0) < 0.1
    assert bands["Alpha"]["relative_power"] > 95


@needs_data
def test_files_and_channels():
    files = list(data.iter_files())
    assert len(files) == 160 and all(data.path(s, c).exists() for s, c in files)
    raw = data.load_raw(1, "nw", "all")
    assert raw.info["sfreq"] == 200 and len(raw.ch_names) == 32
    raw = data.load_raw(1, "nw")
    assert raw.ch_names == data.CHANNELS
    assert np.isfinite(np.array([ch["loc"][:3] for ch in raw.info["chs"]])).all()


@needs_data
def test_device_records_follow_acquisition_order():
    rec = [data.device_record(s, c) for s, c in data.iter_files()]
    assert np.all(np.diff(rec) > 0)


def test_fig7_digitisation_is_a_correlation_matrix():
    mats, err = digitize.fig7_matrices()
    for name in ["clean", "noisy"]:
        m = mats[name]
        assert np.allclose(np.diag(m), 1, atol=0.03)
        assert np.abs(m - m.T).max() < 0.03
        assert err[name] < 0.02


@needs_data
def test_fig5_is_sub02_with_compute_psd_mne():
    psd = load_script("02_psd")
    assert psd.matches_fig5(psd.fig5_stats(2, "mne"))
    assert not psd.matches_fig5(psd.fig5_stats(2, "welch"))
    st = psd.fig5_stats(2)
    for c, (peak, ratio) in paper.FIG5.items():
        assert round(st[c]["peak"], 2) == peak and round(st[c]["ratio"], 2) == ratio


def test_task_labels():
    clf = load_script("06_classification")
    cond = np.array(["nw", "lw", "mw", "hw"] * 3)
    subj = np.repeat([1, 2, 31], 4)
    X = np.arange(len(cond))[:, None]
    _, y, _ = clf.task_data(X, cond, subj, "ternary", [1, 2])
    assert list(y) == [0, 1, 2, 0, 1, 2]
    _, y, g = clf.task_data(X, cond, subj, "binary", [31])
    assert list(y) == [0, 1, 1, 1] and set(g) == {31}
