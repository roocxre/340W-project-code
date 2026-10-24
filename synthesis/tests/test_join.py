import sys
from pathlib import Path

import mne
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nbackjoin import features, loaders, sources


def test_channel_set():
    assert len(sources.CHANNELS) == len(set(sources.CHANNELS)) == 26
    shin = set(loaders.shin_data.EEG_CLAB)
    assert set(sources.CHANNELS) <= shin
    assert shin - set(sources.CHANNELS) == {"AFF5h", "AFF6h"}
    montage = mne.channels.make_standard_montage("standard_1005").ch_names
    assert set(loaders.shin_data.EEG_CLAB) <= set(montage)


def test_spectral_band_assignment():
    fs = sources.FS
    t = np.arange(int(features.WIN_S * fs)) / fs
    x = np.stack([np.sin(2 * np.pi * f * t) for f in (2, 6, 10, 20, 35)])[None]   # one channel per band
    log_abs, rel = features.spectral(x)
    assert np.array_equal(rel[0].argmax(1), np.arange(5))
    assert np.allclose(rel.sum(-1), 1, atol=0.02)
    # a unit sine has power 0.5
    assert np.allclose(10 ** log_abs[0].max(1), 0.5, rtol=0.05)


def _cog_raw(soa=2.008, n=2):
    on, de = [], []
    t = 5.0
    for blk in range(3):
        for k in range(48):
            de.append(f"6{n}22" if k % 3 == 0 else f"6{n}21")
            on.append(t)
            if k % 3 == 0:
                on.append(t + 0.5); de.append(f"6{n}32")
            t += soa
        t += 9.0
    raw = mne.io.RawArray(np.zeros((1, int((t + 5) * 500))), mne.create_info(["Fz"], 500.0, "eeg"), verbose=False)
    raw.set_annotations(mne.Annotations(on, 0, de))
    return raw


@pytest.mark.parametrize("soa", [2.008, 2.516])
def test_cog_blocks(soa):
    blocks = loaders._cog_blocks(_cog_raw(soa), 2)
    assert [len(b.onsets) for b in blocks] == [48, 48, 48]
    for b in blocks:
        assert b.n == 2 and b.is_target.sum() == 16
        assert abs(b.soa - soa) < 1e-6
        assert b.accuracy == 1.0 and abs(b.mean_rt - 0.5) < 1e-6


def test_windows_stay_inside_blocks():
    raw = _cog_raw(2.008)
    run = loaders.Run("cog", "cog-00", 1, raw, loaders._cog_blocks(raw, 2))
    clean = raw.copy().resample(sources.FS)
    ws = list(features.windows(clean, run))
    assert len(ws) == 3 * int(48 * 2.008 // features.WIN_S)
    for x, m in ws:
        assert x.shape == (1, int(features.WIN_S * sources.FS))
        assert m["t_in_block_s"] + features.WIN_S <= 48 * 2.008


def test_bad_channel_rule():
    from nbackjoin.preprocess import _bad_channels
    rng = np.random.default_rng(0)
    names = ["Fp1", "Fp2", "F1", "F2", "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2"]
    # several sources with channel-specific weights, so structure survives the average reference
    src = rng.standard_normal((4, 20000))
    x = (rng.uniform(-1, 1, (len(names), 4)) @ src + 0.3 * rng.standard_normal((len(names), 20000))) * 10e-6
    x[0] *= 3.5                                   # ocular-scale frontal amplitude: keep
    x[5] = 1e-9 * rng.standard_normal(20000)       # near-flat: bad
    x[9] = 10e-6 * rng.standard_normal(20000)      # uncorrelated with everything: bad
    raw = mne.io.RawArray(x, mne.create_info(names, 200.0, "eeg"), verbose=False)
    assert _bad_channels([raw]) == ["Cz", "P4"]
