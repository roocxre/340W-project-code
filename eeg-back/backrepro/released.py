"""The authors' released functions, loaded unmodified from reference/EEGCWLD.

The repository is a function library with no driver script, so the scripts in
scripts/ call these functions directly. The only intervention is restoring
``np.trapz``, which the released code calls and NumPy 2.4 removed; it is the
same function as ``np.trapezoid``.
"""

import importlib.util
from pathlib import Path

import numpy as np

REF = Path(__file__).resolve().parents[1] / "reference" / "EEGCWLD"

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid


def _load(name):
    spec = importlib.util.spec_from_file_location(f"eegcwld_{name}", REF / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


psd_analysis = _load("psd_analysis")
beam_topography = _load("beam_topography")
correlation_analysis = _load("correlation_analysis")
