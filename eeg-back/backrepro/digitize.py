"""Reading values back out of the paper's raster figures.

Fig. 7 (correlation heatmaps) and Fig. 8 (BEAM maps) are embedded in the PDF
as bitmaps. Mapping their colours back through the colormap gives numbers
that can be compared with every recording, which is how the recordings behind
Fig. 7 were identified.
"""

import matplotlib
import numpy as np
import pymupdf
from mne.channels.layout import _find_topomap_coords
from mne.viz.topomap import _check_sphere
from scipy import ndimage

from .data import ROOT

PAPER = ROOT / "docs" / "s41597-026-07607-3_reference.pdf"
FIG7_XREF = 136   # page 9, 1900 x 569 px
FIG8_XREF = 137   # page 9, 1691 x 760 px

FIG8_PANELS = [(g, b, w) for g in ("clean", "noisy") for b in ("Theta", "Alpha") for w in ("W", "NW")]


def figure_image(xref):
    """An embedded figure as an RGB float array in [0, 1]."""
    doc = pymupdf.open(PAPER)
    pix = pymupdf.Pixmap(doc, xref)
    if pix.n - pix.alpha > 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return img[..., :3] / 255.0


def _invert(rgb, cmap, vmin, vmax, n=2001):
    """Nearest-colour inversion of a colormap; also returns the colour error."""
    lut = matplotlib.colormaps[cmap](np.linspace(0, 1, n))[:, :3]
    d = ((rgb[..., None, :] - lut) ** 2).sum(-1)
    return np.linspace(vmin, vmax, n)[d.argmin(-1)], np.sqrt(d.min(-1))


def fig7_matrices():
    """The clean and noisy 16x16 correlation matrices of Fig. 7.

    Both panels are seaborn heatmaps with center=0 and a diagonal of 1, so
    'coolwarm' spans [-1, 1]. Panels are located by their dark-red diagonal.
    Returns {"clean": M, "noisy": M} and the mean colour error per panel.
    """
    img = figure_image(FIG7_XREF)
    dark = (img[..., 0] > 0.6) & (img[..., 1] < 0.15) & (img[..., 2] < 0.25)
    cols = np.where(dark.any(0))[0]
    panels = np.split(cols, np.where(np.diff(cols) > 40)[0] + 1)[:2]
    out, err = {}, {}
    for name, cols in zip(["clean", "noisy"], panels):
        ys, xs = np.where(dark[:, cols[0]:cols[-1] + 1])
        x0, x1 = xs.min() + cols[0], xs.max() + cols[0]
        y0, y1 = ys.min(), ys.max()
        cw, ch = (x1 - x0) / 16, (y1 - y0) / 16
        rgb = np.zeros((16, 16, 3))
        for i in range(16):
            for j in range(16):
                cy, cx = int(y0 + (i + 0.5) * ch), int(x0 + (j + 0.5) * cw)
                rgb[i, j] = np.median(img[cy - 3:cy + 4, cx - 3:cx + 4].reshape(-1, 3), 0)
        out[name], e = _invert(rgb, "coolwarm", -1, 1)
        err[name] = float(e.mean())
    return out, err


def fig8_levels(info):
    """Relative levels (0-1 along 'jet') of the eight Fig. 8 maps at the 16 electrodes.

    The maps are filled ellipses taken as the head outline; electrodes are placed
    by MNE's 2-D projection normalised to the head radius, so positions are
    approximate. Keys follow FIG8_PANELS: (group, band, "W" or "NW").
    """
    img = figure_image(FIG8_XREF)
    sat = (img.max(-1) - img.min(-1)) > 0.25
    lab, _ = ndimage.label(ndimage.binary_closing(sat, iterations=3))
    maps = [sl for i, sl in enumerate(ndimage.find_objects(lab)) if (lab[sl] == i + 1).sum() > 30000]
    maps = sorted(maps, key=lambda s: (s[0].start > img.shape[0] / 2, s[1].start))
    sphere = _check_sphere(None, info)
    xy = _find_topomap_coords(info, picks=list(range(len(info.ch_names))), sphere=sphere) / sphere[3]
    xy = np.clip(xy, -0.92, 0.92)   # Fp/O leads project just outside the head circle
    out = {}
    for key, sl in zip(FIG8_PANELS, maps):
        cy, cx = (sl[0].start + sl[0].stop) / 2, (sl[1].start + sl[1].stop) / 2
        ay, ax = (sl[0].stop - sl[0].start) / 2, (sl[1].stop - sl[1].start) / 2
        rgb = np.array([np.median(img[int(cy - y * ay) - 4:int(cy - y * ay) + 5,
                                      int(cx + x * ax) - 4:int(cx + x * ax) + 5].reshape(-1, 3), 0)
                        for x, y in xy])
        out[key] = _invert(rgb, "jet", 0, 1, n=256)[0]
    return out
