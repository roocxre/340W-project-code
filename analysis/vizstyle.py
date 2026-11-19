"""Figure style for the analysis stages (static PNGs for the paper).

Colors are the dataviz reference palette, validated with its script:
- LEVEL: n-back level is ordinal, so one hue (blue) stepped light -> dark (ordinal check passes).
- SERIES: categorical slots in fixed order, for models and datasets (adjacent CVD check passes;
  slots 3-5 sit below 3:1 contrast, so every such chart carries labels or a table).
- DIVERGING: blue <-> red through a neutral gray, for changes relative to 0-back.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
LEVEL = {0: "#86b6ef", 1: "#3987e5", 2: "#1c5cab", 3: "#0d366b"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DATASET = {"shin": SERIES[0], "cog": SERIES[1]}
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#3987e5", "#f0efec", "#e34948", "#a62f2f"])


def use():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 1.6, "lines.markersize": 5, "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
