"""Shared data loading + style for the accelerator figures (fig8-fig11).

Colors follow the chip, never its rank, and every chip also has its own marker
so identity is never color-alone. Palette validated with the dataviz
validator (all-pairs CVD dE >= 9.2, normal-vision >= 24); aqua is < 3:1 on the
light surface, so series are always direct-labelled.
"""
import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "results", "runs.json")
OUT_DIR = os.path.join(HERE, "..", "out")

CHIP_ORDER = ["A100", "TPU v5e", "TPU v6e"]
CHIP_COLOR = {"A100": "#2a78d6", "TPU v5e": "#eb6834", "TPU v6e": "#1baf7a"}
CHIP_MARKER = {"A100": "o", "TPU v5e": "s", "TPU v6e": "D"}

INK = "#0b0b0b"        # primary text
INK2 = "#52514e"       # secondary text / annotations
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

LEVEL_NAME = {0: "reference loop", 1: "no per-batch host syncs", 2: "+ data resident on device"}


def load():
    d = json.load(open(DATA))
    runs = d["runs"]
    chips = [c for c in CHIP_ORDER if any(r["chip"] == c for r in runs)]
    return d, runs, chips


def get(runs, chip, level, batch, experiment="ladder"):
    for r in runs:
        if (r["chip"] == chip and r["opt_level"] == level and r["batch_size"] == batch
                and r.get("experiment", "ladder") == experiment):
            return r
    return None


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False,
    })


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, name)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


PREDS = os.path.join(HERE, "..", "results", "preds")
FLUX_UNIT = "µmol m⁻² s⁻¹"


def load_preds(name):
    """Test-set predictions in results/preds/, e.g. 'gpu_a100_ladder_o0_b32'."""
    import numpy as np
    d = np.load(os.path.join(PREDS, f"{name}.npz"))
    return {k: d[k] for k in d.files}


def pred_runs(runs, chips, level=0, batch=32):
    """{chip: predictions} for one ladder config, keyed in CHIP_ORDER."""
    out = {}
    for chip in chips:
        r = get(runs, chip, level, batch)
        path = os.path.join(PREDS, f"{r['tag']}_{r['config']}.npz") if r else None
        if path and os.path.exists(path):
            out[chip] = load_preds(f"{r['tag']}_{r['config']}")
    return out
