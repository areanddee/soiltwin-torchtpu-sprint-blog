"""FIGURE 12 — Does the model predict the flux, and do the chips agree?

Predicted vs observed CO2 flux for every half-hour step of the 30 held-out test
traces (naive port: level 0, batch 32), one panel per chip, plus a row of
chip-vs-chip prediction panels (every pair). Density is log-scaled hexbins.
Data: results/preds/*.npz.
"""
import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accel_style import CHIP_COLOR, FLUX_UNIT, INK2, SURFACE, load, pred_runs, save, style  # noqa: E402

LIM = (-0.1, 3.7)


def stats(y, p):
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return r2, np.sqrt(np.mean((y - p) ** 2)), np.mean(p - y)


def hexpanel(ax, x, y, color):
    cmap = LinearSegmentedColormap.from_list("c", [SURFACE, color])
    hb = ax.hexbin(x, y, gridsize=60, extent=LIM + LIM, bins="log", cmap=cmap, mincnt=1, linewidths=0)
    ax.plot(LIM, LIM, color=INK2, lw=1, ls="--", zorder=3)
    ax.set_xlim(LIM); ax.set_ylim(LIM); ax.set_aspect("equal")
    return hb


def main():
    style()
    _, runs, chips = load()
    P = pred_runs(runs, chips)
    names = list(P)
    pairs = list(itertools.combinations(names, 2))
    ncol = max(len(names), len(pairs))
    nrow = 2 if pairs else 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.9 * nrow), squeeze=False)
    for ax, chip in zip(axes[0], names):
        d = P[chip]; m = d["mask"] > 0
        y, p = d["targets_phys"][m], d["preds_phys"][m]
        hexpanel(ax, y, p, CHIP_COLOR[chip])
        r2, rmse, bias = stats(y, p)
        ax.text(0.04, 0.96, f"R² {r2:.3f}\nRMSE {rmse:.3f}\nbias {bias:+.3f}", transform=ax.transAxes,
                va="top", fontsize=8, color=INK2)
        ax.set_title(f"{chip} vs observed")
        ax.set_xlabel(f"observed flux ({FLUX_UNIT})")
        ax.set_ylabel(f"predicted flux ({FLUX_UNIT})")
    for ax in axes[0][len(names):]:
        ax.axis("off")
    for ax, (a, b) in zip(axes[1] if pairs else [], pairs):
        m = P[a]["mask"] > 0
        pa, pb = P[a]["preds_phys"][m], P[b]["preds_phys"][m]
        hexpanel(ax, pa, pb, INK2)
        r = np.corrcoef(pa, pb)[0, 1]
        ax.text(0.04, 0.96, f"r {r:.4f}\nRMS diff {np.sqrt(np.mean((pa - pb) ** 2)):.3f}",
                transform=ax.transAxes, va="top", fontsize=8, color=INK2)
        ax.set_title(f"{b} vs {a}")
        ax.set_xlabel(f"{a} predicted ({FLUX_UNIT})")
        ax.set_ylabel(f"{b} predicted ({FLUX_UNIT})")
    for ax in (axes[1][len(pairs):] if pairs else []):
        ax.axis("off")
    title = ("Every chip learns the flux; the two TPU generations train almost the same model, the A100 a different one"
             if len(names) > 2 else
             "Both chips learn the flux; their models differ by about as much as each misses reality")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.01, -0.02, "Every half-hour step of the 30 held-out test traces (60,480 points); naive port "
             "(level 0, batch 32). Top: each chip vs observations. Bottom: chip vs chip. Colour = log point density; "
             "dashed = perfect agreement. Inference re-run on CPU from each chip's checkpoint.", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "fig12_pred_vs_obs.png")


if __name__ == "__main__":
    main()
