"""FIGURE 14 — Per-trace error against how much there is to predict.

Each point is one held-out trace: observed flux variability (sd) vs the
prediction RMSE on that trace, per chip. On the diagonal RMSE = sd, i.e.
per-trace R^2 = 0 (no better than the trace mean); below it the model has skill.
Shows why pooled R^2 is high while per-trace R^2 is negative for near-zero
traces: their absolute error is small, there is just almost no signal.
Naive port (level 0, batch 32). Data: results/preds/*.npz.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accel_style import CHIP_COLOR, CHIP_MARKER, FLUX_UNIT, INK2, load, pred_runs, save, style  # noqa: E402


def main():
    style()
    _, runs, chips = load()
    P = pred_runs(runs, chips)
    fig, ax = plt.subplots(figsize=(6.4, 5))
    lo, hi = 5e-4, 1.0
    xx = np.array([lo, hi])
    ax.fill_between(xx, xx, hi * 2, color=INK2, alpha=0.07, lw=0)
    ax.plot(xx, xx, color=INK2, lw=1, ls="--")
    ax.text(1.5e-3, 0.4, "worse than the trace mean\n(per-trace R² < 0)", fontsize=8, color=INK2)
    ax.text(0.25, 6e-3, "model has skill\n(per-trace R² > 0)", fontsize=8, color=INK2, ha="center")
    for chip, d in P.items():
        m = d["mask"] > 0
        sd, rmse = [], []
        for i in range(len(m)):
            y, p = d["targets_phys"][i][m[i]], d["preds_phys"][i][m[i]]
            sd.append(y.std()); rmse.append(np.sqrt(np.mean((y - p) ** 2)))
        sd, rmse = np.array(sd), np.array(rmse)
        skill = int((rmse < sd).sum())
        ax.scatter(sd, rmse, color=CHIP_COLOR[chip], marker=CHIP_MARKER[chip], s=42, edgecolor="white",
                   linewidth=1, zorder=3, alpha=0.9, label=f"{chip}: skill on {skill}/{len(sd)} traces")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo * 4, hi)
    ax.set_xlabel(f"observed flux variability, sd ({FLUX_UNIT})")
    ax.set_ylabel(f"prediction RMSE on the trace ({FLUX_UNIT})")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Per-trace skill fails where the flux barely varies; the error floor stays ~0.01")
    fig.text(0.0, -0.04, "One point per held-out trace (30), naive port (level 0, batch 32). "
             "Dashed: RMSE = sd.", fontsize=8, color=INK2)
    save(fig, "fig14_per_trace_skill.png")


if __name__ == "__main__":
    main()
