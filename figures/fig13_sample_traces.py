"""FIGURE 13 — Sample test traces: observed flux vs each chip's prediction.

Three held-out traces chosen by rule, not by eye: the test traces at the 90th,
50th and 10th percentile of observed flux variability (std over the trace).
Naive port (level 0, batch 32). Data: results/preds/*.npz.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accel_style import CHIP_COLOR, FLUX_UNIT, INK, INK2, load, pred_runs, save, style  # noqa: E402

PCTS = [90, 50, 10]
STEP_DAYS = 0.5 / 24


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)


def main():
    style()
    _, runs, chips = load()
    P = pred_runs(runs, chips)
    ref = next(iter(P.values()))
    tg, mask = ref["targets_phys"], ref["mask"] > 0
    sd = np.array([tg[i][mask[i]].std() for i in range(len(tg))])
    order = np.argsort(sd)
    picks = [order[int(round(p / 100 * (len(sd) - 1)))] for p in PCTS]

    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 2.3 * len(picks)), sharex=True)
    for ax, pct, i in zip(axes, PCTS, picks):
        m = mask[i]
        t = np.arange(m.sum()) * STEP_DAYS
        y = tg[i][m]
        ax.plot(t, y, color=INK, lw=1.4, label="observed", zorder=3)
        notes = []
        for chip, d in P.items():
            p = d["preds_phys"][i][m]
            ax.plot(t, p, color=CHIP_COLOR[chip], lw=1.1, alpha=0.9, label=f"{chip} predicted", zorder=2)
            notes.append(f"{chip}: R² {r2(y, p):.2f}, RMSE {np.sqrt(np.mean((y - p) ** 2)):.3f}")
        ax.set_title(f"{pct}th-percentile variability · site {ref['site_id'][i]}, {ref['year'][i]} "
                     f"(observed sd {sd[i]:.3f})", fontsize=9.5)
        ax.text(1.01, 0.5, "\n".join(notes), transform=ax.transAxes, fontsize=8, color=INK2, va="center")
        ax.set_ylabel(f"flux\n({FLUX_UNIT})")
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.13), ncol=1 + len(P), fontsize=8)
    axes[-1].set_xlabel("days since trace start (30-minute steps)")
    fig.suptitle("High-flux traces are tracked closely; near-zero traces are small in error but low in R²",
                 x=0.01, y=1.0, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "fig13_sample_traces.png")


if __name__ == "__main__":
    main()
