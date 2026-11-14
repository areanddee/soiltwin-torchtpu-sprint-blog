"""FIGURE 8 — The same loop optimizations pay off differently on GPU and TPU.

Speedup of steady-state epoch time over each chip's own reference loop,
batch size 32, for the cumulative optimization levels in accel_opt.py.
Data: results/runs.json.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accel_style import CHIP_COLOR, INK2, LEVEL_NAME, get, load, save, style  # noqa: E402

LEVELS = [0, 1, 2]


def main():
    style()
    _, runs, chips = load()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    width = 0.8 / len(chips)
    x = np.arange(len(LEVELS))
    for i, chip in enumerate(chips):
        base = get(runs, chip, 0, 32)["steady_epoch_s"]
        sp = [base / get(runs, chip, lvl, 32)["steady_epoch_s"] for lvl in LEVELS]
        ms = [get(runs, chip, lvl, 32)["steady_epoch_s"] * 1000 for lvl in LEVELS]
        xs = x + (i - (len(chips) - 1) / 2) * width
        ax.bar(xs, sp, width * 0.92, color=CHIP_COLOR[chip], label=chip, zorder=2)
        for xx, s, m in zip(xs, sp, ms):
            ax.text(xx, s + 0.02, f"{s:.2f}×\n{m:.0f} ms", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.axhline(1.0, color=INK2, lw=1, ls=":", zorder=1)
    ax.set_xticks(x, [f"level {l}\n{LEVEL_NAME[l]}" for l in LEVELS])
    ax.set_ylabel("speedup vs own reference (×)")
    ax.set_ylim(0, max(1.6, ax.get_ylim()[1] + 0.15))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=len(chips))
    ax.set_title("Removing host syncs speeds up the GPU; the TPU doesn't notice")
    fig.text(0.0, -0.04, "Steady-state epoch time (median of epochs 2..N), batch 32, same data and command. "
             "Labels: speedup and ms per epoch.", fontsize=8, color=INK2)
    save(fig, "fig8_accel_ladder.png")


if __name__ == "__main__":
    main()
