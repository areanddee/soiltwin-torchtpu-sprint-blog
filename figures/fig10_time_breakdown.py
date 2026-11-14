"""FIGURE 10 — Where the wall-clock goes in a short training job.

Per chip, naive port (level 0, batch 32): startup outside the training loop
(imports, zarr load, test eval: `time` real minus train wall), first-epoch
overhead above a steady epoch (compilation / warm-up), and the steady epochs.
Data: results/runs.json.
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matplotlib.patches import Patch  # noqa: E402
from _accel_style import CHIP_COLOR, INK, INK2, SURFACE, get, load, save, style  # noqa: E402

CONFIGS = [(0, 32, "naive port (level 0, batch 32)")]


def main():
    style()
    _, runs, chips = load()
    rows = []
    for chip in chips:
        for lvl, bs, label in CONFIGS:
            r = get(runs, chip, lvl, bs)
            if r and r["real_s"]:
                warm = r["first_epoch_s"] - r["steady_epoch_s"]
                rows.append((chip, label, r["real_s"] - r["train_wall_s"], warm, r["train_wall_s"] - warm, r))

    fig, ax = plt.subplots(figsize=(8, 0.9 + 0.8 * len(rows)))
    seg_alpha = [0.35, 0.65, 1.0]
    seg_name = ["startup + data load + test", "first-epoch overhead (compile)", "training epochs"]
    for y, (chip, label, startup, warm, steady, r) in enumerate(rows):
        left = 0
        for k, (w, al) in enumerate(zip([startup, warm, steady], seg_alpha)):
            ax.barh(y, w, left=left, color=CHIP_COLOR[chip], alpha=al, edgecolor=SURFACE, linewidth=2, height=0.6)
            if w > 3:
                ax.text(left + w / 2, y, f"{w:.0f} s", ha="center", va="center", fontsize=8,
                        color=INK if al < 0.9 else "white")
            left += w
        ax.text(left + 1, y, f"{left:.0f} s total · {r['epochs_run']} epochs", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)), [c for c, *_ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("wall-clock seconds")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(sum(x[2:5]) for x in rows) * 1.3)
    ax.set_title("Fixed costs are a large share of a one-minute job", pad=22)
    ax.legend(handles=[Patch(facecolor=INK2, alpha=al, label=n) for al, n in zip(seg_alpha, seg_name)],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=8)
    fig.text(0.0, -0.12, "Naive port (level 0, batch 32). Epoch counts differ because early stopping "
             "triggers at different epochs.", fontsize=8, color=INK2)
    save(fig, "fig10_time_breakdown.png")


if __name__ == "__main__":
    main()
