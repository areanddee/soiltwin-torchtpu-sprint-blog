"""FIGURE 11 — Headline: how long until each chip has a usable model?

Validation R^2 against cumulative training seconds (device-synced epoch times,
first epoch includes compilation). Dashed = naive port (level 0, batch 32);
solid = that chip's fastest configuration to the target.
Data: results/runs.json.
"""
import itertools
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accel_style import CHIP_COLOR, CHIP_MARKER, INK2, get, load, save, style  # noqa: E402


def main():
    style()
    d, runs, chips = load()
    target = d["target_r2"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for chip in chips:
        naive = get(runs, chip, 0, 32)
        cands = [r for r in runs if r["chip"] == chip and r["time_to_target_s"] and r["experiment"] == "ladder"]
        best = min(cands, key=lambda r: r["time_to_target_s"])
        for r, ls, lab in [(naive, "--", "naive"), (best, "-", f"best: {best['config']}")]:
            t = list(itertools.accumulate(r["history"]["epoch_time_s"]))
            ax.plot(t, r["history"]["val_r2"], color=CHIP_COLOR[chip], ls=ls, lw=2 if ls == "-" else 1.4,
                    alpha=1 if ls == "-" else 0.7, label=f"{chip} {lab}")
            tt = r["time_to_target_s"]
            if tt:
                ax.plot([tt], [target], marker=CHIP_MARKER[chip], color=CHIP_COLOR[chip], ms=8, zorder=4,
                        mec="white", mew=1.5)
                if ls == "-":
                    ax.plot([tt, tt], [0.5, target], color=CHIP_COLOR[chip], lw=1, ls=":", zorder=1)
                    left = chip == "A100"   # keep the label off the neighbouring naive curve
                    y = {"TPU v6e": 0.60}.get(chip, 0.515)   # stack the two TPU labels (their targets are ~4 s apart)
                    ax.text(tt / 1.06 if left else tt * 1.06, y, f"{chip}\n{tt:.1f} s", fontsize=8,
                            color=INK2, va="bottom", ha="right" if left else "left")
    ax.axhline(target, color=INK2, lw=1, ls=":")
    ax.text(0.55, target + 0.01, f"usable: val R² ≥ {target:.2f}", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(0.5, None)
    ax.set_xlabel("training time, seconds (log scale; first epoch includes compilation)")
    ax.set_ylabel("validation R²")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.2), fontsize=8)
    ax.set_title("Time to a usable model")
    save(fig, "fig11_time_to_accuracy.png")


if __name__ == "__main__":
    main()
