"""FIGURE 9 — Bigger batches raise throughput but not time-to-a-usable-model.

(a) Training throughput vs batch size (log-log) against ideal linear scaling:
    the 2016-step recurrence keeps per-step time nearly flat in batch only up to a point.
(b) Optimizer steps needed to first reach validation R^2 >= target: roughly constant,
    so fewer steps per epoch do not shorten training at a fixed learning rate.
Data: results/runs.json, opt level 2.
"""
import math
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accel_style import CHIP_COLOR, CHIP_MARKER, INK2, get, load, save, style  # noqa: E402

BATCHES = [32, 64, 128, 240]


def steps_to_target(r, target):
    for i, v in enumerate(r["history"]["val_r2"]):
        if v >= target:
            return (i + 1) * math.ceil(r["n_train"] / r["batch_size"])
    return None


def main():
    style()
    d, runs, chips = load()
    target = d["target_r2"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9))

    label_dy = {"A100": 9, "TPU v5e": -9, "TPU v6e": 0}
    tp_dy = {"A100": 0, "TPU v5e": -8, "TPU v6e": 6}   # v5e and v6e end within ~10% of each other
    for chip in chips:
        rs = [get(runs, chip, 2, bs) for bs in BATCHES]
        rs = [r for r in rs if r]
        bs = [r["batch_size"] for r in rs]
        tp = [r["throughput_traces_per_s"] for r in rs]
        c, m = CHIP_COLOR[chip], CHIP_MARKER[chip]
        a.plot(bs, tp, color=c, marker=m, label=chip, zorder=3)
        a.plot(bs, [tp[0] * x / bs[0] for x in bs], color=c, ls="--", lw=1, alpha=0.6, zorder=2)
        a.annotate(f"{chip}  {tp[-1] / tp[0]:.1f}× for {bs[-1] / bs[0]:.1f}× batch",
                   (bs[-1], tp[-1]), xytext=(6, tp_dy.get(chip, 0)), textcoords="offset points", fontsize=8, color=INK2, va="center")

        st = [(r["batch_size"], steps_to_target(r, target)) for r in rs]
        reached = [(x, s) for x, s in st if s]
        b.plot([x for x, _ in reached], [s for _, s in reached], color=c, marker=m, label=chip, zorder=3)
        s0, s1 = reached[0][1], reached[-1][1]
        b.annotate(f"{chip}: {s0} → {s1} steps ({1 - s1 / s0:.0%} fewer)", reached[-1],
                   xytext=(8, label_dy.get(chip, 0)), textcoords="offset points", fontsize=8, color=INK2, va="center")

    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCHES, [str(x) for x in BATCHES])
        ax.minorticks_off()
        ax.set_xlabel("batch size (traces per step)")
        ax.set_xlim(26, 900)
    a.set_yscale("log")
    a.set_ylabel("training throughput (traces/s)")
    a.set_title("(a) 7.5× the batch buys 2.4–3.8× the throughput…")
    a.text(34, a.get_ylim()[1] * 0.75, "dashed = ideal linear scaling", fontsize=8, color=INK2)
    b.set_ylim(0, None)
    b.set_ylabel(f"optimizer steps to val R² ≥ {target:.2f}")
    b.set_title("(b) …but needs nearly as many optimizer steps")
    fig.text(0.0, -0.05, "Opt level 2, same data and learning rate (1e-3) at every batch size. "
             "Batch 240 = whole training set per step.", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "fig9_batch_scaling.png")


if __name__ == "__main__":
    main()
