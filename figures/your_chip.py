"""Your runs next to our reference chips: steady epoch time, time to a usable model, and test R^2.

`yours` is a list of soiltwin.train.train() outputs. Reference runs come from results/runs.json.
"""
import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accel_style import CHIP_COLOR, CHIP_MARKER, INK, INK2, get, load, style  # noqa: E402

YOU = "#7a4fc9"   # your chip: violet, distinct from the three reference colors


def _label(r):
    return f"o{r['opt_level']}_b{r['batch_size']}"


def _time_to(history, target):
    t = 0.0
    for dt, r2 in zip(history["epoch_time_s"], history["val_r2"]):
        t += dt
        if r2 >= target:
            return t
    return None


def compare(yours, target=0.90):
    style()
    _, runs, chips = load()
    you_name = "You: " + yours[0]["results"]["device_name"]
    configs = [(o["results"]["opt_level"], o["results"]["batch_size"]) for o in yours]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.2, 1.4, 1]})

    # (a) steady epoch time per config, ours vs yours
    names = chips + [you_name]
    colors = [CHIP_COLOR[ch] for ch in chips] + [YOU]
    w = 0.8 / len(names)
    for j, (lvl, bs) in enumerate(configs):
        for i, name in enumerate(names):
            if name == you_name:
                v = yours[j]["results"]["steady_epoch_s"]
            else:
                r = get(runs, name, lvl, bs)
                v = r["steady_epoch_s"] if r else np.nan
            x = j + (i - (len(names) - 1) / 2) * w
            a.bar(x, v, w * 0.92, color=colors[i], label=name if j == 0 else None)
            if np.isfinite(v):
                a.text(x, v * 1.08, f"{v:.2f}" if v < 10 else f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
    a.set_yscale("log")
    a.set_xticks(range(len(configs)), [f"o{l}_b{bs}" for l, bs in configs])
    a.set_ylabel("steady epoch time (s, log)")
    a.set_title("(a) seconds per epoch")

    # (b) validation R^2 vs training time, naive port (first config)
    lvl0, bs0 = configs[0]
    for ch in chips:
        r = get(runs, ch, lvl0, bs0)
        if r:
            t = list(itertools.accumulate(r["history"]["epoch_time_s"]))
            b.plot(t, r["history"]["val_r2"], color=CHIP_COLOR[ch], lw=1.4, alpha=0.8, label=ch)
    h = yours[0]["history"]
    b.plot(list(itertools.accumulate(h["epoch_time_s"])), h["val_r2"], color=YOU, lw=2.4, label=you_name)
    b.axhline(target, color=INK2, lw=1, ls=":")
    b.set_xscale("log")
    b.set_ylim(max(0.0, min(0.5, np.floor(min(h["val_r2"]) * 10) / 10)), 1.0)
    b.set_xlabel("training time (s, log; first epoch includes compilation)")
    b.set_ylabel("validation R²")
    b.set_title(f"(b) learning curve, o{lvl0}_b{bs0}")
    tt = _time_to(h, target)
    b.text(0.98, 0.03, f"you reach R² {target:.2f} in " + (f"{tt:.1f} s" if tt else "— (not reached)"),
           transform=b.transAxes, ha="right", va="bottom", fontsize=8, color=INK)

    # (c) test R^2
    vals = []
    for j, (lvl, bs) in enumerate(configs):
        for i, name in enumerate(names):
            if name == you_name:
                v = yours[j]["results"]["test_r2"]
            else:
                r = get(runs, name, lvl, bs)
                v = r["test_r2"] if r else np.nan
            x = j + (i - (len(names) - 1) / 2) * w
            c.bar(x, v, w * 0.92, color=colors[i])
            vals.append((x, v))
    lo = min(0.85, np.floor((min(v for _, v in vals if np.isfinite(v)) - 0.05) * 20) / 20)
    for x, v in vals:
        if np.isfinite(v):
            c.text(x, v + (1 - lo) * 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK2, rotation=90)
    c.set_ylim(lo, 1.0 + (1 - lo) * 0.12)
    c.set_xticks(range(len(configs)), [f"o{l}_b{bs}" for l, bs in configs])
    c.set_ylabel("test R²")
    c.set_title("(c) test R² (30 traces)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in colors]
    fig.legend(handles, names, loc="upper center", ncol=len(names), fontsize=9, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    return fig
