import os
import tempfile

_mpl_dir = os.path.join(
    tempfile.gettempdir(),
    f"matplotlib-{os.getuid()}" if hasattr(os, "getuid") else "matplotlib",
)
os.environ.setdefault("MPLCONFIGDIR", _mpl_dir)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ---------------------------------------------------------
# Font and Style Settings
# ---------------------------------------------------------
plt.rcParams["font.family"] = "Pretendard"
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["text.color"] = "black"

# ---------------------------------------------------------
# Color Palette & Styling
# ---------------------------------------------------------
COLOR_BLUE = "#3478BA"
COLOR_ORANGE = "#F9A825"
COLOR_RED = "#E53935"

color_c2c = COLOR_BLUE
color_w2w = COLOR_ORANGE
color_lost = COLOR_RED

color_gained_correct = mcolors.to_rgba(COLOR_BLUE, alpha=0.45)
color_gained_wrong = mcolors.to_rgba(COLOR_ORANGE, alpha=0.45)

# ---------------------------------------------------------
# ARC p=80 settings (from arc_paper_extract (1).md)
# ---------------------------------------------------------
N_TOTAL = 1165
P = 80

# From "Fixed Percentile p=80 Summary" (Transition 1->2 row)
low_ratio = 0.8000
low_acc_stage1 = 0.7484

# Per-transition low-subset flip rates at p=80 (within low subset)
# 1->2, 2->3, 3->4
w2c_step_rates = np.array([0.0, 0.0575, 0.0263, 0.0257], dtype=float)
c2w_step_rates = np.array([0.0, 0.0375, 0.0271, 0.0201], dtype=float)

# ---------------------------------------------------------
# Build counts with rounding (p=80 low-subset)
# ---------------------------------------------------------
N_LOW = int(np.round(low_ratio * N_TOTAL))

base_c_val = int(np.round(low_acc_stage1 * N_LOW))
base_w_val = int(N_LOW - base_c_val)

gained_correct_counts = np.round(w2c_step_rates * N_LOW).astype(int)
gained_wrong_counts = np.round(c2w_step_rates * N_LOW).astype(int)

w2c_counts = np.cumsum(gained_correct_counts)
c2w_counts = np.cumsum(gained_wrong_counts)

stages = ["Stage 1\n(Initial)", "Stage 1 -> 2", "Stage 1 -> 3", "Stage 1 -> 4"]
x = np.arange(len(stages))
width = 0.22
offset = 0.14

base_c = np.full(len(stages), base_c_val, dtype=int)
base_w = np.full(len(stages), base_w_val, dtype=int)

c2c_counts = base_c - c2w_counts
w2w_counts = base_w - w2c_counts

print(f"[ARC p={P}] N_TOTAL={N_TOTAL}, low_ratio={low_ratio}, N_LOW≈{N_LOW}")
print(f"Stage1 low-subset base: correct={base_c_val}, wrong={base_w_val}, acc≈{base_c_val/N_LOW:.4f}")
print("Incremental gained (per step):")
print("W2C:", gained_correct_counts.tolist())
print("C2W:", gained_wrong_counts.tolist())

# ---------------------------------------------------------
# Decide whether to use broken axis
# ---------------------------------------------------------
BROKEN_AXIS_THRESHOLD = 2500  # MMLU-scale only
use_broken_axis = max(int(base_c_val), int(base_w_val)) >= int(BROKEN_AXIS_THRESHOLD)

if not use_broken_axis:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    ax.bar(x - offset, c2c_counts, width, color=color_c2c, edgecolor="black", zorder=3)
    ax.bar(x - offset, c2w_counts, width, bottom=c2c_counts, color=color_lost,
           hatch="//", edgecolor="black", zorder=3)
    ax.bar(x - offset, -w2w_counts, width, color=color_w2w, edgecolor="black", zorder=3)
    ax.bar(x - offset, -w2c_counts, width, bottom=-w2w_counts, color=color_lost,
           hatch="//", edgecolor="black", zorder=3)

    ax.bar(x + offset, gained_correct_counts, width, color=color_gained_correct,
           edgecolor="black", zorder=3)
    ax.bar(x + offset, -gained_wrong_counts, width, color=color_gained_wrong,
           edgecolor="black", zorder=3)

    ax.axhline(0, color="black", linewidth=1.5, zorder=4)

    pad = max(30, int(0.08 * max(base_c_val, base_w_val)))
    ax.set_ylim(-(base_w_val + int(pad * 1.4)), base_c_val + int(pad * 1.4))

    ax.set_xticks(x)
    ax.set_xticklabels(stages)

    formatter = ticker.FuncFormatter(lambda y, pos: f"{abs(int(y)):,}")
    ax.yaxis.set_major_formatter(formatter)
    ax.set_ylabel("Sample Count", fontsize=18)

    # --- Count annotations ---
    y_off = max(8, int(0.02 * (base_c_val + base_w_val)))
    for i in range(len(stages)):
        if i == 0:
            ax.text(x[i] - offset, base_c[i] + y_off, f"{base_c[i]:,}",
                    ha="center", va="bottom", color="black", fontweight="bold")
            ax.text(x[i] - offset, -base_w[i] - y_off, f"{base_w[i]:,}",
                    ha="center", va="top", color="black", fontweight="bold")
        else:
            ax.text(x[i] - offset, c2c_counts[i] - y_off, f"{c2c_counts[i]:,}",
                    ha="center", va="top", color="white", fontweight="bold")
            ax.text(x[i] - offset, c2c_counts[i] + max(1, c2w_counts[i]) / 2, f"{c2w_counts[i]:,}",
                    ha="center", va="center", color="white", fontweight="bold")
            ax.text(x[i] - offset, base_c[i] + y_off, f"{base_c[i]:,}",
                    ha="center", va="bottom", color="black", fontweight="bold")

            ax.text(x[i] - offset, -w2w_counts[i] + y_off, f"{w2w_counts[i]:,}",
                    ha="center", va="bottom", color="black", fontweight="bold")
            ax.text(x[i] - offset, -w2w_counts[i] - max(1, w2c_counts[i]) / 2, f"{w2c_counts[i]:,}",
                    ha="center", va="center", color="white", fontweight="bold")
            ax.text(x[i] - offset, -base_w[i] - y_off, f"{base_w[i]:,}",
                    ha="center", va="top", color="black", fontweight="bold")

            ax.text(x[i] + offset, gained_correct_counts[i] + y_off, f"{gained_correct_counts[i]:,}",
                    ha="center", va="bottom", fontweight="bold")
            ax.text(x[i] + offset, -gained_wrong_counts[i] - y_off, f"{gained_wrong_counts[i]:,}",
                    ha="center", va="top", fontweight="bold")

    legend_elements = [
        Patch(facecolor=color_c2c, edgecolor="black", label="Maintained Correct"),
    Patch(facecolor=color_w2w, edgecolor="black", label="Maintained Wrong"),
    Patch(facecolor=color_lost, hatch="//", edgecolor="black", label="Cumulative W2C/C2W"),
    Patch(facecolor=color_gained_correct, edgecolor="black", label="W2C"),
    Patch(facecolor=color_gained_wrong, edgecolor="black", label="C2W"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.18),
              ncol=3, fontsize=13)
    plt.tight_layout()
    plt.show()
    raise SystemExit(0)

# ---------------------------------------------------------
# Create 3 subplots for the Broken Axis (Cut) effect
# ---------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, sharex=True, figsize=(12, 10),
    gridspec_kw={"height_ratios": [3, 1.3, 3]},
)
fig.subplots_adjust(hspace=0.1)

for ax in [ax1, ax2, ax3]:
    ax.bar(x - offset, c2c_counts, width, color=color_c2c, edgecolor="black", zorder=3)
    ax.bar(x - offset, c2w_counts, width, bottom=c2c_counts, color=color_lost,
           hatch="//", edgecolor="black", zorder=3)

    ax.bar(x - offset, -w2w_counts, width, color=color_w2w, edgecolor="black", zorder=3)
    ax.bar(x - offset, -w2c_counts, width, bottom=-w2w_counts, color=color_lost,
           hatch="//", edgecolor="black", zorder=3)

    ax.bar(x + offset, gained_correct_counts, width, color=color_gained_correct,
           edgecolor="black", zorder=3)
    ax.bar(x + offset, -gained_wrong_counts, width, color=color_gained_wrong,
           edgecolor="black", zorder=3)

    ax.axhline(0, color="black", linewidth=1.5, zorder=4)

# Y limits
mid_lim = max(
    300,
    int(np.max(np.abs(gained_correct_counts)) + 120),
    int(np.max(np.abs(gained_wrong_counts)) + 120),
)
ax2.set_ylim(-mid_lim, mid_lim)

# Dynamic pads for broken axes (robust for small datasets)
top_pad_low = max(60, int(0.15 * base_c_val))
top_pad_high = max(60, int(0.08 * base_c_val))
ax1_low = max(base_c_val - top_pad_low, int(mid_lim * 1.25))
ax1_high = base_c_val + top_pad_high
ax1.set_ylim(ax1_low, ax1_high)

bot_pad_low = max(60, int(0.15 * base_w_val))
bot_pad_high = max(60, int(0.08 * base_w_val))
ax3_low = -(base_w_val + bot_pad_high)
ax3_high = -max(base_w_val - bot_pad_low, int(mid_lim * 1.25), 1)
ax3.set_ylim(ax3_low, ax3_high)

ax1.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["bottom"].set_visible(False)
ax3.spines["top"].set_visible(False)

ax1.tick_params(labeltop=False, bottom=False)
ax2.tick_params(bottom=False, top=False)
ax3.xaxis.tick_bottom()

# Cut marks
d = 0.015
kwargs = dict(color="black", clip_on=False, linewidth=1.5)
for top_ax, btm_ax in [(ax1, ax2), (ax2, ax3)]:
    if top_ax is ax1:
        y_top, y_btm = -d * 3, 1 - d
    else:
        y_top, y_btm = -d, 1 - d * 3
    top_ax.plot((-d, +d), (y_top, -y_top), transform=top_ax.transAxes, **kwargs)
    top_ax.plot((1 - d, 1 + d), (y_top, -y_top), transform=top_ax.transAxes, **kwargs)
    btm_ax.plot((-d, +d), (y_btm, 1 + d * 3 if btm_ax is ax3 else 1 + d),
                transform=btm_ax.transAxes, **kwargs)
    btm_ax.plot((1 - d, 1 + d), (y_btm, 1 + d * 3 if btm_ax is ax3 else 1 + d),
                transform=btm_ax.transAxes, **kwargs)

# Text annotations
for i in range(len(stages)):
    if i == 0:
        ax1.text(x[i] - offset, base_c[i] + 20, f"{base_c[i]:,}",
                 ha="center", va="bottom", color="black", fontweight="bold")
        ax3.text(x[i] - offset, -base_w[i] - 20, f"{base_w[i]:,}",
                 ha="center", va="top", color="black", fontweight="bold")
    else:
        ax1.text(x[i] - offset, c2c_counts[i] - 35, f"{c2c_counts[i]:,}",
                 ha="center", va="top", color="white", fontweight="bold")
        ax1.text(x[i] - offset, c2c_counts[i] + c2w_counts[i] / 2, f"{c2w_counts[i]:,}",
                 ha="center", va="center", color="white", fontweight="bold")
        ax1.text(x[i] - offset, base_c[i] + 20, f"{base_c[i]:,}",
                 ha="center", va="bottom", color="black", fontweight="bold")

        ax3.text(x[i] - offset, -w2w_counts[i] + 35, f"{w2w_counts[i]:,}",
                 ha="center", va="bottom", color="black", fontweight="bold")
        ax3.text(x[i] - offset, -w2w_counts[i] - w2c_counts[i] / 2, f"{w2c_counts[i]:,}",
                 ha="center", va="center", color="white", fontweight="bold")
        ax3.text(x[i] - offset, -base_w[i] - 20, f"{base_w[i]:,}",
                 ha="center", va="top", color="black", fontweight="bold")

        ax2.text(x[i] + offset, gained_correct_counts[i] + 20, f"{gained_correct_counts[i]:,}",
                 ha="center", va="bottom", fontweight="bold")
        ax2.text(x[i] + offset, -gained_wrong_counts[i] - 20, f"{gained_wrong_counts[i]:,}",
                 ha="center", va="top", fontweight="bold")

ax3.set_xticks(x)
ax3.set_xticklabels(stages)

formatter = ticker.FuncFormatter(lambda y, pos: f"{abs(int(y)):,}")
for ax in [ax1, ax2, ax3]:
    ax.yaxis.set_major_formatter(formatter)

fig.text(0.04, 0.5, "Sample Count", va="center", rotation="vertical", fontsize=18)

legend_elements = [
    Patch(facecolor=color_c2c, edgecolor="black", label="Maintained Correct"),
    Patch(facecolor=color_w2w, edgecolor="black", label="Maintained Wrong"),
    Patch(facecolor=color_lost, hatch="//", edgecolor="black", label="Cumulative W2C/C2W"),
    Patch(facecolor=color_gained_correct, edgecolor="black", label="W2C"),
    Patch(facecolor=color_gained_wrong, edgecolor="black", label="C2W"),
]
ax1.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.45),
           ncol=3, fontsize=13)

plt.tight_layout()
plt.show()

