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
from matplotlib.patches import Patch

# ---------------------------------------------------------
# Font and Style Settings (Pretendard)
# ---------------------------------------------------------
plt.rcParams["font.family"] = "Pretendard"
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 13

# Colors
COLOR_OUTPUT = "#3478BA"
COLOR_GAP = "#F9A825"
COLOR_UNDER = "#E53935"
GAP_ALPHA = 0.8

# ---------------------------------------------------------
# ARC Reliability Bin Summary (from arc_paper_extract (1).md)
# ---------------------------------------------------------
bins = np.arange(0.05, 1.0, 0.1)
width = 0.1

ece_stage = {
    "Stage 1": 0.0726,
    "Stage 2": 0.0542,
    "Stage 3": 0.0537,
    "Stage 4": 0.0480,
}

def _arr(vals_20_to_100):
    out = [0.0, 0.0] + [float(x) for x in vals_20_to_100]
    return np.asarray(out, dtype=float)

# Stage 1
acc_s1 = _arr([0.4070, 0.3181, 0.4006, 0.4801, 0.5697, 0.6320, 0.7062, 0.9273])
conf_s1 = _arr([0.2836, 0.3636, 0.4575, 0.5497, 0.6498, 0.7519, 0.8536, 0.9741])

# Stage 2
acc_s2 = _arr([0.2417, 0.3177, 0.4291, 0.5542, 0.6528, 0.6814, 0.7667, 0.9506])
conf_s2 = _arr([0.2875, 0.3618, 0.4622, 0.5445, 0.6487, 0.7504, 0.8539, 0.9751])

# Stage 3
acc_s3 = _arr([0.2563, 0.4127, 0.4448, 0.5188, 0.6228, 0.7064, 0.7874, 0.9541])
conf_s3 = _arr([0.2874, 0.3637, 0.4528, 0.5493, 0.6500, 0.7517, 0.8528, 0.9743])

# Stage 4
acc_s4 = _arr([0.3640, 0.3437, 0.4214, 0.5677, 0.6420, 0.7263, 0.8091, 0.9573])
conf_s4 = _arr([0.2866, 0.3584, 0.4560, 0.5481, 0.6487, 0.7515, 0.8536, 0.9738])

stages = [
    ("Stage 1", acc_s1, conf_s1, ece_stage["Stage 1"]),
    ("Stage 2", acc_s2, conf_s2, ece_stage["Stage 2"]),
    ("Stage 3", acc_s3, conf_s3, ece_stage["Stage 3"]),
    ("Stage 4", acc_s4, conf_s4, ece_stage["Stage 4"]),
]

fig, axes = plt.subplots(1, len(stages), figsize=(20, 5), sharey=True)
if len(stages) == 1:
    axes = [axes]

for ax, (title, acc, conf, ece) in zip(axes, stages):
    acc_pct = acc * 100.0
    conf_pct = conf * 100.0

    ax.plot([0, 1.0], [0, 100], linestyle="--", color="#555555", linewidth=2, zorder=3)

    for b, a, c in zip(bins, acc_pct, conf_pct):
        if a == 0 and c == 0:
            continue
        if c >= a:
            ax.bar(b, a, width=width, color=COLOR_OUTPUT, edgecolor="black", linewidth=1.2, zorder=2)
            gap_height = c - a
            if gap_height > 0:
                ax.bar(b, gap_height, bottom=a, width=width, color=COLOR_GAP, alpha=GAP_ALPHA,
                       edgecolor="black", linewidth=1.2, zorder=2)
        else:
            ax.bar(b, a, width=width, color=COLOR_OUTPUT, edgecolor="black", linewidth=1.2, zorder=2)
            gap_height = a - c
            ax.bar(b, gap_height, bottom=c, width=width, color=COLOR_UNDER, alpha=0.85,
                   edgecolor="black", hatch="//", linewidth=1.2, zorder=3)

    ax.set_title(title, pad=15)
    ax.set_xlabel("Confidence")
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 100])
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, axis="y", linestyle="-", alpha=0.3, zorder=0)

    props = dict(boxstyle="square,pad=0.4", facecolor="#f8f9fa", alpha=0.9, edgecolor="#ced4da")
    ax.text(
        0.95, 0.05,
        f"ECE = {ece * 100:.2f}",
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
        zorder=4,
        fontsize=14,
    )

axes[0].set_ylabel("Accuracy (%)")
legend_elements = [
    Patch(facecolor=COLOR_OUTPUT, edgecolor="black", label="Outputs"),
    Patch(facecolor=COLOR_GAP, alpha=GAP_ALPHA, edgecolor="black", label="Overconfidence Gap"),
    Patch(facecolor=COLOR_UNDER, alpha=0.85, hatch="//", edgecolor="black", label="Underconfidence Gap"),
]
axes[0].legend(handles=legend_elements, loc="upper left", fontsize=11)

plt.tight_layout()
plt.show()

