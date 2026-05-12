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
COLOR_OUTPUT = "#3478BA"  # Outputs (Accuracy)
COLOR_GAP = "#F9A825"     # Overconfidence gap (Conf > Acc)
COLOR_UNDER = "#E53935"   # Underconfidence gap (Acc > Conf)
GAP_ALPHA = 0.8

# ---------------------------------------------------------
# CSQA Reliability Bin Summary (from csqa_paper_extract (1).md)
# Conf bins: 00-10, 10-20, ..., 90-100
# Values are per-bin means (Acc, AvgConf) for each stage.
# ---------------------------------------------------------
bins = np.arange(0.05, 1.0, 0.1)
width = 0.1

# Stage-wise ECE (scalar, from Stage-Wise Metrics table)
ece_stage = {
    "Stage 1": 0.0974,
    "Stage 2": 0.0683,
    "Stage 3": 0.0660,
    "Stage 4": 0.0637,
    "Stage 5": 0.0611,
}

# Helper to build 10-length arrays with first two bins empty
def _arr(vals_20_to_100):
    out = [0.0, 0.0] + [float(x) for x in vals_20_to_100]
    return np.asarray(out, dtype=float)

# Stage 1
acc_s1 = _arr([0.3825, 0.3321, 0.3774, 0.4662, 0.5568, 0.6117, 0.6949, 0.8934])
conf_s1 = _arr([0.2780, 0.3592, 0.4572, 0.5490, 0.6503, 0.7515, 0.8536, 0.9690])

# Stage 2
acc_s2 = _arr([0.2992, 0.3538, 0.4198, 0.5443, 0.5985, 0.6649, 0.7578, 0.9213])
conf_s2 = _arr([0.2758, 0.3635, 0.4596, 0.5462, 0.6499, 0.7512, 0.8528, 0.9690])

# Stage 3
acc_s3 = _arr([0.2698, 0.3652, 0.4288, 0.5321, 0.5733, 0.6897, 0.7710, 0.9288])
conf_s3 = _arr([0.2809, 0.3593, 0.4554, 0.5498, 0.6507, 0.7495, 0.8522, 0.9682])

# Stage 4
acc_s4 = _arr([0.3241, 0.3341, 0.4272, 0.5422, 0.6008, 0.6854, 0.7867, 0.9340])
conf_s4 = _arr([0.2767, 0.3598, 0.4568, 0.5486, 0.6491, 0.7503, 0.8524, 0.9675])

# Stage 5
acc_s5 = _arr([0.2599, 0.3103, 0.4552, 0.5403, 0.6244, 0.6877, 0.7845, 0.9385])
conf_s5 = _arr([0.2780, 0.3576, 0.4562, 0.5494, 0.6488, 0.7512, 0.8524, 0.9669])

stages = [
    ("Stage 1", acc_s1, conf_s1, ece_stage["Stage 1"]),
    ("Stage 2", acc_s2, conf_s2, ece_stage["Stage 2"]),
    ("Stage 3", acc_s3, conf_s3, ece_stage["Stage 3"]),
    ("Stage 4", acc_s4, conf_s4, ece_stage["Stage 4"]),
    ("Stage 5", acc_s5, conf_s5, ece_stage["Stage 5"]),
]

fig, axes = plt.subplots(1, len(stages), figsize=(24, 5), sharey=True)
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

