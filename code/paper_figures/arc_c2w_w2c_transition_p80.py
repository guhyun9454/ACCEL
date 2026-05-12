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

# ---------------------------------------------------------
# Font and Style Settings
# ---------------------------------------------------------
plt.rcParams["font.family"] = "Pretendard"
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["text.color"] = "black"

# Color Palette
COLOR_W2C = "#3478BA"
COLOR_C2W = "#E3A220"

# ---------------------------------------------------------
# ARC p=80 settings (from arc_paper_extract (1).md)
# ---------------------------------------------------------
N_TOTAL = 1165
P = 80
low_ratio = 0.8000

transitions = ["1->2", "2->3", "3->4"]
w2c_rates = np.array([0.0575, 0.0263, 0.0257], dtype=float)
c2w_rates = np.array([0.0375, 0.0271, 0.0201], dtype=float)

N_LOW = int(np.round(low_ratio * N_TOTAL))
w2c_counts = np.round(w2c_rates * N_LOW).astype(int)
c2w_counts = np.round(c2w_rates * N_LOW).astype(int)

print(f"[ARC p={P}] N_TOTAL={N_TOTAL}, low_ratio={low_ratio}, N_LOW≈{N_LOW}")
print("W2C step counts:", dict(zip(transitions, w2c_counts.tolist())))
print("C2W step counts:", dict(zip(transitions, c2w_counts.tolist())))

# ---------------------------------------------------------
# Mountain plot
# ---------------------------------------------------------
n = len(transitions)
xs_left = list(range(-(n - 1), 1))
xs_right = list(range(0, n))

y_c2w = c2w_counts[::-1].tolist()
y_w2c = w2c_counts.tolist()

fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(xs_left, y_c2w, color=COLOR_C2W, marker="o", markersize=9, linewidth=2.5, label="Correct -> Wrong (C2W)")
ax.plot(xs_right, y_w2c, color=COLOR_W2C, marker="o", markersize=9, linewidth=2.5, label="Wrong -> Correct (W2C)")

ax.fill_between(xs_left, 0, y_c2w, color=COLOR_C2W, alpha=0.3)
ax.fill_between(xs_right, 0, y_w2c, color=COLOR_W2C, alpha=0.3)

ax.axvline(x=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.5)

ax.text(-1, -0.10, "C2W", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=18, color="black")
ax.text(1, -0.10, "W2C", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=18, color="black")

def _annotate_counts(xs, ys):
    for x, y in zip(xs, ys):
        ax.annotate(
            f"{int(y):,}",
            xy=(x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
            color="black",
            clip_on=False,
        )

_annotate_counts(xs_left, y_c2w)
_annotate_counts(xs_right, y_w2c)

xticks = xs_left[:-1] + xs_right
xticklabels = transitions[::-1][:-1] + transitions
ax.set_xticks(xticks)
ax.set_xticklabels([f"Stage {t}" for t in xticklabels])
ax.set_xlabel("Transition", labelpad=55, fontsize=20)

formatter = ticker.FuncFormatter(lambda y, pos: f"{int(y):,}")
ax.yaxis.set_major_formatter(formatter)
ax.set_ylabel("Sample Count", labelpad=15, fontsize=20)

ax.set_xlim(-(n - 1), (n - 1))
_maxy = max(int(np.max(w2c_counts)), int(np.max(c2w_counts)))
ax.set_ylim(0, _maxy * 1.60 + 12)
ax.legend(loc="upper right", fontsize=18, markerscale=1.6, framealpha=0.9, edgecolor="gray")
ax.grid(True, axis="y", linestyle=":", alpha=0.6)

plt.tight_layout(rect=(0, 0, 1, 0.98))
plt.show()

