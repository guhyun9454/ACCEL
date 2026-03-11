#!/usr/bin/env python3
import argparse
import os
import platform

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def _set_korean_font():
    """
    OS별 한글 폰트 자동 설정.
    - 폰트가 없으면 기본 폰트로 fallback (경고 없이 진행)
    """
    try:
        sys_name = platform.system()
        if sys_name == "Windows":
            plt.rc("font", family="Malgun Gothic")
        elif sys_name == "Darwin":
            plt.rc("font", family="AppleGothic")
        else:
            # Linux: 다양한 후보 중 존재하는 걸 사용
            try:
                from matplotlib import font_manager

                candidates = [
                    "NanumGothic",
                    "Noto Sans CJK KR",
                    "Noto Sans KR",
                    "AppleGothic",
                    "Malgun Gothic",
                ]
                available = {f.name for f in font_manager.fontManager.ttflist}
                for c in candidates:
                    if c in available:
                        plt.rc("font", family=c)
                        break
            except Exception:
                pass
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def draw_state_column(ax, x_pos, title, blocks):
    """
    Draw one column of stacked blocks at x_pos.
    """
    # 컬럼 제목
    ax.text(
        x_pos + 0.4,
        8.5,
        title,
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

    total_val = sum([float(b["h"]) for b in blocks]) if blocks else 1.0
    y_curr = 2.0

    # 바닥에서부터 위로 쌓기 위해 역순으로 그림
    for block in reversed(blocks):
        h_norm = (float(block["h"]) / total_val) * 6.0

        rect = patches.Rectangle(
            (x_pos, y_curr),
            0.8,
            h_norm,
            linewidth=2,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)

        text_y = y_curr + (h_norm / 2)
        ax.text(x_pos + 1.0, text_y, str(block["text"]), va="center", fontsize=12)

        y_curr += h_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="state_transition_diagram.png", help="Output PNG path")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--width", type=float, default=14.0)
    ap.add_argument("--height", type=float, default=6.0)
    args = ap.parse_args()

    _set_korean_font()

    fig, ax = plt.subplots(figsize=(float(args.width), float(args.height)))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    col1_data = [
        {"h": 1.0, "text": "(0)"},
        {"h": 1.0, "text": "(1)"},
    ]

    col2_data = [
        {"h": 2.0, "text": "(0, 0)"},
        {"h": 0.8, "text": "(0, 1)"},
        {"h": 1.2, "text": "(1, 0)"},
        {"h": 2.0, "text": "(1, 1)"},
    ]

    col3_data = [
        {"h": 1.5, "text": "(0, 0, 0)"},
        {"h": 0.5, "text": "(0, 0, 1)"},
        {"h": 0.5, "text": "(0, 1, 0)"},
        {"h": 1.0, "text": "(0, 1, 1)"},
        {"h": 0.8, "text": "(1, 0, 0)"},
        {"h": 0.5, "text": "(1, 0, 1)"},
        {"h": 0.5, "text": "(1, 1, 0)"},
        {"h": 1.5, "text": "(1, 1, 1)"},
    ]

    draw_state_column(ax, 0.5, "Initial\n(Default)", col1_data)
    draw_state_column(ax, 3.5, "Only\nFlip (Cost=2)", col2_data)
    draw_state_column(ax, 7.5, "Cyclic\n(Cost=4)", col3_data)

    # Legend / note
    ax.text(
        9.2,
        1.1,
        "0 = incorrect\n1 = correct",
        ha="left",
        va="bottom",
        fontsize=12,
    )

    plt.tight_layout()

    out_path = str(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()