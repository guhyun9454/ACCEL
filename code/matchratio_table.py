#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, csv
import matplotlib.pyplot as plt
import numpy as np

def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            rows.append(r)
    header, data = rows[0], rows[1:]
    return header, data

def save_markdown(header, data, out_md):
    with open(out_md, "w", encoding="utf-8") as w:
        w.write("| " + " | ".join(header) + " |\n")
        w.write("| " + " | ".join(["---"] * len(header)) + " |\n")
        for r in data:
            w.write("| " + " | ".join(r) + " |\n")

def save_latex(header, data, out_tex):
    with open(out_tex, "w", encoding="utf-8") as w:
        cols = "l" + "c"*(len(header)-1)
        w.write("\\begin{tabular}{" + cols + "}\n\\toprule\n")
        w.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for r in data:
            w.write(" & ".join(r) + " \\\\\n")
        w.write("\\bottomrule\n\\end{tabular}\n")

def save_table_png(header, data, out_png, title="Answer Match Rate (%)"):
    fig, ax = plt.subplots(figsize=(len(header)*1.2, max(2.5, 0.45*len(data)+1)))
    ax.axis("off")
    # 값 퍼센트 포맷
    fmt = []
    for r in data:
        row = [r[0]]
        for v in r[1:]:
            row.append(("" if v=="" else f"{100*float(v):.1f}%"))
        fmt.append(row)
    table = ax.table(cellText=fmt, colLabels=header, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    ax.set_title(title, pad=10, fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_md", default="viz_out/match_table.md")
    ap.add_argument("--out_tex", default="viz_out/match_table.tex")
    ap.add_argument("--out_png", default="viz_out/match_table.png")
    ap.add_argument("--title", default="Answer Match Rate (%)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    header, data = load_csv(args.csv)
    save_markdown(header, data, args.out_md)
    save_latex(header, data, args.out_tex)
    save_table_png(header, data, args.out_png, args.title)
    print(f"[DONE] MD:  {args.out_md}")
    print(f"[DONE] TEX: {args.out_tex}")
    print(f"[DONE] PNG: {args.out_png}")

if __name__ == "__main__":
    main()
