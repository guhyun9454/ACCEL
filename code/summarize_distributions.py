#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, argparse, statistics as stats
from collections import defaultdict

def load_rows(path_json):
    return json.load(open(path_json, "r", encoding="utf-8"))

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="out")
    ap.add_argument("--summary_json", type=str, default=None)
    ap.add_argument("--make_plots", action="store_true")
    args = ap.parse_args()

    summary_json = args.summary_json or os.path.join(args.root, "report", "compare_summary_enriched.json")
    rows = load_rows(summary_json)

    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    report_dir = ensure_dir(os.path.join(args.root, "report"))
    table_tsv  = os.path.join(report_dir, "model_level_summary.tsv")

    import csv
    with open(table_tsv, "w", encoding="utf-8", newline="") as tf:
        w = csv.writer(tf, delimiter="\t")
        w.writerow(["model","n_pairs",
                    "i2c_rate_mean","i2c_rate_std",
                    "c2i_rate_mean","c2i_rate_std",
                    "acc_from_mean","acc_from_std",
                    "acc_to_mean","acc_to_std",
                    "acc_delta_mean","acc_delta_std",
                    "matching_ratio_mean","matching_ratio_std"])
        for m, arr in sorted(by_model.items()):
            def safe_mean(xs): 
                return sum(xs)/len(xs) if xs else float('nan')
            def safe_std(xs):
                return stats.pstdev(xs) if len(xs)>=1 else float('nan')
            i2c_rate = [x["i2c_rate"] for x in arr]
            c2i_rate = [x["c2i_rate"] for x in arr]
            acc_from = [x["acc_from"] for x in arr]
            acc_to   = [x["acc_to"]   for x in arr]
            acc_delta= [x["acc_delta"]for x in arr]
            mr       = [x["matching_ratio"] for x in arr]

            w.writerow([m, len(arr),
                        safe_mean(i2c_rate), safe_std(i2c_rate),
                        safe_mean(c2i_rate), safe_std(c2i_rate),
                        safe_mean(acc_from), safe_std(acc_from),
                        safe_mean(acc_to),   safe_std(acc_to),
                        safe_mean(acc_delta),safe_std(acc_delta),
                        safe_mean(mr),       safe_std(mr)])

    print(f"[OK] Wrote per-model table: {table_tsv}")

    if args.make_plots:
        import matplotlib.pyplot as plt
        for m, arr in sorted(by_model.items()):
            vals = {
                "i2c_rate": [x["i2c_rate"] for x in arr],
                "c2i_rate": [x["c2i_rate"] for x in arr],
                "acc_delta": [x["acc_delta"] for x in arr],
                "matching_ratio": [x["matching_ratio"] for x in arr],
            }
            for key, xs in vals.items():
                if not xs: 
                    continue
                plt.figure()
                plt.hist(xs, bins=30)
                plt.title(f"{m} — {key} (n={len(xs)})")
                plt.xlabel(key)
                plt.ylabel("count")
                out_png = os.path.join(report_dir, f"{m}_{key}_hist.png")
                plt.savefig(out_png, bbox_inches="tight")
                plt.close()
                print(f"[OK] Saved {out_png}")

if __name__ == "__main__":
    main()

