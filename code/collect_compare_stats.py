#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse, json, glob, csv

def find_overall_files(root):
    return glob.glob(os.path.join(root, "*", "compare", "*_to_*", "__overall.json"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="out")
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--out_tsv", type=str, default=None)
    args = ap.parse_args()

    files = find_overall_files(args.root)
    rows = []
    for f in sorted(files):
        try:
            data = json.load(open(f, "r", encoding="utf-8"))
            parts = f.split(os.sep)
            model = parts[-4] if len(parts) >= 4 else "unknown"
            total = int(data.get("total") or 0)
            i2c = int(data.get("i2c") or 0)
            c2i = int(data.get("c2i") or 0)
            rows.append({
                "model": model,
                "id_from": data.get("id_from"),
                "id_to": data.get("id_to"),
                "matching_ratio": data.get("matching_ratio"),
                "matches": data.get("matches"),
                "total": total,
                "i2c": i2c,
                "c2i": c2i,
                "both_correct": data.get("both_correct"),
                "both_incorrect": data.get("both_incorrect"),
                "i2c_rate": (i2c/total) if total else 0.0,
                "c2i_rate": (c2i/total) if total else 0.0,
            })
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")

    out_json = args.out_json or os.path.join(args.root, "compare_summary.json")
    out_tsv  = args.out_tsv  or os.path.join(args.root, "compare_summary.tsv")

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(rows, jf, indent=2, ensure_ascii=False)

    with open(out_tsv, "w", encoding="utf-8", newline="") as tf:
        w = csv.writer(tf, delimiter="\t")
        w.writerow(["model","id_from","id_to","matching_ratio","matches","total",
                    "i2c","c2i","both_correct","both_incorrect","i2c_rate","c2i_rate"])
        for r in rows:
            w.writerow([r.get(k,"") for k in ["model","id_from","id_to","matching_ratio","matches","total",
                                              "i2c","c2i","both_correct","both_incorrect","i2c_rate","c2i_rate"]])

    print(f"[OK] Wrote {out_json} and {out_tsv} ({len(rows)} pairs).")

if __name__ == "__main__":
    main()
