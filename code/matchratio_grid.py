#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make a matrix heatmap: rows=models, cols=datasets, value=answer match rate
- It reads: <root>/<dataset>/<model>/<T*/P0.jsonl>
- Compare token pairs (e.g., T0 vs T1), or average of all pairs.
- Outputs: a PNG heatmap + a CSV table.

Example:
  python code/matchratio_grid.py \
    --root routes_out/token_only/pride_on \
    --datasets arc csqa \
    --pair T0 T1 \
    --out_png viz_out/answer_match_matrix_on_T0T1.png \
    --out_csv viz_out/answer_match_matrix_on_T0T1.csv
"""

import os, json, argparse, numpy as np
import matplotlib.pyplot as plt

def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out

def load_preds_one(dir_model_token):
    """
    Load <dir_model>/<T*/P0.jsonl> -> dict: qid -> pred
    - Use keys: pred, pred_idx, prediction_idx, prediction (fallbacks)
    """
    f = os.path.join(dir_model_token, "P0.jsonl")
    if not os.path.exists(f):
        return {}
    rows = read_jsonl(f)
    d = {}
    for i, r in enumerate(rows):
        qid = r.get("qid", str(i))
        pred = r.get("pred", None)
        if pred is None:
            for k in ["pred_idx","prediction_idx","prediction"]:
                if k in r:
                    pred = r[k]; break
        if pred is None:
            continue
        try:
            d[qid] = int(pred)
        except Exception:
            # graceful skip
            continue
    return d

def agreement_between(dA, dB):
    """mean( predA == predB ) on intersection qids"""
    keys = sorted(set(dA.keys()) & set(dB.keys()))
    if not keys:
        return np.nan
    same = sum(1 for q in keys if dA[q] == dB[q])
    return same / float(len(keys))

def pairwise_pairs(tokens):
    pairs = []
    for i in range(len(tokens)):
        for j in range(i+1, len(tokens)):
            pairs.append((tokens[i], tokens[j]))
    return pairs

def simple_model_name(raw):
    # heuristics to shorten labels
    s = raw.replace("meta-llama_", "Meta ").replace("google_", "Google ") \
           .replace("Qwen_", "Qwen ").replace("naver-hyperclovax_", "Naver ") \
           .replace("kakaocorp_", "Kakao ").replace("LGAI-EXAONE_", "LG AI ") \
           .replace("K-intelligence_", "KT ").replace("_", " ")
    return s.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="root dir: routes_out/token_only/<pride_on|pride_off>")
    ap.add_argument("--datasets", nargs="+", default=["arc", "csqa"], help="subdirs under root")
    ap.add_argument("--pair", nargs="+", default=["T0","T1"],
                    help='Choose two tokens (e.g., T0 T1). Use special value "ALL" to average over all pairs of T0,T1,T2.')
    ap.add_argument("--tokens", nargs="+", default=["T0","T1","T2"], help="available tokens")
    ap.add_argument("--out_png", default="viz_out/answer_match_matrix.png")
    ap.add_argument("--out_csv", default="viz_out/answer_match_matrix.csv")
    ap.add_argument("--min_models", type=int, default=1, help="skip dataset if <min_models models found")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # detect models by union across datasets
    model_set = set()
    per_ds_models = {}
    for ds in args.datasets:
        dsd = os.path.join(args.root, ds)
        if not os.path.isdir(dsd):
            continue
        models = [m for m in os.listdir(dsd) if os.path.isdir(os.path.join(dsd, m))]
        per_ds_models[ds] = sorted(models)
        model_set.update(models)

    models = sorted(model_set)
    if not models:
        raise SystemExit(f"No models found under {args.root}/<dataset>/*")

    # choose token pairs
    if len(args.pair) == 1 and args.pair[0].upper() == "ALL":
        pairs = pairwise_pairs(args.tokens)
        pair_name = "mean(T-pairs)"
    else:
        if len(args.pair) != 2:
            raise SystemExit("--pair must be two tokens (e.g., T0 T1) or 'ALL'")
        pairs = [tuple(args.pair)]
        pair_name = f"{args.pair[0]} vs {args.pair[1]}"

    # build matrix
    M = np.full((len(models), len(args.datasets)), np.nan, dtype=float)

    for j, ds in enumerate(args.datasets):
        dsd = os.path.join(args.root, ds)
        if not os.path.isdir(dsd):
            continue
        mods = [m for m in os.listdir(dsd) if os.path.isdir(os.path.join(dsd, m))]
        if len(mods) < args.min_models:  # optional guard
            continue
        for i, model in enumerate(models):
            dir_model = os.path.join(dsd, model)
            if not os.path.isdir(dir_model):
                continue
            # load predictions for all tokens needed
            cache = {}
            for t in set(sum(([p[0], p[1]] for p in pairs), [])):
                cache[t] = load_preds_one(os.path.join(dir_model, t))
            # aggregate over pairs
            vals = []
            for a,b in pairs:
                if a not in cache or b not in cache:
                    continue
                vals.append(agreement_between(cache[a], cache[b]))
            if len(vals) > 0:
                M[i, j] = float(np.nanmean(vals))

    # sort rows by mean across available datasets (descending)
    row_means = np.nanmean(M, axis=1)
    order = np.argsort(-row_means)
    M = M[order, :]
    models_sorted = [models[idx] for idx in order]

    # save CSV
    with open(args.out_csv, "w", encoding="utf-8") as w:
        w.write("model," + ",".join(args.datasets) + "\n")
        for i, m in enumerate(models_sorted):
            row = [simple_model_name(m)] + [
                ("" if np.isnan(M[i, j]) else f"{M[i,j]:.4f}") for j in range(M.shape[1])
            ]
            w.write(",".join(row) + "\n")

    # plot heatmap
    plt.figure(figsize=(8, max(3, 0.35*len(models_sorted)+1)))
    plt.imshow(M, vmin=0.0, vmax=1.0, aspect="auto")
    plt.title("Answer Match Rate", fontsize=16, pad=10)
    plt.xticks(ticks=range(len(args.datasets)), labels=[ds.upper() for ds in args.datasets])
    ylabels = [simple_model_name(m) for m in models_sorted]
    plt.yticks(ticks=range(len(models_sorted)), labels=ylabels)
    plt.colorbar()
    # annotate
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                plt.text(j, i, f"{100*M[i,j]:.1f}%", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=180)
    plt.close()

    print(f"[DONE] PNG: {args.out_png}")
    print(f"[DONE] CSV: {args.out_csv}")
    print(f"[INFO] Pair used: {pair_name}")
    print(f"[INFO] Root: {args.root}")
