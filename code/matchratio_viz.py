
#!/usr/bin/env python3
"""
matchratio_viz.py

Load per-variant JSONLs (one per token/cyclic combination) and compute:
- Accuracy per variant
- Pairwise matching ratio (MR): P( pred_i == pred_j )
- Token-only vs Cyclic-only diversity summaries
- Simple ensembles (arithmetic/geometric mean across variants)
- (Optional) temperature calibration on a heldout split
- Visualizations (heatmaps saved as PNGs)

USAGE
-----
$ python matchratio_viz.py --root outputs_variants
  [--model "Qwen/Qwen2.5-1.5B-Instruct"] [--eval_name arc]
  [--outdir /path/to/save]
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------
# Utility
# ----------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def imsave_heatmap(matrix, labels, title, out_png):
    fig = plt.figure()  # one plot per figure
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, interpolation='nearest')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def poisson_binomial_majority(ps):
    """Probability that majority is correct among len(ps) independent voters,
    where each predicts the correct class with probability p_i.
    Uses Poisson binomial via DP."""
    n = len(ps)
    # DP over probabilities of k successes
    dp = [0.0] * (n + 1)
    dp[0] = 1.0
    for p in ps:
        ndp = [0.0]*(n+1)
        for k in range(n+1):
            # not correct
            ndp[k] += dp[k] * (1.0 - p)
            # correct
            if k+1 <= n:
                ndp[k+1] += dp[k] * p
        dp = ndp
    need = (n // 2) + 1   # majority threshold
    return sum(dp[need:])

def ece(probs, correct, n_bins=10):
    """Expected Calibration Error for multiclass top-1 confidence.
    probs: shape (N, C), correct: shape (N,) bool"""
    top_conf = probs.max(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece_val = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b+1]
        mask = (top_conf > lo) & (top_conf <= hi) if b>0 else (top_conf >= lo) & (top_conf <= hi)
        if mask.sum()==0: 
            continue
        acc = correct[mask].mean()
        conf = top_conf[mask].mean()
        ece_val += (mask.mean()) * abs(acc - conf)
    return float(ece_val)

def temperature_scale(logits, labels, max_iter=50, lr=0.1):
    """Simple temperature scaling on logits (N,C) w.r.t. labels (N,) [0..C-1]."""
    N, C = logits.shape
    T = 1.0
    for _ in range(max_iter):
        # gradient of NLL w.r.t T (approx, use central diff on small epsilon)
        eps = 1e-4
        def nll(temp):
            z = logits / temp
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
            ll = -np.log(p[np.arange(N), labels] + 1e-12).mean()
            return ll
        f0 = nll(T)
        f1 = nll(T+eps)
        grad = (f1 - f0)/eps
        T_new = max(0.05, T - lr*grad)  # keep it positive and not too tiny
        if abs(T_new - T) < 1e-6:
            break
        T = T_new
    return T

# ----------------------
# Loading & Aggregation
# ----------------------

def load_runs(root, model=None, eval_name=None):
    """
    root/
      <model_tag>/
        T0P0.jsonl, T0P1.jsonl, ..., T2P3.jsonl
    Returns: dict[(model_tag, eval_name)][variation_id] -> list of records
    """
    store = defaultdict(lambda: defaultdict(list))
    for mdir in glob.glob(os.path.join(root, "*")):
        if not os.path.isdir(mdir):
            continue
        model_tag = os.path.basename(mdir)
        if model is not None and model_tag != model.replace("/", "_"):
            continue
        for jf in glob.glob(os.path.join(mdir, "*.jsonl")):
            variation_id = os.path.splitext(os.path.basename(jf))[0]
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    e = rec.get("eval_name", "eval")
                    if eval_name is not None and e != eval_name:
                        continue
                    store[(model_tag, e)][variation_id].append(rec)
    return store

def compute_metrics(group):
    """
    group: dict[variation_id] -> list[rec]
    Returns:
      acc: dict[variation_id] -> accuracy
      mr: 2D array (V x V) pairwise matching ratios
      order: list[variation_id] in same order as mr axes
    """
    order = sorted(group.keys())  # stable
    # Align questions across variants by qid
    # Build map: var_id -> qid -> (pred, gold, probs)
    aligned = {}
    all_qids = set()
    for vid in order:
        m = {}
        for r in group[vid]:
            m[r["qid"]] = (r["pred"], r["gold"], r.get("probs", None))
        aligned[vid] = m
        all_qids |= set(m.keys())
    # keep only qids present in every variant
    common_qids = [q for q in all_qids if all(q in aligned[vid] for vid in order)]
    if not common_qids:
        return {}, np.zeros((len(order), len(order))), order, pd.DataFrame()

    # Accuracy per variant
    acc = {}
    preds = {}
    golds = None
    probs = {}
    for vid in order:
        p = []
        c = 0
        P = []
        for q in common_qids:
            pred, gold, pr = aligned[vid][q]
            p.append(pred)
            c += 1 if pred == gold else 0
            if pr is not None:
                P.append(pr)
        acc[vid] = c / len(common_qids)
        preds[vid] = np.array(p, dtype=int)
        if P:
            probs[vid] = np.array(P, dtype=float)
        if golds is None:
            golds = np.array([aligned[order[0]][q][1] for q in common_qids], dtype=int)

    # Matching ratio matrix
    V = len(order)
    mr = np.zeros((V, V), dtype=float)
    for i, vi in enumerate(order):
        for j, vj in enumerate(order):
            mr[i, j] = (preds[vi] == preds[vj]).mean()

    # Also pack a per-variant DataFrame
    rows = []
    for vid in order:
        rows.append({"variation_id": vid, "accuracy": acc[vid], "n": len(common_qids)})
    df = pd.DataFrame(rows).sort_values("variation_id").reset_index(drop=True)

    return acc, mr, order, df

def ensemble_probs(variant_probs):
    """Given dict[vid] -> (N,C) probs aligned on same questions,
       return simple arithmetic mean ensemble (N,C)."""
    mats = [variant_probs[vid] for vid in sorted(variant_probs.keys())]
    P = np.stack(mats, axis=0).mean(axis=0)
    # re-normalize (should already be normalized)
    P = np.clip(P, 1e-12, None)
    P /= P.sum(axis=1, keepdims=True)
    return P

def geometric_ensemble_probs(variant_probs):
    mats = [np.clip(variant_probs[vid], 1e-12, None) for vid in sorted(variant_probs.keys())]
    logsum = np.sum(np.log(mats), axis=0) / len(mats)
    P = np.exp(logsum)
    P /= P.sum(axis=1, keepdims=True)
    return P

# ----------------------
# Main
# ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs_variants")
    ap.add_argument("--model", default=None, help="Exact model_tag (slashes replaced by underscores) or None for first available")
    ap.add_argument("--eval_name", default=None, help="e.g., arc or csqa")
    ap.add_argument("--outdir", default="viz_out")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    store = load_runs(args.root, model=args.model, eval_name=args.eval_name)
    if not store:
        print("[WARN] No runs found. Did you already run variants_runner.py?")
        return

    for (model_tag, eval_name), group in store.items():
        print(f"\n== Model: {model_tag} | Eval: {eval_name} ==")
        acc, mr, order, df = compute_metrics(group)
        if not acc:
            print("No common qids or empty stats.")
            continue

        # Save table
        tsv_path = os.path.join(args.outdir, f"{model_tag}_{eval_name}_variant_acc.tsv")
        df.to_csv(tsv_path, sep="\t", index=False)
        print("Saved:", tsv_path)

        # Heatmap of Matching Ratio
        png_mr = os.path.join(args.outdir, f"{model_tag}_{eval_name}_mr.png")
        imsave_heatmap(mr, order, f"Matching Ratio — {model_tag} / {eval_name}", png_mr)
        print("Saved:", png_mr)

        # Build token-only and cyclic-only summaries
        # Order is like T0P0, T0P1, ..., T2P3
        token_sets = sorted({vid[:2] for vid in order})  # T0,T1,T2
        shifts = sorted({int(vid[3:]) for vid in order})
        # Average MR across same-token different-shift (cyclic diversity within token-set)
        rows = []
        for t in token_sets:
            vids = [v for v in order if v.startswith(t)]
            # pairwise MR among these
            idx = [order.index(v) for v in vids]
            sub = mr[np.ix_(idx, idx)]
            mrr = (np.sum(sub) - np.trace(sub)) / (sub.size - len(idx))
            rows.append({"token_set": t, "avg_MR_within_token": mrr})
        df_cyc = pd.DataFrame(rows)
        cyc_tsv = os.path.join(args.outdir, f"{model_tag}_{eval_name}_cyclic_within_token.tsv")
        df_cyc.to_csv(cyc_tsv, sep="\t", index=False)
        print("Saved:", cyc_tsv)

        # Average MR across different-token but same-shift (token diversity for each shift)
        rows = []
        for p in shifts:
            vids = [v for v in order if v.endswith(str(p))]
            idx = [order.index(v) for v in vids]
            sub = mr[np.ix_(idx, idx)]
            mrr = (np.sum(sub) - np.trace(sub)) / (sub.size - len(idx))
            rows.append({"permute_shift": p, "avg_MR_across_tokens": mrr})
        df_tok = pd.DataFrame(rows)
        tok_tsv = os.path.join(args.outdir, f"{model_tag}_{eval_name}_token_across_shifts.tsv")
        df_tok.to_csv(tok_tsv, sep="\t", index=False)
        print("Saved:", tok_tsv)

        # Simple ensembles
        # Collect probs on common qids
        # (re-run compute_metrics to retain probs)
        # We'll rebuild aligned probs here:
        # ---
        # Align again
        aligned = {}
        all_qids = set()
        for vid in order:
            m = {}
            for r in group[vid]:
                m[r["qid"]] = (r["pred"], r["gold"], r.get("probs", None))
            aligned[vid] = m
            all_qids |= set(m.keys())
        common_qids = [q for q in all_qids if all(q in aligned[vid] for vid in order)]
        variant_probs = {}
        golds = np.array([aligned[order[0]][q][1] for q in common_qids], dtype=int)
        for vid in order:
            P = [aligned[vid][q][2] for q in common_qids]
            if any(p is None for p in P):
                variant_probs = {}
                break
            variant_probs[vid] = np.array(P, dtype=float)

        if variant_probs:
            P_avg = ensemble_probs(variant_probs)
            P_geo = geometric_ensemble_probs(variant_probs)

            ens_acc_avg = (P_avg.argmax(axis=1) == golds).mean()
            ens_acc_geo = (P_geo.argmax(axis=1) == golds).mean()

            # ECE
            ece_avg = ece(P_avg, P_avg.argmax(axis=1) == golds)
            ece_geo = ece(P_geo, P_geo.argmax(axis=1) == golds)

            with open(os.path.join(args.outdir, f"{model_tag}_{eval_name}_ensemble.txt"), "w") as f:
                f.write(f"Arithmetic-mean ensemble accuracy: {ens_acc_avg:.4f}\n")
                f.write(f"Geometric-mean ensemble accuracy : {ens_acc_geo:.4f}\n")
                f.write(f"ECE (arith): {ece_avg:.4f}\n")
                f.write(f"ECE (geom) : {ece_geo:.4f}\n")

            # Heatmap for pairwise MR is already saved. Nothing more to plot per the no-subplots rule.
            print("Saved:", os.path.join(args.outdir, f"{model_tag}_{eval_name}_ensemble.txt"))
        else:
            print("[Note] 'probs' missing in JSONLs; ensembles/ECE skipped. Make sure eval_clm.py saves 'probs'.")

if __name__ == "__main__":
    main()
