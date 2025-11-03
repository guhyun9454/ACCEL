#!/usr/bin/env python3
import os, json, argparse, glob
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt

def read_jsonl(p):
    D={}
    with open(p,'r',encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            q=r.get("qid")
            if q is None:
                # fallback: 순번으로라도 고유화
                q=f"@{len(D)}"
            D[q]=r
    return D

def compare_pair(p1, p2):
    A=read_jsonl(p1); B=read_jsonl(p2)
    keys=sorted(set(A.keys()) & set(B.keys()))
    n=len(keys)
    same=0; both_correct=0; both_wrong=0; a_only=0; b_only=0
    for q in keys:
        ra, rb = A[q], B[q]
        pa, pb = ra.get("pred",-1), rb.get("pred",-1)
        ga, gb = ra.get("gold",-1), rb.get("gold",-1)
        if pa==pb: same+=1
        ca = (pa==ga)
        cb = (pb==gb)
        if ca and cb: both_correct+=1
        elif (not ca) and (not cb): both_wrong+=1
        elif ca and (not cb): a_only+=1
        elif (not ca) and cb: b_only+=1
    return {
        "N": n,
        "match_ratio": same/max(n,1),
        "both_correct": both_correct/max(n,1),
        "both_wrong": both_wrong/max(n,1),
        "T1_only_correct": a_only/max(n,1),
        "T2_only_correct": b_only/max(n,1),
    }

def bar_plot(stats, title, outpng):
    labels = ["match","both_correct","both_wrong","T1_only","T2_only"]
    vals = [stats["match_ratio"], stats["both_correct"], stats["both_wrong"], stats["T1_only_correct"], stats["T2_only_correct"]]
    plt.figure(figsize=(6,3.5))
    plt.bar(labels, vals)
    plt.ylim(0,1)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpng), exist_ok=True)
    plt.savefig(outpng, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="routes_out/token_only/<eval_tag>")
    ap.add_argument("--tokenA", default="T1", choices=["T0","T1","T2"])
    ap.add_argument("--tokenB", default="T2", choices=["T0","T1","T2"])
    ap.add_argument("--outdir", default="viz_out/matchratio")
    args = ap.parse_args()

    eval_tag = os.path.basename(args.root.rstrip("/"))
    models = sorted(glob.glob(os.path.join(args.root, "*")))
    rows=[]
    for mdir in models:
        model_tag=os.path.basename(mdir)
        pA=os.path.join(mdir, args.tokenA, "P0.jsonl")
        pB=os.path.join(mdir, args.tokenB, "P0.jsonl")
        if not (os.path.exists(pA) and os.path.exists(pB)):
            print(f"[SKIP] {model_tag} missing {pA} or {pB}")
            continue
        s=compare_pair(pA,pB)
        rows.append((model_tag, s))
        print(f"[{eval_tag}] {model_tag}: N={s['N']}  match={s['match_ratio']:.3f}  both_corr={s['both_correct']:.3f}  T1_only={s['T1_only_correct']:.3f}  T2_only={s['T2_only_correct']:.3f}")
        outpng=os.path.join(args.outdir, eval_tag, f"{model_tag}_{args.tokenA}_vs_{args.tokenB}.png")
        bar_plot(s, f"{model_tag}: {args.tokenA} vs {args.tokenB}", outpng)

    # 전체 평균도 한 줄
    if rows:
        N=sum(r[1]["N"] for r in rows)
        def wavg(key): return sum(r[1][key]*r[1]["N"] for r in rows)/max(N,1)
        print(f"\n== Weighted Avg over models (N={N}) ==")
        for k in ["match_ratio","both_correct","both_wrong","T1_only_correct","T2_only_correct"]:
            print(f"{k}: {wavg(k):.3f}")

if __name__ == "__main__":
    main()
