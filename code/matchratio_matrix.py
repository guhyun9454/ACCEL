#!/usr/bin/env python3
import os, glob, argparse
from matchratio_pair import compare_pair  # 재사용
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True)   # routes_out/token_only/arc
    ap.add_argument("--outcsv", default=None)  # viz_out/matchratio/arc_matrix.csv
    args=ap.parse_args()

    tokens=["T0","T1","T2"]
    models=sorted(glob.glob(os.path.join(args.root, "*")))
    rows=[]
    print(f"[root] {args.root}")
    for mdir in models:
        m=os.path.basename(mdir)
        exists={t: os.path.exists(os.path.join(mdir,t,"P0.jsonl")) for t in tokens}
        if sum(exists.values())<2: 
            print(f"[SKIP] {m}: need >=2 tokens"); 
            continue
        def p(t): return os.path.join(mdir,t,"P0.jsonl")
        pairs=[("T0","T1"),("T0","T2"),("T1","T2")]
        rec={"model":m}
        for a,b in pairs:
            if exists[a] and exists[b]:
                s=compare_pair(p(a),p(b))
                rec[f"{a}_vs_{b}"]=s["match_ratio"]
            else:
                rec[f"{a}_vs_{b}"]=None
        rows.append(rec)
        print(f"{m}: T0vsT1={rec['T0_vs_T1']}, T0vsT2={rec['T0_vs_T2']}, T1vsT2={rec['T1_vs_T2']}")
    if args.outcsv:
        os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
        import csv
        keys=["model","T0_vs_T1","T0_vs_T2","T1_vs_T2"]
        with open(args.outcsv,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows: w.writerow(r)
        print(f"[saved] {args.outcsv}")
if __name__=="__main__":
    main()
