#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, argparse, subprocess
from collections import defaultdict, Counter
import numpy as np

# 토큰 세트 정의 (위 파일과 동일)
TOKSETS_EN4 = {"T0":"A,B,C,D","T1":"a,b,c,d","T2":"1,2,3,4"}
TOKSETS_EN5 = {"T0":"A,B,C,D,E","T1":"a,b,c,d,e","T2":"1,2,3,4,5"}
TOKSETS_KO4 = {"T0":"A,B,C,D","T1":"가,나,다,라","T2":"1,2,3,4"}
TOKSETS_KO5 = {"T0":"A,B,C,D,E","T1":"가,나,다,라,마","T2":"1,2,3,4,5"}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--eval_name", required=True)
    ap.add_argument("--num_shifts", type=int, default=4)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.80)
    ap.add_argument("--root_out", default="routes_out/pride_off")
    ap.add_argument("--extra", default="")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--ko", action="store_true")
    return ap.parse_args()

def run_eval(model, eval_name, data_root, out_jsonl, tok4, tok5, shift, ko, extra):
    cmd = [
        "python", "code/eval_clm.py",
        "--pretrained_model_path", model,
        "--data_root", data_root,
        "--eval_names", eval_name,
        "--option_ids4", tok4,
        "--option_ids5", tok5,
        "--save_preds", out_jsonl,
        "--permute_shift", str(shift),
    ]
    if ko: cmd.append("--ko")
    if extra.strip(): cmd += extra.strip().split()
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    print("[CMD]", " ".join(cmd))
    return subprocess.call(cmd)

# ===== 아래 merge/save 유틸은 with_pride.py와 동일 =====
def _read_jsonl(p):
    out=[]; 
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            try: out.append(json.loads(line))
            except: pass
    return out
def _pull(r, ks, d=None):
    for k in ks:
        if k in r and r[k] is not None: return r[k]
    return d
def _norm(p):
    import numpy as np
    p=np.asarray(p,dtype=float); p=np.clip(p,1e-12,None); s=p.sum()
    return (p/s) if s>0 else np.ones_like(p)/len(p)

from collections import defaultdict, Counter
import numpy as np

def merge_prob_files(paths):
    bag_probs=defaultdict(list); bag_votes=defaultdict(list); gold={}
    for p in paths:
        for r in _read_jsonl(p):
            q=_pull(r,["qid","id"])
            if q is None: continue
            g=_pull(r,["gold_idx","label_idx","label","gold","answer"],-1)
            try: g=int(g)
            except: g=-1
            gold[q]=g
            pr=_pull(r,["probs"],None)
            if isinstance(pr,(list,tuple)) and len(pr)>0:
                bag_probs[q].append(_norm(pr))
            else:
                pi=_pull(r,["pred_idx","prediction_idx","pred","prediction"],-1)
                try: pi=int(pi)
                except: pi=-1
                if pi>=0: bag_votes[q].append(pi)
    merged={}
    for q in set(list(bag_probs.keys())+list(bag_votes.keys())+list(gold.keys())):
        G=gold.get(q,-1)
        if bag_probs[q]:
            import numpy as np
            P=np.mean(bag_probs[q],axis=0); P=_norm(P)
            pred=int(np.argmax(P)); conf=float(P[pred])
            merged[q]=dict(gold_idx=G,pred_idx=pred,probs=P.tolist(),conf=conf)
        else:
            vs=bag_votes[q]
            if vs:
                c=Counter(vs); pred, _ = max(c.items(), key=lambda x:(x[1],-x[0]))
            else:
                pred=-1
            merged[q]=dict(gold_idx=G,pred_idx=pred,probs=None,conf=None)
    qids=sorted(merged.keys())
    corr=[1 if (merged[q]["pred_idx"]==merged[q]["gold_idx"] and merged[q]["pred_idx"]>=0) else 0 for q in qids]
    confs=[merged[q]["conf"] for q in qids if merged[q]["conf"] is not None]
    mean_conf=float(np.mean(confs)) if confs else 0.0
    acc=float(np.mean(corr)) if corr else 0.0
    return merged, mean_conf, acc

def save_merged(merged, out_jsonl):
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl,"w",encoding="utf-8") as w:
        for q in sorted(merged.keys()):
            w.write(json.dumps(dict(qid=q, **merged[q]), ensure_ascii=False)+"\n")
    print(f"[SAVE] {out_jsonl} ({len(merged)} rows)")

def main():
    args = parse_args()
    task = args.eval_name.split(",")[0]
    tok4_map = TOKSETS_KO4 if args.ko else TOKSETS_EN4
    tok5_map = TOKSETS_KO5 if args.ko else TOKSETS_EN5
    models = args.models or ["K-intelligence/Midm-2.0-Mini-Instruct"]

    for model in models:
        model_tag = model.split("/")[-1]
        eval_tag = args.eval_name.replace(",", "_")
        sdir = os.path.join(args.root_out, task, model_tag, eval_tag)
        os.makedirs(sdir, exist_ok=True)

        outs1=[]
        jf=os.path.join(sdir,"T0P0.jsonl")
        if run_eval(model,args.eval_name,args.data_root,jf,tok4_map["T0"],tok5_map["T0"],0,args.ko,args.extra)==0:
            outs1.append(jf)
        merged, mc, acc = merge_prob_files(outs1)
        save_merged(merged, os.path.join(sdir,"stage1_merged.jsonl"))
        print(f"[{model_tag}][{eval_tag}] Stage1 mean_conf={mc:.3f}, acc={acc:.3f}")
        if mc>=args.lam:
            print(f"  -> confidence {mc:.3f} >= λ {args.lam:.2f}, skip later stages.")
            continue

        outs2=outs1[:]
        for t in ["T1","T2"]:
            jf=os.path.join(sdir,f"{t}P0.jsonl")
            if run_eval(model,args.eval_name,args.data_root,jf,tok4_map[t],tok5_map[t],0,args.ko,args.extra)==0:
                outs2.append(jf)
        merged, mc, acc = merge_prob_files(outs2)
        save_merged(merged, os.path.join(sdir,"stage2_merged.jsonl"))
        print(f"[{model_tag}][{eval_tag}] Stage2 mean_conf={mc:.3f}, acc={acc:.3f}")
        if mc>=args.lam:
            print(f"  -> confidence {mc:.3f} >= λ {args.lam:.2f}, skip stage3.")
            continue

        outs3=outs2[:]
        for shift in range(1, args.num_shifts):
            for t in ["T0","T1","T2"]:
                jf=os.path.join(sdir,f"{t}P{shift}.jsonl")
                if run_eval(model,args.eval_name,args.data_root,jf,tok4_map[t],tok5_map[t],shift,args.ko,args.extra)==0:
                    outs3.append(jf)
        merged, mc, acc = merge_prob_files(outs3)
        save_merged(merged, os.path.join(sdir,"stage3_merged.jsonl"))
        print(f"[{model_tag}][{eval_tag}] Stage3 mean_conf={mc:.3f}, acc={acc:.3f}")
        print("  -> Finished.")

if __name__ == "__main__":
    main()
