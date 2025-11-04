#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, argparse, subprocess, shutil, math
from collections import defaultdict, Counter

import numpy as np

TOKSETS_EN4 = {
    "T0": "A,B,C,D",
    "T1": "a,b,c,d",
    "T2": "1,2,3,4",
}
TOKSETS_EN5 = {
    "T0": "A,B,C,D,E",
    "T1": "a,b,c,d,e",
    "T2": "1,2,3,4,5",
}
TOKSETS_KO4 = {
    "T0": "A,B,C,D",          # 모델별 토큰 처리 일관성을 위해 내부 헤더는 영문 고정 권장
    "T1": "가,나,다,라",
    "T2": "1,2,3,4",
}
TOKSETS_KO5 = {
    "T0": "A,B,C,D,E",
    "T1": "가,나,다,라,마",
    "T2": "1,2,3,4,5",
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--eval_name", required=True, help="e.g., arc,0 or csqa,0")
    ap.add_argument("--num_shifts", type=int, default=4)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.80)
    ap.add_argument("--root_out", default="routes_out/pride_on")
    ap.add_argument("--extra", default="", help="extra args passed to eval_clm.py")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--ko", action="store_true")
    ap.add_argument("--pride", default="method=paraphrase,k=3,seed=42")
    return ap.parse_args()

def run_eval(model, eval_name, data_root, out_jsonl, tok4, tok5, shift, ko, extra, pride_cfg):
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
    if ko:
        cmd.append("--ko")
    if extra.strip():
        cmd += extra.strip().split()
    if pride_cfg:
        cmd += ["--pride", pride_cfg]

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    print("[CMD]", " ".join(cmd))
    rc = subprocess.call(cmd)
    return rc

def _read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def _pull(rec, key_list, default=None):
    for k in key_list:
        if k in rec and rec[k] is not None:
            return rec[k]
    return default

def _norm(p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, None)
    s = p.sum()
    return (p / s) if s > 0 else np.ones_like(p)/len(p)

def merge_prob_files(jsonl_paths):
    """
    jsonl들의 probs를 qid별로 평균.
    probs가 없는 파일만 있는 경우에는 pred_idx 다수결.
    """
    bag_probs = defaultdict(list)   # qid -> [probs...]
    bag_votes = defaultdict(list)   # qid -> [pred_idx...]
    gold_map = {}

    for p in jsonl_paths:
        for r in _read_jsonl(p):
            qid = _pull(r, ["qid","id"])
            if qid is None: 
                continue
            gold_idx = _pull(r, ["gold_idx","label_idx","label","gold","answer"], -1)
            if gold_idx is not None:
                try: gold_idx = int(gold_idx)
                except Exception: gold_idx = -1
            gold_map[qid] = gold_idx

            probs = _pull(r, ["probs"], None)
            if isinstance(probs, (list,tuple)) and len(probs) > 0:
                bag_probs[qid].append(_norm(probs))
            else:
                pred_idx = _pull(r, ["pred_idx","prediction_idx","pred","prediction"], -1)
                try: pred_idx = int(pred_idx)
                except Exception: pred_idx = -1
                if pred_idx >= 0:
                    bag_votes[qid].append(pred_idx)

    merged = {}
    for qid in set(list(bag_probs.keys()) + list(bag_votes.keys()) + list(gold_map.keys())):
        G = gold_map.get(qid, -1)
        if bag_probs[qid]:
            P = np.mean(bag_probs[qid], axis=0)
            P = _norm(P)
            pred = int(np.argmax(P))
            conf = float(P[pred])
            merged[qid] = dict(gold_idx=G, pred_idx=pred, probs=P.tolist(), conf=conf)
        else:
            # probs가 없으면 다수결
            votes = bag_votes[qid]
            if votes:
                cnt = Counter(votes)
                pred, _ = max(cnt.items(), key=lambda x: (x[1], -x[0]))
            else:
                pred = -1
            merged[qid] = dict(gold_idx=G, pred_idx=pred, probs=None, conf=None)

    # 요약 통계
    qids = sorted(merged.keys())
    correct = [1 if (merged[q]["pred_idx"] == merged[q]["gold_idx"] and merged[q]["pred_idx"] >= 0) else 0 for q in qids]
    confs = [merged[q]["conf"] for q in qids if merged[q]["conf"] is not None]
    mean_conf = float(np.mean(confs)) if confs else 0.0
    acc = float(np.mean(correct)) if correct else 0.0
    return merged, mean_conf, acc

def save_merged(merged, out_jsonl):
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as w:
        for qid in sorted(merged.keys()):
            rec = dict(qid=qid, **merged[qid])
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[SAVE] {out_jsonl} ({len(merged)} rows)")

def main():
    args = parse_args()
    task = args.eval_name.split(",")[0]
    tok4_map = TOKSETS_KO4 if args.ko else TOKSETS_EN4
    tok5_map = TOKSETS_KO5 if args.ko else TOKSETS_EN5

    models = args.models or [
        "K-intelligence/Midm-2.0-Mini-Instruct",
    ]

    for model in models:
        model_tag = model.split("/")[-1]
        eval_tag = args.eval_name.replace(",", "_")
        sdir = os.path.join(args.root_out, task, model_tag, eval_tag)
        os.makedirs(sdir, exist_ok=True)

        # ---------- Stage 1: T0P0 ----------
        outs1 = []
        jf = os.path.join(sdir, "T0P0.jsonl")
        rc = run_eval(model, args.eval_name, args.data_root, jf,
                      tok4_map["T0"], tok5_map["T0"], 0, args.ko, args.extra, args.pride)
        if rc == 0: outs1.append(jf)

        merged, mean_conf, acc = merge_prob_files(outs1)
        save_merged(merged, os.path.join(sdir, "stage1_merged.jsonl"))
        print(f"[{model_tag}][{eval_tag}] Stage1 mean_conf={mean_conf:.3f}, acc={acc:.3f}")

        if mean_conf >= args.lam:
            print(f"  -> confidence {mean_conf:.3f} >= λ {args.lam:.2f}, skip later stages.")
            continue

        # ---------- Stage 2: T1/T2 with P0 ----------
        outs2 = outs1[:]
        for t in ["T1","T2"]:
            jf = os.path.join(sdir, f"{t}P0.jsonl")
            rc = run_eval(model, args.eval_name, args.data_root, jf,
                          tok4_map[t], tok5_map[t], 0, args.ko, args.extra, args.pride)
            if rc == 0: outs2.append(jf)

        merged, mean_conf, acc = merge_prob_files(outs2)
        save_merged(merged, os.path.join(sdir, "stage2_merged.jsonl"))
        print(f"[{model_tag}][{eval_tag}] Stage2 mean_conf={mean_conf:.3f}, acc={acc:.3f}")

        if mean_conf >= args.lam:
            print(f"  -> confidence {mean_conf:.3f} >= λ {args.lam:.2f}, skip stage3.")
            continue

        # ---------- Stage 3: add P1..P(S-1) for T0/T1/T2 ----------
        outs3 = outs2[:]
        for shift in range(1, args.num_shifts):
            for t in ["T0","T1","T2"]:
                jf = os.path.join(sdir, f"{t}P{shift}.jsonl")
                rc = run_eval(model, args.eval_name, args.data_root, jf,
                              tok4_map[t], tok5_map[t], shift, args.ko, args.extra, args.pride)
                if rc == 0: outs3.append(jf)

        merged, mean_conf, acc = merge_prob_files(outs3)
        save_merged(merged, os.path.join(sdir, "stage3_merged.jsonl"))
        print(f"[{model_tag}][{eval_tag}] Stage3 mean_conf={mean_conf:.3f}, acc={acc:.3f}")
        print(f"  -> Finished all stages for {model_tag}.")

if __name__ == "__main__":
    main()
