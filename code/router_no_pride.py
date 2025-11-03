#!/usr/bin/env python3
"""
3-Stage Self-Escalation Router for MCQ (ARC/CSQA)

Stage 1: T0P0
Stage 2: + T1P0, T2P0   (token 확대)
Stage 3: + P1..P(S-1)   (순환 확대; S=num_shifts)

각 단계마다 JSONL을 저장하고, probs가 있으면 확률 평균으로 confidence를 평가해
임계치(lambda) 이상이면 다음 단계를 생략합니다.

eval_clm.py는 다음 패치를 적용했다고 가정합니다:
  --save_preds <jsonl_path>
  --permute_shift <int>
  (선택) --pride "<cfg>" (router_with_pride만 사용)

USAGE (ARC, 4지선다):
  python router_no_pride.py --eval_name arc,0 --num_shifts 4 --lambda 0.80 --extra "--prompt_lang en"

USAGE (CSQA, 5지선다):
  python router_no_pride.py --eval_name csqa,0 --num_shifts 5 --lambda 0.80 --extra "--prompt_lang en"
"""

import argparse, subprocess, os, json, numpy as np
from collections import defaultdict

TOKEN_SETS_4 = {"T0": "A,B,C,D", "T1": "a,b,c,d", "T2": "1,2,3,4"}
TOKEN_SETS_5 = {"T0": "A,B,C,D,E", "T1": "a,b,c,d,e", "T2": "1,2,3,4,5"}

DEFAULT_MODELS = [
    "K-intelligence/Midm-2.0-Mini-Instruct",
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    "kakaocorp/kanana-1.5-2.1b-instruct-2505",
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
    "skt/A.X-4.0-Light",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-3-1b-it",
]

def sanitize(s): return s.replace("/", "_").replace(" ", "_")

def read_jsonl(p):
    data = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def merge_probs(jsonl_paths):
    """Load multiple JSONLs and average probs per qid. Assumes all sets cover same qids.
       Returns dict: qid -> (avg_probs, gold) and summary stats (mean top1 conf, acc)."""
    bags = defaultdict(list)
    gold_map = {}
    for p in jsonl_paths:
        arr = read_jsonl(p)
        for r in arr:
            qid = r["qid"]
            pr = r.get("probs", None)
            if pr is None:
                return None, None, None  # no probs -> stop and force full run
            bags[qid].append(np.asarray(pr, dtype=float))
            if qid not in gold_map:
                gold_map[qid] = int(r["gold"])

    qids = sorted(bags.keys())
    avg_probs = {}
    corrects = []
    confs = []
    for q in qids:
        P = np.stack(bags[q], axis=0).mean(axis=0)
        P = np.clip(P, 1e-12, None)
        P /= P.sum()
        avg_probs[q] = (P, gold_map[q])
        pred = int(np.argmax(P))
        corrects.append(1 if pred == gold_map[q] else 0)
        confs.append(float(P[pred]))
    mean_conf = float(np.mean(confs))
    acc = float(np.mean(corrects))
    return avg_probs, mean_conf, acc

def ensure_dir(p): os.makedirs(p, exist_ok=True)

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
    if ko: cmd.append("--ko")
    if extra.strip(): cmd += extra.strip().split()
    
    print("  >>", " ".join(cmd))
    return subprocess.run(cmd).returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../LLM-MCQ-Bias_data")
    ap.add_argument("--eval_name", default="arc,0", help="단일 평가셋만 (예: arc,0 | csqa,0)")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--num_shifts", type=int, default=4, help="ARC=4, CSQA=5")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.80, help="평균 top-1 confidence 임계")
    ap.add_argument("--root_out", default="routes_out")
    ap.add_argument("--prompt_lang", default="en")
    ap.add_argument("--ko", action="store_true")
    ap.add_argument("--extra", default="", help='eval_clm.py로 포워드할 추가 플래그 (예: "--prompt_lang en")')
    
    args = ap.parse_args()

    # 토큰셋 결정
    tok4_map = TOKEN_SETS_4
    tok5_map = TOKEN_SETS_5

    eval_tag = args.eval_name.split(",")[0]
    for model in args.models:
        model_tag = sanitize(model)
        base_dir = os.path.join(args.root_out, eval_tag, model_tag)
        ensure_dir(base_dir)

        # Stage 1: T0P0
        stage = 1
        sdir = os.path.join(base_dir, f"stage{stage}")
        ensure_dir(sdir)
        out1 = os.path.join(sdir, "T0P0.jsonl")
        rc = run_eval(model, args.eval_name, args.data_root, out1,
                      tok4_map["T0"], tok5_map["T0"], 0, args.ko, args.extra, None)
        if rc != 0:
            print(f"[WARN] {model_tag} stage1 failed; continue to next model")
            continue

        # Check confidence
        merged, mean_conf, acc = merge_probs([out1])
        if merged is not None:
            print(f"[{model_tag}][{eval_tag}] Stage1 mean_conf={mean_conf:.3f}, acc={acc:.3f}")
            if mean_conf >= args.lam:
                print(f"  -> Stop at Stage1 (λ={args.lam})")
                continue
        else:
            print(f"[{model_tag}] No probs found; will execute all stages.")

        # Stage 2: add T1P0, T2P0
        stage = 2
        sdir = os.path.join(base_dir, f"stage{stage}")
        ensure_dir(sdir)
        outs = [out1]
        for t in ["T1", "T2"]:
            jf = os.path.join(sdir, f"{t}P0.jsonl")
            rc = run_eval(model, args.eval_name, args.data_root, jf,
                          tok4_map[t], tok5_map[t], 0, args.ko, args.extra, None)
            if rc == 0: outs.append(jf)

        merged, mean_conf, acc = merge_probs(outs)
        if merged is not None:
            print(f"[{model_tag}][{eval_tag}] Stage2 mean_conf={mean_conf:.3f}, acc={acc:.3f}")
            if mean_conf >= args.lam:
                print(f"  -> Stop at Stage2 (λ={args.lam})")
                continue

        # Stage 3: add shifts P1..P(S-1) for T0,T1,T2
        stage = 3
        sdir = os.path.join(base_dir, f"stage{stage}")
        ensure_dir(sdir)
        outs3 = outs[:]
        for shift in range(1, args.num_shifts):
            for t in ["T0", "T1", "T2"]:
                jf = os.path.join(sdir, f"{t}P{shift}.jsonl")
                rc = run_eval(model, args.eval_name, args.data_root, jf,
                              tok4_map[t], tok5_map[t], shift, args.ko, args.extra, None)
                if rc == 0: outs3.append(jf)

        merged, mean_conf, acc = merge_probs(outs3)
        if merged is not None:
            print(f"[{model_tag}][{eval_tag}] Stage3 mean_conf={mean_conf:.3f}, acc={acc:.3f}")
        print(f"  -> Finished all stages for {model_tag}.")
        
if __name__ == "__main__":
    main()
