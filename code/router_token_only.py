#!/usr/bin/env python3
import argparse, subprocess, os, json

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
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def run_once(model, eval_name, data_root, out_jsonl, tok4, tok5, ko, extra):
    cmd = [
        "python", "code/eval_clm.py",
        "--pretrained_model_path", model,
        "--data_root", data_root,
        "--eval_names", eval_name,
        "--option_ids4", tok4,
        "--option_ids5", tok5,
        "--save_preds", out_jsonl,
        "--permute_shift", "0",
    ]
    if ko: cmd.append("--ko")
    if extra.strip(): cmd += extra.strip().split()
    print(">>", " ".join(cmd))
    return subprocess.run(cmd).returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--eval_name", default="arc,0")  # e.g., arc,0 | csqa,0
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--token", choices=["T0","T1","T2"], required=True)
    ap.add_argument("--root_out", default="routes_out/token_only")
    ap.add_argument("--extra", default="--prompt_lang en")
    ap.add_argument("--ko", action="store_true")
    args = ap.parse_args()

    tok4 = TOKEN_SETS_4[args.token]
    tok5 = TOKEN_SETS_5[args.token]
    eval_tag = args.eval_name.split(",")[0]

    for model in args.models:
        model_tag = sanitize(model)
        out_dir = os.path.join(args.root_out, eval_tag, model_tag, args.token)
        ensure_dir(out_dir)
        out_jsonl = os.path.join(out_dir, "P0.jsonl")
        rc = run_once(model, args.eval_name, args.data_root, out_jsonl, tok4, tok5, args.ko, args.extra)
        if rc != 0:
            print(f"[WARN] failed: {model_tag} ({args.token})")

if __name__ == "__main__":
    main()
