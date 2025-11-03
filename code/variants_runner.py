
#!/usr/bin/env python3
"""
variants_runner.py  (updated)

- 3 token sets:
    T0: 4-choice "A,B,C,D", 5-choice "A,B,C,D,E"
    T1: 4-choice "a,b,c,d", 5-choice "a,b,c,d,e"
    T2: 4-choice "1,2,3,4", 5-choice "1,2,3,4,5"
- permute shifts: 0..(num_shifts-1)
- Saves per-variation JSONL via --save_preds
- Works with eval_clm.py patch that supports --permute_shift and remaps back to ORIGINAL order

USAGE (ARC=4지선다):
  python variants_runner.py --num_shifts 4 --eval_names arc,0 --extra "--prompt_lang en"

USAGE (CSQA=5지선다):
  python variants_runner.py --num_shifts 5 --eval_names csqa,0 --extra "--prompt_lang en"

(여러 평가셋 동시 가능하지만, 4/5지선다 섞여 있으면 num_shifts가 안 맞을 수 있어
 ARC/CSQA를 따로 한 번씩 돌리는 걸 추천합니다.)
"""

import argparse
import os
import subprocess

TOKEN_SETS_4 = {
    "T0": "A,B,C,D",
    "T1": "a,b,c,d",
    "T2": "1,2,3,4",
}
TOKEN_SETS_5 = {
    "T0": "A,B,C,D,E",
    "T1": "a,b,c,d,e",
    "T2": "1,2,3,4,5",
}

def sanitize(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../LLM-MCQ-Bias_data")
    ap.add_argument("--eval_names", nargs="+", default=["arc,0", "csqa,0"])
    ap.add_argument("--models", nargs="+", default=[
        "K-intelligence/Midm-2.0-Mini-Instruct",
        "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        "kakaocorp/kanana-1.5-2.1b-instruct-2505",
        "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
        "skt/A.X-4.0-Light",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "google/gemma-3-1b-it",
    ])
    ap.add_argument("--root_out", default="outputs_variants")
    ap.add_argument("--use_pride", action="store_true")
    ap.add_argument("--pride", default="method=paraphrase,k=3,seed=42")
    ap.add_argument("--prompt_lang", default="en")
    ap.add_argument("--ko", action="store_true")
    ap.add_argument("--extra", default="", help="Extra flags forwarded to eval_clm.py")
    ap.add_argument("--num_shifts", type=int, default=4, help="4-choice=4, 5-choice=5")
    args = ap.parse_args()

    os.makedirs(args.root_out, exist_ok=True)

    for model in args.models:
        model_tag = sanitize(model)
        model_dir = os.path.join(args.root_out, model_tag)
        os.makedirs(model_dir, exist_ok=True)

        for tname in ["T0", "T1", "T2"]:
            tok4 = TOKEN_SETS_4[tname]
            tok5 = TOKEN_SETS_5[tname]
            for shift in range(args.num_shifts):
                variation_id = f"{tname}P{shift}"
                out_jsonl = os.path.join(model_dir, f"{variation_id}.jsonl")

                cmd = [
                    "python", "code/eval_clm.py",
                    "--pretrained_model_path", model,
                    "--data_root", args.data_root,
                    "--prompt_lang", args.prompt_lang,
                    "--option_ids4", tok4,
                    "--option_ids5", tok5,
                    "--save_preds", out_jsonl,
                    "--permute_shift", str(shift),
                ]
                cmd += ["--eval_names"] + args.eval_names

                if args.ko:
                    cmd.append("--ko")
                if args.use_pride:
                    cmd += ["--pride", args.pride]
                if args.extra.strip():
                    cmd += args.extra.strip().split()

                print(f"[Run] {model} | {variation_id}")
                print("      ", " ".join(cmd))
                proc = subprocess.run(cmd)
                if proc.returncode != 0:
                    print(f"[WARN] Non-zero return code for {model} {variation_id}: {proc.returncode}")

    print("\nAll variants attempted. Check JSONLs under:", args.root_out)

if __name__ == "__main__":
    main()
