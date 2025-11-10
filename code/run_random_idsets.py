#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse, random, string, subprocess, shlex, time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))

def sanitize_model_name(model_id: str) -> str:
    return model_id.split("/")[-1].replace("/", "_").replace(" ", "_")

def make_random_idset(n=4, allow_upper=True, allow_lower=True, allow_digits=False):
    pool = []
    if allow_upper: pool += list(string.ascii_uppercase)
    if allow_lower: pool += list(string.ascii_lowercase)
    if allow_digits: pool += list("12345")  # keep 4 options safe
    choices = random.sample(pool, n)
    label = "".join(choices)
    arg = ",".join(choices)
    return label, arg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, nargs="+", required=True)
    ap.add_argument("--n_pairs", type=int, default=10)
    ap.add_argument("--base_set", type=str, default="ABCD")
    ap.add_argument("--eval_names", type=str, nargs="+", required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--save_path", type=str, default="out")
    ap.add_argument("--cache_dir", type=str, default="../models")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print_prompt_example", action="store_true")
    ap.add_argument("--allow_digits", action="store_true")
    ap.add_argument("--cuda", type=str, default=None)
    args = ap.parse_args()

    random.seed(args.seed)

    if "," in args.base_set:
        base_label = "".join([c for c in args.base_set if c.isalnum()])
        base_arg = args.base_set
    else:
        base_label = "".join([c for c in args.base_set if c.isalnum()])
        base_arg = ",".join(list(args.base_set))

    logs_dir = os.path.join(args.save_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    for model in args.models:
        model_name = sanitize_model_name(model)
        for k in range(args.n_pairs):
            mix_label, mix_arg = make_random_idset(
                n=4, allow_upper=True, allow_lower=True, allow_digits=args.allow_digits
            )
            if mix_label == base_label:
                mix_label, mix_arg = make_random_idset(
                    n=4, allow_upper=True, allow_lower=True, allow_digits=args.allow_digits
                )

            cmd = [
                sys.executable, os.path.join(HERE, "eval_clm.py"),
                "--pretrained_model_path", model,
                "--model_name", model_name,
                "--eval_names"
            ] + args.eval_names + [
                "--option_id_sets", base_label, mix_arg,
                "--save_path", args.save_path,
                "--cache_dir", args.cache_dir,
                "--data_root", args.data_root
            ]
            if args.print_prompt_example:
                cmd.append("--print_prompt_example")

            env = os.environ.copy()
            if args.cuda is not None:
                env["CUDA_VISIBLE_DEVICES"] = args.cuda

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(logs_dir, f"{ts}_{model_name}_{base_label}_to_{mix_label}.log")
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write("# CMD: " + " ".join(shlex.quote(x) for x in cmd) + "\n")
                lf.flush()
                try:
                    p = subprocess.Popen(cmd, stdout=lf, stderr=lf, env=env)
                    p.wait()
                except KeyboardInterrupt:
                    return
                except Exception as e:
                    lf.write(f"\n[run_random_idsets] ERROR: {e}\n")
            time.sleep(1.0)

    print(f"[OK] Completed {args.n_pairs} random pairs for {len(args.models)} models.")

if __name__ == "__main__":
    main()
