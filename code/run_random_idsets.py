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
    if allow_digits: pool += list("12345")  # 4옵션 안전 범위
    choices = random.sample(pool, n)
    label = "".join(choices)          # ← 콤마 없는 4글자 (eval_clm.py가 요구)
    arg = ",".join(choices)           # 로그용/기록용
    return label, arg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, nargs="+", required=True)
    ap.add_argument("--n_pairs", type=int, default=10)
    ap.add_argument("--base_set", type=str, default="ABCD")
    ap.add_argument("--eval_names", type=str, nargs="+", required=True)
    # ▼ 필수 → 옵션으로 변경 (code/ 안에서 상대경로 쓰면 불필요)
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--save_path", type=str, default="out")
    ap.add_argument("--cache_dir", type=str, default="../models")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print_prompt_example", action="store_true")
    ap.add_argument("--allow_digits", action="store_true")
    ap.add_argument("--cuda", type=str, default=None)
    args = ap.parse_args()

    random.seed(args.seed)

    # base_set 라벨/인자 정규화
    if "," in args.base_set:
        base_label = "".join([c for c in args.base_set if c.isalnum()])
    else:
        base_label = "".join([c for c in args.base_set if c.isalnum()])

    logs_dir = os.path.join(args.save_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # 라운드별 공통 랜덤 idset을 먼저 뽑음 → 각 라운드에서 모든 모델에 동일 적용
    idsets = []
    seen = set([base_label])  # base와 동일한 조합은 제외
    while len(idsets) < args.n_pairs:
        mix_label, mix_arg = make_random_idset(
            n=4, allow_upper=True, allow_lower=True, allow_digits=args.allow_digits
        )
        if mix_label in seen:
            continue
        seen.add(mix_label)
        idsets.append((mix_label, mix_arg))

    # 이번 실행에서 사용한 idset 기록(로그)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(logs_dir, f"idsets_{run_tag}.txt"), "w", encoding="utf-8") as f:
        for i, (lab, arg) in enumerate(idsets):
            f.write(f"{i:04d}\tlabel={lab}\targ={arg}\n")

    # 각 라운드 k의 idset을 모든 모델에 공통 적용
    for k, (mix_label, mix_arg) in enumerate(idsets):
        for model in args.models:
            model_name = sanitize_model_name(model)

            cmd = [
                sys.executable, os.path.join(HERE, "eval_clm.py"),
                "--pretrained_model_path", model,
                "--model_name", model_name,
                "--eval_names"
            ] + args.eval_names + [
                # ★ 핵심: eval_clm.py는 '콤마 없는 4글자'를 받으므로 mix_label 사용
                "--option_id_sets", base_label, mix_label,
                "--save_path", args.save_path,
                "--cache_dir", args.cache_dir,
            ]
            # data_root가 있을 때만 전달 (없으면 code/ 기준 상대경로 사용)
            if args.data_root:
                cmd += ["--data_root", args.data_root]

            if args.print_prompt_example:
                cmd.append("--print_prompt_example")

            env = os.environ.copy()
            if args.cuda is not None:
                env["CUDA_VISIBLE_DEVICES"] = args.cuda

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(
                logs_dir, f"{ts}_{k:04d}_{model_name}_{base_label}_to_{mix_label}.log"
            )
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
            time.sleep(1.0)  # 로그 타임스탬프 충돌 방지

    print(f"[OK] Completed {args.n_pairs} random pairs (shared across {len(args.models)} models).")

if __name__ == "__main__":
    main()
