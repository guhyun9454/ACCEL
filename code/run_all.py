# code/run_all.py
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

# utils.py에 아래 함수들이 있어야 합니다:
# - apply_shift_to_option_ids
# - num_min_runs_from_ids
# - min_runs_for_task
from utils import (
    apply_shift_to_option_ids,
    num_min_runs_from_ids,
    min_runs_for_task,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models_txt", type=str, default="models.txt",
                   help="모델 id가 줄 단위로 적힌 파일")
    p.add_argument("--models", type=str, nargs="+", default=None,
                   help="모델 id 직접 나열 (지정하면 models.txt 무시)")
    p.add_argument("--eval_names", type=str, nargs='+', default=["arc,0", "csqa,0"],
                   help="예: arc,0 csqa,0 mmlu,0")
    p.add_argument("--data_root", type=str, default="data",
                   help="data_{task} 루트 디렉터리")
    p.add_argument("--ko", action="store_true",
                   help="한국어 CSV 사용 (*_dev.ko.csv / *_test.ko.csv)")
    p.add_argument("--prompt_lang", type=str, choices=["en", "ko"], default="ko",
                   help="프롬프트 언어")
    p.add_argument("--option_ids4", type=str, default="가,나,다,라",
                   help="4지 라벨 (예: A,B,C,D / 가,나,다,라)")
    p.add_argument("--option_ids5", type=str, default="가,나,다,라,마",
                   help="5지 라벨 (예: A,B,C,D,E / 가,나,다,라,마)")

    # 한 번만 돌릴 때는 permute_shift를 쓰고,
    # 여러 번 자동으로 돌리려면 --auto_cycle 사용
    p.add_argument("--permute_shift", type=int, default=0,
                   help="라벨 회전 시프트(단일 실행용)")
    p.add_argument("--auto_cycle", action="store_true",
                   help="각 task별 최소 회전 횟수(N=선지 수)만큼 자동 반복")

    p.add_argument("--save_path", type=str, default="results",
                   help="eval_clm.py의 결과 베이스 디렉터리")
    p.add_argument("--tag", type=str, default="",
                   help="베이스 하위 서브폴더 (예: T0 / T1 / T2)")
    p.add_argument("--extra", type=str, default="",
                   help="eval_clm.py로 그대로 전달할 추가 인자 문자열")
    p.add_argument("--python", type=str, default=sys.executable,
                   help="파이썬 실행 파일 경로")
    p.add_argument("--separate_shift_subdirs", action="store_true", default=True,
                   help="shift별로 S00, S01 ... 하위 폴더를 만들어 덮어쓰기 방지")
    p.add_argument("--shift_prefix", type=str, default="S",
                   help="shift 하위 폴더 접두사 (기본 S)")
    p.add_argument("--dry_run", action="store_true",
                   help="명령만 출력하고 실행하지 않음")
    return p.parse_args()

def read_models(models_txt: Path) -> List[str]:
    if not models_txt.exists():
        raise FileNotFoundError(f"모델 목록 파일을 찾을 수 없습니다: {models_txt}")
    models: List[str] = []
    for line in models_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        models.append(line)
    if not models:
        raise ValueError("models.txt에 유효한 모델이 없습니다.")
    return models

def _blue(s: str) -> str:
    return f"\033[94m{s}\033[0m"

def main() -> None:
    args = parse_args()

    # 모델 소스 확정
    if args.models and len(args.models) > 0:
        models = args.models
    else:
        models = read_models(Path(args.models_txt))

    # 베이스 저장경로 결정
    base = Path(args.save_path)
    if args.tag:
        base = base / args.tag
    base.mkdir(parents=True, exist_ok=True)

    # eval_clm.py는 eval_names를 한꺼번에 넘겨도 되지만
    # 각 task별로 "회전 횟수"가 다르기 때문에 여기서는
    # task별로 **개별 실행**해서 N회 돌려준다.
    for model in models:
        print(_blue(f"Model: {model}"))

        for eval_name in args.eval_names:
            # eval_name 파싱: "task,0" 또는 "task,0,setting"
            parts = [e.strip() for e in eval_name.split(",")]
            task = parts[0].lower()

            # 이 task가 4지면 option_ids4 길이, 5지면 option_ids5 길이만큼 반복
            if args.auto_cycle:
                runs = min_runs_for_task(task, args.option_ids4, args.option_ids5)
                shifts = list(range(runs))
            else:
                # 단일 실행: permute_shift만 사용 (각 라벨 길이에 mod 연산)
                if task == "csqa":
                    n = num_min_runs_from_ids(args.option_ids5)
                else:
                    n = num_min_runs_from_ids(args.option_ids4)
                shifts = [args.permute_shift % max(1, n)]

            for s in shifts:
                # 이 실행에서 사용할 라벨 문자열 (4/5 각각 회전 적용)
                opt4 = apply_shift_to_option_ids(args.option_ids4, s)
                opt5 = apply_shift_to_option_ids(args.option_ids5, s)

                # 덮어쓰기 방지용 subdir
                if args.separate_shift_subdirs:
                    out_dir = base / f"{args.shift_prefix}{s:02d}"
                else:
                    out_dir = base
                out_dir.mkdir(parents=True, exist_ok=True)

                # 개별 task만 넘긴다 (task별 N회 반복을 위해)
                cmd = [
                    args.python, "code/eval_clm.py",
                    "--pretrained_model_path", model,
                    "--eval_names", eval_name,  # 단일 task
                    "--data_root", args.data_root,
                    "--prompt_lang", args.prompt_lang,
                    "--option_ids4", opt4,
                    "--option_ids5", opt5,
                    "--permute_shift", str(s),
                    "--save_path", str(out_dir),
                ]
                if args.ko:
                    cmd.append("--ko")
                if args.extra.strip():
                    cmd.extend(args.extra.strip().split())

                print(_blue(f"Task: {eval_name} | Shift: {s} | Save: {out_dir}"))
                print(_blue("Command: " + " ".join(cmd)))

                if args.dry_run:
                    continue

                proc = subprocess.run(cmd)
                if proc.returncode != 0:
                    print(f"[FAIL] model={model}, task={eval_name}, shift={s} (rc={proc.returncode})")
                else:
                    print(f"[OK] model={model}, task={eval_name}, shift={s}")

if __name__ == "__main__":
    main()
