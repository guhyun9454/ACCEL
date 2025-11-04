import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_txt", type=str, default="models.txt",
                        help="모델 리스트가 줄 단위로 적힌 파일 경로")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="모델 id를 직접 나열해서 지정 (지정 시 models.txt 무시)")
    parser.add_argument("--eval_names", type=str, nargs='+', default=["arc,0", "csqa,0"],
                        help="평가 작업 이름 리스트. 예) arc,0 csqa,0 mmlu,0")
    parser.add_argument("--data_root", type=str, default="data",
                        help="데이터 루트 디렉터리 (data_{task}들이 있는 곳)")
    parser.add_argument("--ko", action="store_true",
                        help="한국어 CSV 사용 ( *_dev.ko.csv / *_test.ko.csv )")
    parser.add_argument("--prompt_lang", type=str, choices=["en", "ko"], default="ko",
                        help="프롬프트 언어")
    parser.add_argument("--option_ids4", type=str, default="가,나,다,라",
                        help="4지선다 표기. 예) A,B,C,D 또는 가,나,다,라")
    parser.add_argument("--option_ids5", type=str, default="가,나,다,라,마",
                        help="5지선다 표기. 예) A,B,C,D,E 또는 가,나,다,라,마")
    parser.add_argument("--permute_shift", type=int, default=0,
                        help="선지 순열 시프트 (P0=0 고정 권장)")
    parser.add_argument("--save_path", type=str, default="results",
                        help="eval_clm.py가 결과(JSONL)를 저장할 디렉토리 (기본: results)")
    parser.add_argument("--tag", type=str, default="",
                        help="저장 경로 하위에 붙일 서브폴더 태그 (예: T0/T1/T2)")
    parser.add_argument("--extra", type=str, default="",
                        help="eval_clm.py에 그대로 전달할 추가 인자 문자열 (공백 구분)")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="파이썬 실행 파일 경로")
    return parser.parse_args()

def read_models(models_txt: Path) -> List[str]:
    models: List[str] = []
    if not models_txt.exists():
        raise FileNotFoundError(f"모델 목록 파일을 찾을 수 없습니다: {models_txt}")
    for line in models_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        models.append(line)
    if not models:
        raise ValueError("models.txt에 유효한 모델 항목이 없습니다.")
    return models

def main() -> None:
    args = parse_args()

    # 모델 소스: --models 우선, 없으면 models.txt
    if args.models and len(args.models) > 0:
        models = args.models
    else:
        models = read_models(Path(args.models_txt))

    # save_path 결정
    out_base = Path(args.save_path)
    if args.tag:
        out_base = out_base / args.tag
    out_base.mkdir(parents=True, exist_ok=True)

    for model in models:
        # 공통 커맨드
        cmd = [
            args.python, "code/eval_clm.py",
            "--pretrained_model_path", model,
            "--eval_names", *args.eval_names,
            "--data_root", args.data_root,
            "--prompt_lang", args.prompt_lang,
            "--option_ids4", args.option_ids4,
            "--option_ids5", args.option_ids5,
            "--permute_shift", str(args.permute_shift),
            "--save_path", str(out_base),
        ]
        if args.ko:
            cmd.append("--ko")
        if args.extra.strip():
            cmd.extend(args.extra.strip().split())

        blue, reset = "\033[94m", "\033[0m"
        print(f"{blue}Model: {model}{reset}")
        print(f"{blue}Command: {' '.join(cmd)}{reset}")

        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"[FAIL] model={model} (returncode={proc.returncode})")
        else:
            print(f"[OK] model={model}")

if __name__ == "__main__":
    main()
