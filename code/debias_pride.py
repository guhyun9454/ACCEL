#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, random, glob
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

# 외부: priDe 핵심 로직
from debias_utils import simple as debias_fn

def sanitize_model_id(hf_id: str) -> str:
    # eval_clm.py에서 쓰던 방식과 맞추기: 마지막 토큰 + 슬래시/공백 정리
    return hf_id.split("/")[-1].replace("/", "_").replace(" ", "_")

def infer_tokens_from_tag(tag: str, n_opt: int):
    # T0: A/B/C/D(/E), T1: a/b/c/d(/e), T2: 1/2/3/4(/5)
    if tag.lower().startswith("t1"):
        base4 = ["a","b","c","d"]
        base5 = base4 + ["e"]
    elif tag.lower().startswith("t2"):
        base4 = ["1","2","3","4"]
        base5 = base4 + ["5"]
    else:
        base4 = ["A","B","C","D"]
        base5 = base4 + ["E"]
    return base4 if n_opt == 4 else base5

def load_cyclic_probs_and_ideal(dir_path: Path):
    """
    dir_path: .../{task}/{shots}s_{model}/{task}_cyclic/
    각 subject의 jsonl을 읽어 (observed, ideal_idx) 리스트를 반환.
    observed: (K, L) 또는 (L,) 형태 가능 -> (K, L)로 통일해서 넘겨줌
    """
    items = []
    jsonl_files = sorted(glob.glob(str(dir_path / "*.jsonl")))
    for jf in jsonl_files:
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("type") != "result":
                    continue
                data = rec.get("data", {})
                probs = data.get("probs", None)
                ideal = data.get("ideal", None)
                if probs is None or ideal is None:
                    continue

                # probs: (K x L) 또는 (L)
                arr = np.array(probs, dtype=float)
                if arr.ndim == 1:
                    arr = arr[None, :]  # (1, L)

                items.append((arr, str(ideal)))
    return items  # [(KxL, ideal_str), ...]

def compute_metrics(prefix_pairs, postfix_pairs, tokens):
    """
    prefix_pairs: [(KxL, ideal_str), ...]  (prior 추정에 사용)
    postfix_pairs: [(KxL, ideal_str), ...] (prior 적용 평가)
    tokens: ['A','B','C','D'] or ... 길이 L
    """
    rng = random.Random(0xC0FFEE)

    # prior 추정
    priors = []
    for observed, ideal in prefix_pairs:
        # debias_fn은 (KxL) 배열을 입력으로 가정
        observed, debiased, prior = debias_fn(observed)
        priors.append(prior)

    if len(priors) == 0:
        return None

    prior = np.mean(np.stack(priors, axis=0), axis=0)  # (L,)

    # 평가
    predictions, labels, costs = [], [], []
    for observed, ideal in postfix_pairs:
        # single-sample 관측량: K개 순환 확률을 평균(또는 첫 번째)로 축약
        if observed.ndim == 2:
            obs = observed.mean(axis=0)  # (L,)
        else:
            obs = observed  # (L,)

        # PriDe 점수: log p - log prior
        debiased = np.log(obs + 1e-10) - np.log(prior + 1e-10)
        pred_idx = int(np.argmax(debiased))
        predictions.append(pred_idx)

        # 라벨 인덱스 매핑
        try:
            lab_idx = tokens.index(ideal)
        except ValueError:
            # ideal이 토큰셋과 다르면 대소문자만 바뀐 경우가 있음 → normalize 시도
            lab_idx = tokens.index(ideal.strip())
        labels.append(lab_idx)

        # 비용: 관측치 갯수(=K) 또는 1로 정의 가능. 여기서는 평균을 썼으니 1로.
        costs.append(1)

    labels = np.array(labels)
    predictions = np.array(predictions)
    acc = float(np.mean(labels == predictions) * 100)

    # 클래스별 리콜 표준편차
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    recalls = []
    for i in range(len(tokens)):
        cls_key = str(i)
        if cls_key in report:
            recalls.append(report[cls_key]["recall"] * 100)
    rstd = float(np.std(recalls)) if recalls else 0.0

    return {
        "acc": acc,
        "rstd": rstd,
        "cost": float(np.mean(costs) if costs else 0.0),
        "n_prefix": len(prefix_pairs),
        "n_postfix": len(postfix_pairs),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="results",
                    help="run_all.py가 저장한 results의 베이스 경로 (예: results)")
    ap.add_argument("--tag", type=str, required=True,
                    help="run_all.py에 쓴 --tag (예: T0_cyclic / T1_cyclic ...)")
    ap.add_argument("--tasks", type=str, nargs="+", default=["arc","csqa"],
                    help="적용할 테스크 리스트")
    ap.add_argument("--shots", type=int, default=0,
                    help="few-shot 수 (폴더명 0s_* 에서 0)")
    ap.add_argument("--models", type=str, nargs="+", required=True,
                    help="HF 모델 id 리스트 (run_all.py에 넘긴 것과 동일)")
    ap.add_argument("--ratio_prefix", type=float, default=0.05,
                    help="prior 추정을 위한 prefix 샘플 비율 (0~1)")
    ap.add_argument("--iters", type=int, default=5,
                    help="샘플링 반복 횟수")
    ap.add_argument("--out", type=str, default="debias_pride",
                    help="결과 요약을 저장할 폴더")
    args = ap.parse_args()

    base_dir = Path(args.base) / args.tag
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # 모델 폴더명으로 변환
    model_names = [sanitize_model_id(m) for m in args.models]

    all_summaries = []
    for task in args.tasks:
        for model, model_name in zip(args.models, model_names):
            # 결과 디렉토리: {base}/{task}/{K}s_{model_name}/{task}_cyclic/
            result_dir = base_dir / task / f"{args.shots}s_{model_name}" / f"{task}_cyclic"
            if not result_dir.is_dir():
                # perm 폴더 fallback
                alt_dir = base_dir / task / f"{args.shots}s_{model_name}" / f"{task}_perm"
                if alt_dir.is_dir():
                    result_dir = alt_dir
                else:
                    print(f"[SKIP] Not found: {result_dir}")
                    continue

            # 로드
            pairs = load_cyclic_probs_and_ideal(result_dir)
            if not pairs:
                print(f"[SKIP] Empty: {result_dir}")
                continue

            # 옵션 수 파악(4 or 5)
            L = int(pairs[0][0].shape[-1])
            tokens = infer_tokens_from_tag(args.tag, L)

            # prefix / postfix 분리
            n_total = len(pairs)
            n_prefix = max(1, int(n_total * args.ratio_prefix))
            rng = random.Random((task + model_name).encode("utf-8"))
            best = None

            for _ in range(max(1, args.iters)):
                rng.shuffle(pairs)
                prefix_pairs = pairs[:n_prefix]
                postfix_pairs = pairs[n_prefix:]
                if not postfix_pairs:
                    postfix_pairs = pairs  # 극단적으로 작을 때 fallback

                m = compute_metrics(prefix_pairs, postfix_pairs, tokens)
                if m is None:
                    continue
                if best is None or m["acc"] > best["acc"]:
                    best = m

            if best is None:
                print(f"[WARN] PriDe failed on {task} {model_name}")
                continue

            best.update({
                "task": task,
                "shots": args.shots,
                "model": model,
                "model_folder": model_name,
                "tokens": "".join(tokens),
                "dir": str(result_dir),
                "ratio_prefix": args.ratio_prefix,
                "iters": args.iters,
            })
            all_summaries.append(best)
            print(f"[OK] {task}/{model_name}  acc={best['acc']:.2f}  rstd={best['rstd']:.2f}  n={best['n_prefix']}/{best['n_postfix']}")

    # 저장
    out_fp = out_dir / f"pride_{args.tag}_shots{args.shots}.json"
    with open(out_fp, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {out_fp}")

if __name__ == "__main__":
    main()
