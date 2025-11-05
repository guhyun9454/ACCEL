#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ece_viz.py
- 디렉토리 구조: <root>/<token>/<dataset>/<model>/<dataset>.jsonl
- 각 jsonl 한 줄은 다음 둘 중 하나를 가정
  (A) {"type":"result","data":{ ... "probs":[...], "correct":true/false, "qid":... }}
  (B) 평면형 { "probs":[...], "correct":true/false, "qid":... }  (+ 일부 키 변형 허용)
- 출력:
  1) 히스토그램(정답/오답 confidence 분포, T0/T1/T2 색상 고정: R/B/G)
  2) 리라이어빌리티 다이어그램(토큰별)
  3) ECE/정확도 요약 TSV

사용 예)
  python code/ece_viz.py --root results --datasets arc csqa \
    --models "Qwen/Qwen2.5-1.5B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" \
    --outdir viz_out/ece
"""

import os, json, argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np

# 헤드리스 환경 안전
import matplotlib
if os.environ.get("MPLBACKEND") is None:
    os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt


# ────────────── 공통 유틸 ──────────────
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_tag(s: str) -> str:
    """파일명/경로에 안전한 태그로 치환"""
    return str(s).replace("/", "_").replace("\\", "_").strip()

def flatten_record(rec: dict) -> dict:
    """{"type":"result","data":{...}} → {...} 로 평탄화"""
    if isinstance(rec, dict) and "data" in rec and isinstance(rec["data"], dict):
        d = dict(rec["data"])
        if "qid" not in d and "idx" in d:
            d["qid"] = str(d["idx"])
        return d
    return rec

def normalize_letter(x: Optional[str]) -> Optional[str]:
    """문자형 정답을 비교하기 쉽게 정규화"""
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    ch = s[0]
    if ch in "([{":
        ch = s[1] if len(s) > 1 else ch
    if ch.isalpha():
        ch = ch.upper()
    return ch

def fold_probs(p: np.ndarray, n_opts_hint: Optional[int] = None) -> np.ndarray:
    """
    probs가 2D면 평균, 길이가 2*C면 (공백/비공백 등) 접어서 C로 합산.
    """
    arr = np.array(p, dtype=float)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    if arr.ndim == 1 and arr.size % 2 == 0 and (n_opts_hint is None or arr.size == 2*n_opts_hint):
        arr = arr.reshape(2, arr.size // 2).sum(axis=0)
    arr = np.clip(arr, 1e-12, None)
    arr = arr / arr.sum()
    return arr

def ece_from_conf(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """top-1 confidence 기반 ECE"""
    bins = np.linspace(0., 1., n_bins + 1)
    N = len(conf)
    ece_val = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (conf >= lo) & (conf <= hi) if b == 0 else ((conf > lo) & (conf <= hi))
        if not np.any(mask):
            continue
        acc_b = correct[mask].mean()
        conf_b = conf[mask].mean()
        ece_val += (mask.sum() / N) * abs(acc_b - conf_b)
    return float(ece_val)


# ────────────── 로딩 ──────────────
def collect_models(root: str, tokens: List[str], datasets: List[str]) -> List[str]:
    """결과 폴더에서 모델 디렉토리 자동 수집 (폴더명 기준: 슬래시가 '_'로 치환된 형태일 가능성 큼)"""
    mset = set()
    for t in tokens:
        for ds in datasets:
            base = Path(root) / t / ds
            if base.is_dir():
                for m in base.iterdir():
                    if m.is_dir():
                        mset.add(m.name)
    return sorted(mset)

def filter_models(all_models: List[str], wants: Optional[List[str]]) -> List[str]:
    """--models 가 주어지면 부분문자열 매칭(원문/치환문 모두)으로 필터링"""
    if not wants:
        return all_models
    wants_sanit = [safe_tag(w).lower() for w in wants]
    wants_raw   = [w.lower() for w in wants]
    out = []
    for m in all_models:
        ml = m.lower()
        keep = any(w in ml for w in wants_sanit) or any(w in ml for w in wants_raw)
        if keep:
            out.append(m)
    return sorted(set(out))

def load_token_records(fp: Path) -> List[dict]:
    if not fp.is_file():
        return []
    rows: List[dict] = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            rows.append(flatten_record(rec))
    return rows

def extract_conf_correct(rows: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    records → (conf, correct). probs 없으면 스킵.
    correct 없으면 pred vs gold로 복원(가능한 경우).
    """
    confs: List[float] = []
    rights: List[int] = []
    for r in rows:
        p = r.get("probs") or r.get("all_probs")
        if p is None:
            continue
        probs = fold_probs(p)
        confs.append(float(np.max(probs)))

        # correct 우선 사용
        c = r.get("correct", None)
        if isinstance(c, str):
            c = c.lower() in ("true", "1", "yes")
        if isinstance(c, bool):
            rights.append(1 if c else 0)
            continue

        # correct가 없으면 pred/gold로 복원
        # 문자 비교: sampled vs ideal
        p_letter = normalize_letter(r.get("sampled"))
        g_letter = normalize_letter(r.get("ideal"))
        if p_letter is not None and g_letter is not None:
            rights.append(1 if p_letter == g_letter else 0)
            continue

        # 인덱스 비교
        pred_idx = None
        for k in ("pred_idx","prediction_idx","answer_idx","choice_idx","selected_idx","output_idx"):
            if k in r:
                try:
                    vi = int(r[k])
                    if vi >= 0:
                        pred_idx = vi
                        break
                except Exception:
                    pass
        gold_idx = None
        for k in ("gold_idx","label_idx","correct_idx","target_idx","solution_idx"):
            if k in r:
                try:
                    vi = int(r[k])
                    if vi >= 0:
                        gold_idx = vi
                        break
                except Exception:
                    pass
        if pred_idx is None:
            pred_idx = int(np.argmax(probs))
        if gold_idx is None:
            # gold 정보를 전혀 못 구하면 이 샘플은 제외
            confs.pop()
            continue
        rights.append(1 if pred_idx == gold_idx else 0)

    if not confs:
        return np.array([]), np.array([])
    return np.array(confs, dtype=float), np.array(rights, dtype=int)


# ────────────── 시각화 ──────────────
def plot_hist_by_token(model_tag: str,
                       dataset: str,
                       conf_by_tok: Dict[str, np.ndarray],
                       right_by_tok: Dict[str, np.ndarray],
                       outdir: Path,
                       bins: int = 20) -> None:
    """토큰별(빨/파/초) 정답/오답 히스토그램 오버레이(단일 플롯)."""
    color_map = {"T0": "red", "T1": "blue", "T2": "green"}
    fig = plt.figure()
    for t in sorted(conf_by_tok.keys()):
        conf = conf_by_tok[t]; right = right_by_tok[t]
        if conf.size == 0:
            continue
        conf_ok  = conf[right == 1]
        conf_bad = conf[right == 0]
        plt.hist(conf_bad, bins=bins, range=(0.0, 1.0), alpha=0.35,
                 color=color_map.get(t, "gray"), density=True,
                 label=f"{t} wrong (n={len(conf_bad)})")
        plt.hist(conf_ok, bins=bins, range=(0.0, 1.0), alpha=0.55,
                 color=color_map.get(t, "gray"), density=True,
                 label=f"{t} correct (n={len(conf_ok)})")
    plt.xlabel("Confidence (top-1 prob)")
    plt.ylabel("Density")
    plt.title(f"{model_tag} — {dataset} | Confidence distribution by token")
    plt.legend(fontsize=9)
    out_png = outdir / f"{safe_tag(model_tag)}_{dataset}_conf_hist.png"
    ensure_dir(out_png.parent)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_reliability_by_token(model_tag: str,
                              dataset: str,
                              conf_by_tok: Dict[str, np.ndarray],
                              right_by_tok: Dict[str, np.ndarray],
                              outdir: Path,
                              n_bins: int = 15) -> None:
    """토큰별 리라이어빌리티 다이어그램(단일 플롯)."""
    color_map = {"T0": "red", "T1": "blue", "T2": "green"}
    fig = plt.figure()
    xs = np.linspace(0, 1, 101)
    plt.plot(xs, xs, linestyle="--", color="black", linewidth=1, label="Ideal")
    for t in sorted(conf_by_tok.keys()):
        conf = conf_by_tok[t]; right = right_by_tok[t]
        if conf.size == 0:
            continue
        bins = np.linspace(0., 1., n_bins + 1)
        mids = 0.5 * (bins[:-1] + bins[1:])
        accs = []
        for b in range(n_bins):
            lo, hi = bins[b], bins[b + 1]
            mask = (conf >= lo) & (conf <= hi) if b == 0 else ((conf > lo) & (conf <= hi))
            accs.append(right[mask].mean() if np.any(mask) else np.nan)
        accs = np.array(accs, dtype=float)
        plt.plot(mids, accs, marker="o", color=color_map.get(t, "gray"), label=f"{t}")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title(f"{model_tag} — {dataset} | Reliability by token")
    plt.legend()
    out_png = outdir / f"{safe_tag(model_tag)}_{dataset}_reliability.png"
    ensure_dir(out_png.parent)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ────────────── 메인 ──────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="결과 루트 (예: results)")
    ap.add_argument("--datasets", nargs="+", default=["arc", "csqa"])
    ap.add_argument("--tokens", nargs="+", default=["T0", "T1", "T2"])
    ap.add_argument("--models", nargs="*", default=None,
                    help='특정 모델만 선택 (부분문자열 매칭 가능). 비우면 자동 탐색.')
    ap.add_argument("--outdir", default="viz_out/ece")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--n_bins_ece", type=int, default=15)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    auto_models = collect_models(args.root, args.tokens, args.datasets)
    models = filter_models(auto_models, args.models)

    if not models:
        print("[WARN] 모델 디렉토리를 찾지 못했습니다.")
        return

    # 요약 TSV 헤더
    tsv_lines = ["model\tdataset\ttoken\tn\tacc\tece\n"]

    for model in models:
        for ds in args.datasets:
            conf_by_tok: Dict[str, np.ndarray] = {}
            right_by_tok: Dict[str, np.ndarray] = {}

            for t in args.tokens:
                jf = Path(args.root) / t / ds / model / f"{ds}.jsonl"
                rows = load_token_records(jf)
                conf, right = extract_conf_correct(rows)
                conf_by_tok[t] = conf
                right_by_tok[t] = right

            # 그림들 저장
            plot_hist_by_token(model, ds, conf_by_tok, right_by_tok, outdir, bins=args.bins)
            plot_reliability_by_token(model, ds, conf_by_tok, right_by_tok, outdir, n_bins=args.n_bins_ece)

            # 수치 요약(ECE/ACC)
            for t in args.tokens:
                conf = conf_by_tok[t]; right = right_by_tok[t]
                if conf.size == 0:
                    continue
                ece_val = ece_from_conf(conf, right, n_bins=args.n_bins_ece)
                acc_val = float(right.mean())
                tsv_lines.append(f"{model}\t{ds}\t{t}\t{len(conf)}\t{acc_val:.4f}\t{ece_val:.4f}\n")

    # TSV 저장
    tsv_path = outdir / "ece_summary.tsv"
    ensure_dir(tsv_path.parent)
    with tsv_path.open("w", encoding="utf-8") as w:
        w.writelines(tsv_lines)
    print(f"[DONE] Saved: {tsv_path}")

if __name__ == "__main__":
    main()
