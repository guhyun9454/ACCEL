#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECE & Matching-Ratio visualizer (pretty palette, font tweaks OFF)

지원 경로:
  1) <root>/<token>/<dataset>/<model>/<dataset>.jsonl
  2) <root>/<token>/<dataset>/<model>/<dataset>/<dataset>.jsonl

출력:
  - 히스토그램(정답/오답 완전 분리: 좌/우 패널, T0/T1/T2 고정색)
  - (옵션) 토큰별 오버레이 히스토그램
  - 리라이어빌리티 다이어그램(토큰별)
  - 매칭비율 히트맵 3종: all / correct-only / wrong-only  (셀에 % 주석, 축=ABCD/abcd/1234)
  - ECE/정확도 요약 TSV
"""

import os, json, argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np

# --- headless 안전 ---
import matplotlib
if os.environ.get("MPLBACKEND") is None:
    os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# =========================
#  스타일 & 팔레트
# =========================
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("ggplot")
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10

# 기본색(정답=진한, 오답=연한)
BASE_COL = {"T0": "#4C78A8", "T1": "#59A14F", "T2": "#E15759"}
def lighten(c, amount=0.55):
    r, g, b = mcolors.to_rgb(c)
    return (1 - amount) + amount*r, (1 - amount) + amount*g, (1 - amount) + amount*b

TOKEN_LABEL = {"T0": "ABCD", "T1": "abcd", "T2": "1234"}

# =========================
#  유틸/파서
# =========================
def safe_tag(s: str) -> str:
    return str(s).replace("/", "_").replace("\\", "_").strip()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def flatten_record(rec: dict) -> dict:
    if isinstance(rec, dict) and "data" in rec and isinstance(rec["data"], dict):
        d = dict(rec["data"])
        if "qid" not in d and "idx" in d:
            d["qid"] = str(d["idx"])
        return d
    return rec

def normalize_letter(x: Optional[str]) -> Optional[str]:
    if x is None: return None
    s = str(x).strip()
    if not s: return None
    ch = s[0]
    if ch in "([{": ch = s[1] if len(s) > 1 else ch
    if ch.isalpha(): ch = ch.upper()
    return ch

def token_letters(token: str) -> List[str]:
    if token in ("T0","T1"): return list("ABCD")
    if token == "T2": return list("1234")
    return list("ABCD")

def letter_to_idx(token: str, letter: str) -> Optional[int]:
    if token == "T2":
        ch = str(letter).strip()[:1] if letter else None
    else:
        ch = normalize_letter(letter)
    if not ch: return None
    L = token_letters(token)
    return L.index(ch) if ch in L else None

def fold_probs(arr: np.ndarray, n_opts: int) -> np.ndarray:
    p = np.array(arr, dtype=float)
    if p.ndim == 2: p = p.mean(axis=0)
    if p.ndim == 1 and p.size == 2*n_opts:
        p = p.reshape(2, n_opts).sum(axis=0)
    p = np.clip(p, 1e-12, None); p /= p.sum()
    return p

def locate_jsonl(root: str, token: str, dataset: str, model: str) -> Optional[Path]:
    base = Path(root) / token / dataset / model
    c1 = base / f"{dataset}.jsonl"
    c2 = base / dataset / f"{dataset}.jsonl"
    if c1.is_file(): return c1
    if c2.is_file(): return c2
    return None

def collect_models(root: str, tokens: List[str], datasets: List[str]) -> List[str]:
    mset = set()
    for t in tokens:
        for ds in datasets:
            base = Path(root) / t / ds
            if base.is_dir():
                for m in base.iterdir():
                    if m.is_dir(): mset.add(m.name)
    return sorted(mset)

def filter_models(all_models: List[str], wants: Optional[List[str]]) -> List[str]:
    if not wants: return all_models
    wants_s = [safe_tag(w).lower() for w in wants]
    wants_r = [w.lower() for w in wants]
    out = []
    for m in all_models:
        ml = m.lower()
        if any(w in ml for w in wants_s) or any(w in ml for w in wants_r):
            out.append(m)
    return sorted(set(out))

def load_records(fp: Optional[Path]) -> List[dict]:
    if not fp or not fp.is_file(): return []
    rows = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            try:
                rows.append(flatten_record(json.loads(s)))
            except Exception:
                continue
    return rows

def parse_one_rec(rec: dict, token: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[float], Optional[bool]]:
    qid = rec.get("qid") or rec.get("id") or rec.get("uid") or rec.get("question_id")
    if qid is None and "idx" in rec: qid = str(rec["idx"])
    if qid is None: return None, None, None, None, None

    n_opts = len(token_letters(token))

    # gold
    gold_idx = None
    g_letter = rec.get("ideal") or rec.get("gold_letter") or rec.get("label_letter") or rec.get("answer") \
               or rec.get("gold") or rec.get("label") or rec.get("solution") or rec.get("target")
    gi = letter_to_idx(token, g_letter) if g_letter is not None else None
    if gi is not None:
        gold_idx = gi
    else:
        for k in ("gold_idx","label_idx","correct_idx","target_idx","solution_idx","answer_idx"):
            if k in rec:
                try:
                    vi = int(rec[k]); 
                    if vi >= 0: gold_idx = vi; break
                except Exception: pass

    # pred
    pred_idx = None
    p_letter = rec.get("sampled") or rec.get("pred_letter") or rec.get("prediction_letter") or rec.get("model_answer") \
               or rec.get("choice") or rec.get("selected") or rec.get("output") or rec.get("final_answer") \
               or rec.get("pred") or rec.get("prediction") or rec.get("answer")
    pi = letter_to_idx(token, p_letter) if p_letter is not None else None
    if pi is not None:
        pred_idx = pi
    else:
        for k in ("pred_idx","prediction_idx","answer_idx","choice_idx","selected_idx","output_idx","model_pred_idx"):
            if k in rec:
                try:
                    vi = int(rec[k]); 
                    if vi >= 0: pred_idx = vi; break
                except Exception: pass
        if pred_idx is None and "probs" in rec:
            p = fold_probs(rec["probs"], n_opts)
            pred_idx = int(np.argmax(p))

    # conf
    top1_conf = None
    if "probs" in rec:
        p = fold_probs(rec["probs"], n_opts)
        top1_conf = float(p.max())

    # correct
    cf = rec.get("correct", None)
    if isinstance(cf, str): cf = cf.lower() in ("true","1","yes")
    if isinstance(cf, bool):
        correct_flag = cf
    else:
        correct_flag = (pred_idx is not None and gold_idx is not None and pred_idx == gold_idx)

    return str(qid), pred_idx, gold_idx, top1_conf, correct_flag

def load_arrays_for_token(root: str, token: str, dataset: str, model: str):
    fp = locate_jsonl(root, token, dataset, model)
    rows = load_records(fp)
    qids, preds, golds, confs, rights = [], [], [], [], []
    for r in rows:
        qid, pi, gi, conf, cf = parse_one_rec(r, token)
        if qid is None or pi is None or gi is None or conf is None: 
            continue
        qids.append(qid); preds.append(int(pi)); golds.append(int(gi))
        confs.append(float(conf)); rights.append(1 if cf else 0)
    return np.array(qids), np.array(preds), np.array(golds), np.array(confs), np.array(rights, dtype=int)

# =========================
#  ECE/그림
# =========================
def ece_from_conf(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0., 1., n_bins + 1)
    N = len(conf); e = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (conf >= lo) & (conf <= hi) if b == 0 else ((conf > lo) & (conf <= hi))
        if not np.any(mask): continue
        e += (mask.sum()/N) * abs(correct[mask].mean() - conf[mask].mean())
    return float(e)

def _hist(ax, vec, *, color, label, bins, xlim, alpha):
    ax.hist(
        vec, bins=bins, range=xlim, density=True,
        alpha=alpha, color=color, label=label,
        edgecolor="white", linewidth=0.6
    )

def plot_hist_correct_wrong(model: str, dataset: str, conf_by_tok: Dict[str, np.ndarray],
                            right_by_tok: Dict[str, np.ndarray], outdir: Path,
                            bins=20, per_model_dir=False, xlim=(0.0,1.0),
                            alpha_correct=0.80, alpha_wrong=0.55):
    base = outdir / safe_tag(model) if per_model_dir else outdir
    ensure_dir(base)

    data_ok, data_bad = {}, {}
    for t, conf in conf_by_tok.items():
        if conf.size == 0: continue
        right = right_by_tok[t]
        data_ok[t]  = conf[right==1]
        data_bad[t] = conf[right==0]

    if not data_ok and not data_bad: return

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for t, vec in data_ok.items():
        if vec.size == 0: continue
        _hist(axL, vec, color=BASE_COL.get(t, "#808080"),
              label=f"{TOKEN_LABEL.get(t,t)} (n={len(vec)})",
              bins=bins, xlim=xlim, alpha=alpha_correct)
    axL.set_title("Correct only"); axL.set_xlabel("Confidence (top-1 prob)"); axL.set_ylabel("Density"); axL.legend(fontsize=9)

    for t, vec in data_bad.items():
        if vec.size == 0: continue
        _hist(axR, vec, color=lighten(BASE_COL.get(t, "#808080"), 0.55),
              label=f"{TOKEN_LABEL.get(t,t)} (n={len(vec)})",
              bins=bins, xlim=xlim, alpha=alpha_wrong)
    axR.set_title("Wrong only"); axR.set_xlabel("Confidence (top-1 prob)"); axR.legend(fontsize=9)

    fig.suptitle(f"{model} — {dataset} | Confidence (correct vs wrong)", fontsize=14)
    out_png = base / f"{safe_tag(model)}_{dataset}_conf_hist_split.png"
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out_png, dpi=220); plt.close(fig)
    print(f"[SAVE] {out_png}")

def plot_hist_combined_overlay(model: str, dataset: str, conf_by_tok: Dict[str, np.ndarray],
                               right_by_tok: Dict[str, np.ndarray], outdir: Path,
                               bins=20, per_model_dir=False, xlim=(0.0,1.0),
                               alpha_correct=0.75, alpha_wrong=0.50):
    base = outdir / safe_tag(model) if per_model_dir else outdir
    ensure_dir(base)
    fig = plt.figure()
    any_data = False
    for t, conf in conf_by_tok.items():
        if conf.size == 0: continue
        any_data = True
        right = right_by_tok[t]
        ok  = conf[right==1]; bad = conf[right==0]
        base_col  = BASE_COL.get(t, "#808080")
        light_col = lighten(base_col, 0.55)
        plt.hist(bad, bins=bins, range=xlim, alpha=alpha_wrong, color=light_col, density=True,
                 label=f"{TOKEN_LABEL.get(t,t)} wrong (n={len(bad)})", edgecolor="white", linewidth=0.6)
        plt.hist(ok,  bins=bins, range=xlim, alpha=alpha_correct, color=base_col, density=True,
                 label=f"{TOKEN_LABEL.get(t,t)} correct (n={len(ok)})", edgecolor="white", linewidth=0.6)
    if any_data:
        plt.xlabel("Confidence (top-1 prob)"); plt.ylabel("Density")
        plt.title(f"{model} — {dataset} | Confidence distribution by token")
        plt.legend(ncol=2, fontsize=9)
        out_png = base / f"{safe_tag(model)}_{dataset}_conf_hist_overlay.png"
        plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()
        print(f"[SAVE] {out_png}")

def plot_reliability(model: str, dataset: str, conf_by_tok: Dict[str, np.ndarray],
                     right_by_tok: Dict[str, np.ndarray], outdir: Path,
                     n_bins=15, per_model_dir=False):
    base = outdir / safe_tag(model) if per_model_dir else outdir
    ensure_dir(base)
    fig = plt.figure()
    xs = np.linspace(0,1,101)
    plt.plot(xs, xs, "--", color="#555555", linewidth=1.1, label="Ideal")
    for t, conf in conf_by_tok.items():
        if conf.size == 0: continue
        right = right_by_tok[t]
        bins = np.linspace(0.,1.,n_bins+1)
        mids = 0.5*(bins[:-1]+bins[1:])
        accs = []
        for b in range(n_bins):
            lo,hi = bins[b], bins[b+1]
            mask = (conf >= lo) & (conf <= hi) if b==0 else ((conf>lo)&(conf<=hi))
            accs.append(right[mask].mean() if np.any(mask) else np.nan)
        plt.plot(mids, np.array(accs, float), marker="o", linewidth=2,
                 color=BASE_COL.get(t,"#808080"), label=f"{TOKEN_LABEL.get(t,t)}")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.ylim(0,1)
    plt.title(f"{model} — {dataset} | Reliability by token"); plt.legend()
    out_png = base / f"{safe_tag(model)}_{dataset}_reliability.png"
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)
    print(f"[SAVE] {out_png}")

# =========================
#  Matching Ratio (MR)
# =========================
def mr_matrix(preds_by_tok: Dict[str, Dict[str,int]],
              mask_qids: Optional[set]=None,
              tok_order: Optional[List[str]]=None):
    order = tok_order or sorted(preds_by_tok.keys())
    V = len(order)
    M = np.zeros((V,V), dtype=float)
    for i, ti in enumerate(order):
        for j, tj in enumerate(order):
            Qi = set(preds_by_tok[ti].keys())
            Qj = set(preds_by_tok[tj].keys())
            keys = Qi & Qj
            if mask_qids is not None:
                keys &= mask_qids
            if not keys:
                M[i,j] = np.nan
            else:
                same = sum(1 for q in keys if preds_by_tok[ti][q] == preds_by_tok[tj][q])
                M[i,j] = same / float(len(keys))
    return order, M

def _plot_matrix(M: np.ndarray, order_tokens: List[str], title: str, out_png: Path,
                 vmin=0.0, vmax=1.0, fmt="{:.1%}"):
    fig, ax = plt.subplots()
    tick_labels = [TOKEN_LABEL.get(t, t) for t in order_tokens]
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap="viridis", aspect="equal",
                   interpolation="none")
    ax.set_xticks(range(len(tick_labels))); ax.set_yticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels);       ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Token"); ax.set_ylabel("Token")
    ax.set_title(title)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                val = M[i, j]
                ax.text(j, i, fmt.format(val),
                        ha="center", va="center",
                        color=("black" if val > 0.6 else "white"),
                        fontsize=11)
    cbar = fig.colorbar(im, ax=ax); cbar.outline.set_visible(False)
    fig.tight_layout(); ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=220); plt.close(fig); print(f"[SAVE] {out_png}")

def plot_mr_triplet(model: str, dataset: str, tokens: List[str],
                    preds_by_tok: Dict[str, Dict[str,int]],
                    correct_by_tok: Dict[str, Dict[str,int]],
                    outdir: Path, per_model_dir=False):
    base = outdir / safe_tag(model) if per_model_dir else outdir
    ensure_dir(base)
    order = tokens
    _, M_all = mr_matrix(preds_by_tok, tok_order=order)
    S_corr = None
    for t in order:
        cur = {q for q,v in correct_by_tok[t].items() if v==1}
        S_corr = cur if S_corr is None else (S_corr & cur)
    _, M_corr = mr_matrix(preds_by_tok, mask_qids=S_corr, tok_order=order)
    S_wrong = None
    for t in order:
        cur = {q for q,v in correct_by_tok[t].items() if v==0}
        S_wrong = cur if S_wrong is None else (S_wrong & cur)
    _, M_wrong = mr_matrix(preds_by_tok, mask_qids=S_wrong, tok_order=order)
    _plot_matrix(M_all,  order, f"Matching Ratio — {model} / {dataset} (all)",
                 base / f"{safe_tag(model)}_{dataset}_mr_all.png")
    _plot_matrix(M_corr, order, f"Matching Ratio — {model} / {dataset} (correct)",
                 base / f"{safe_tag(model)}_{dataset}_mr_correct.png")
    _plot_matrix(M_wrong,order, f"Matching Ratio — {model} / {dataset} (wrong)",
                 base / f"{safe_tag(model)}_{dataset}_mr_wrong.png")

# =========================
#  메인
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="+", default=["arc","csqa"])
    ap.add_argument("--tokens",   nargs="+", default=["T0_cyclic","T1_cyclic","T2_cyclic"])
    ap.add_argument("--models",   nargs="*", default=None)
    ap.add_argument("--outdir",   default="viz_out/ece")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--n_bins_ece", type=int, default=15)
    ap.add_argument("--per_model_dir", action="store_true")
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=1.0)
    ap.add_argument("--skip_reliability", action="store_true")
    ap.add_argument("--also_combined", action="store_true")
    # ✅ 투명도 옵션
    ap.add_argument("--alpha_correct", type=float, default=0.80,
                    help="정답 히스토그램 막대 투명도(0~1). 값이 작을수록 더 투명")
    ap.add_argument("--alpha_wrong",   type=float, default=0.55,
                    help="오답 히스토그램 막대 투명도(0~1). 값이 작을수록 더 투명")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir); ensure_dir(outdir)
    xlim = (args.xmin, args.xmax)

    auto_models = collect_models(args.root, args.tokens, args.datasets)
    models = filter_models(auto_models, args.models)
    if not models:
        print("[WARN] 모델 디렉토리를 찾지 못했습니다."); return

    tsv_lines = ["model\tdataset\ttoken\tn\tacc\tece\n"]

    for model in models:
        for ds in args.datasets:
            conf_by_tok: Dict[str, np.ndarray] = {}
            right_by_tok_arr: Dict[str, np.ndarray] = {}
            preds_by_tok: Dict[str, Dict[str,int]] = {}
            right_by_tok_qid: Dict[str, Dict[str,int]] = {}

            for t in args.tokens:
                qid, pred, gold, conf, right = load_arrays_for_token(args.root, t, ds, model)
                conf_by_tok[t]  = conf
                right_by_tok_arr[t] = right
                preds_by_tok[t]   = {q:int(pi) for q,pi in zip(qid, pred)}
                right_by_tok_qid[t] = {q:int(r) for q,r in zip(qid, right)}

                if conf.size > 0:
                    ece_val = ece_from_conf(conf, right, n_bins=args.n_bins_ece)
                    tsv_lines.append(f"{model}\t{ds}\t{t}\t{len(conf)}\t{right.mean():.4f}\t{ece_val:.4f}\n")

            # 히스토그램(정답/오답 분리)
            plot_hist_correct_wrong(
                model, ds, conf_by_tok, right_by_tok_arr, outdir,
                bins=args.bins, per_model_dir=args.per_model_dir, xlim=xlim,
                alpha_correct=args.alpha_correct, alpha_wrong=args.alpha_wrong
            )
            if args.also_combined:
                plot_hist_combined_overlay(
                    model, ds, conf_by_tok, right_by_tok_arr, outdir,
                    bins=args.bins, per_model_dir=args.per_model_dir, xlim=xlim,
                    alpha_correct=max(0.05, min(0.95, args.alpha_correct-0.05)),
                    alpha_wrong=max(0.05, min(0.95, args.alpha_wrong-0.05))
                )

            if not args.skip_reliability:
                plot_reliability(model, ds, conf_by_tok, right_by_tok_arr, outdir,
                                 n_bins=args.n_bins_ece, per_model_dir=args.per_model_dir)

            plot_mr_triplet(model, ds, args.tokens, preds_by_tok, right_by_tok_qid,
                            outdir, per_model_dir=args.per_model_dir)

            if len(args.tokens) >= 2:
                first = args.tokens[0]
                for other in args.tokens[1:]:
                    inter = set(preds_by_tok[first]).intersection(preds_by_tok[other])
                    if inter:
                        same = sum(1 for q in inter if preds_by_tok[first][q]==preds_by_tok[other][q])
                        if same == len(inter):
                            print(f"[WARN] {model}/{ds}: {first} vs {other} 교집합 {len(inter)}개에서 예측 100% 동일. (경로/파서 점검 권장)")

    tsv_path = outdir / "ece_summary.tsv"
    ensure_dir(tsv_path.parent)
    with tsv_path.open("w", encoding="utf-8") as w:
        w.writelines(tsv_lines)
    print(f"[DONE] Saved: {tsv_path}")

if __name__ == "__main__":
    main()
