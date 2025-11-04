#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Answer-match matrix (rows=models, cols=datasets).

기능
- 기준 선택(--mode):
  * correct  : "정답 맞춘 qid 집합" 간 겹침(Jaccard/overlap/inter)  [기존 동작]
  * pred_idx : 같은 qid에서 예측 인덱스가 일치한 비율
  * pred_letter : 같은 qid에서 예측 '문자(A/B/C/D, a/b/c/d, 1/2/3/4 등)'가 일치한 비율
- 입력: <root>/<dataset>/<model>/<T*/P0.jsonl>
- 토큰쌍: --pair T0 T1  (또는 --pair ALL 로 T0,T1,T2 모든 쌍 평균)
- 지표(--metric): jaccard(기본), overlap, inter   ※ correct 모드에서만 사용
- 디버그: --verbose, --dry-run, --write_debug <json>
- 모델필터: --models <substr ...>
- 토큰-문자 매핑: --token_map "T0:ABCD,T1:abcd,T2:1234" (pred_letter 모드에 필요)
"""
import os, sys, json, argparse, numpy as np
import traceback
from pathlib import Path

# --- Matplotlib 안전 백엔드 ---
import matplotlib
if os.environ.get("MPLBACKEND") is None:
    os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt

# ---------- 유틸 ----------
def die(msg, code=2):
    print(f"[FATAL] {msg}", flush=True)
    raise SystemExit(code)

def excepthook(exc_type, exc, tb):
    print("[EXCEPTION] Uncaught exception!", flush=True)
    traceback.print_exception(exc_type, exc, tb)
    os._exit(1)
sys.excepthook = excepthook

def read_jsonl(path):
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

def simple_model_name(raw):
    s = (raw.replace("meta-llama_", "Meta ").replace("google_", "Google ")
             .replace("Qwen_", "Qwen ").replace("naver-hyperclovax_", "Naver ")
             .replace("kakaocorp_", "Kakao ").replace("LGAI-EXAONE_", "LG AI ")
             .replace("K-intelligence_", "KT ").replace("_", " "))
    return s.strip()

def filter_models(models, substrs):
    if not substrs:
        return models
    keep=[]
    for m in models:
        for s in substrs:
            if s in m:
                keep.append(m); break
    return sorted(set(keep))

# ---------- 지표 ----------
def jaccard(A,B):
    if not A and not B: return np.nan
    return len(A & B)/len(A | B) if (A|B) else np.nan

def overlap_coeff(A,B):
    if not A or not B: return np.nan
    return len(A & B)/min(len(A),len(B))

def inter_ratio(A,B):
    if not A: return np.nan
    return len(A & B)/len(A)

# ---------- 레코드 파싱 ----------
def parse_pred_gold(rec):
    """
    반환: (qid:str, pred_idx:int|None, gold_idx:int|None)
    ※ 인덱스 기반만 추출. 문자(A/B/C/...)는 별도로 매핑해서 복원.
    """
    qid = rec.get("qid", rec.get("id"))
    if qid is None:
        return None, None, None
    pred = (rec.get("pred_idx") or rec.get("prediction_idx")
            or rec.get("pred") or rec.get("prediction"))
    gold = (rec.get("gold_idx") or rec.get("label_idx")
            or rec.get("label") or rec.get("gold"))
    try:
        pred = int(pred) if pred is not None else None
    except Exception:
        pred = None
    try:
        gold = int(gold) if gold is not None else None
    except Exception:
        gold = None
    return str(qid), pred, gold

# ---------- 토큰-문자 매핑 ----------
def parse_token_map(arg_str, default_map=None):
    """
    "T0:ABCD,T1:abcd,T2:1234" -> {"T0": ["A","B","C","D"], "T1":[...], "T2":[...]}
    """
    if default_map is None:
        default_map = {"T0": list("ABCD"), "T1": list("abcd"), "T2": list("1234")}
    if not arg_str:
        return default_map
    m = {}
    for chunk in arg_str.split(","):
        if not chunk.strip(): continue
        if ":" not in chunk: continue
        t, letters = chunk.split(":", 1)
        t = t.strip()
        letters = [ch for ch in list(letters.strip())]
        m[t] = letters
    # 섞어서 반환(명시된 건 override)
    base = dict(default_map)
    base.update(m)
    return base

def idx_to_letter(token_name, idx, token_map):
    letters = token_map.get(token_name)
    if letters is None: 
        return None
    if not isinstance(idx, int): 
        return None
    if 0 <= idx < len(letters):
        return letters[idx]
    return None

# ---------- 로딩 ----------
def load_for_token(dir_model, token_name, mode, token_map, verbose=False):
    """
    mode에 따라 반환 형태가 다름:
      - "correct"    -> set[qid] (정답 맞춘 qid 집합)
      - "pred_idx"   -> dict[qid] = pred_idx (int)
      - "pred_letter"-> dict[qid] = pred_letter (str)
    """
    jf = os.path.join(dir_model, token_name, "P0.jsonl")
    if verbose:
        print(f"  [SCAN] {jf} exists={os.path.exists(jf)}", flush=True)
    if not os.path.exists(jf):
        return set() if mode=="correct" else {}

    rows = read_jsonl(jf)
    if mode == "correct":
        ok = set()
        for rec in rows:
            qid, pred, gold = parse_pred_gold(rec)
            if qid is None or pred is None or gold is None: 
                continue
            if pred == gold:
                ok.add(qid)
        if verbose:
            print(f"  [OKSET] token={token_name} correct={len(ok)}", flush=True)
        return ok

    # pred 기반
    out = {}
    for rec in rows:
        qid, pred, _ = parse_pred_gold(rec)
        if qid is None or pred is None: 
            continue
        if mode == "pred_idx":
            out[qid] = pred
        elif mode == "pred_letter":
            ch = idx_to_letter(token_name, pred, token_map)
            # 매핑 실패 시 건너뜀(혹은 pred 그대로 보관하려면 str(pred))
            if ch is not None:
                out[qid] = ch
    if verbose:
        print(f"  [PRED] token={token_name} loaded={len(out)} mode={mode}", flush=True)
    return out

# ---------- 페어 계산 ----------
def pairwise_pairs(tokens):
    out=[]
    for i in range(len(tokens)):
        for j in range(i+1,len(tokens)):
            out.append((tokens[i],tokens[j]))
    return out

def compute_pair_value(objA, objB, mode, metric):
    """
    mode에 따라 objA/objB 해석:
      - correct: A,B는 set ; metric은 jaccard/overlap/inter
      - pred_* : A,B는 dict[qid]->value ; 반환은 교집합에서 value 동일한 비율
    """
    if mode == "correct":
        if metric == "jaccard":      return jaccard(objA, objB)
        elif metric == "overlap":    return overlap_coeff(objA, objB)
        elif metric == "inter":      return inter_ratio(objA, objB)
        else: return np.nan

    # pred_* : equality rate on intersection
    keys = set(objA.keys()) & set(objB.keys())
    if not keys: 
        return np.nan
    same = sum(1 for k in keys if objA[k] == objB[k])
    return same / float(len(keys))

# ---------- 메인 ----------
def main():
    print(f"[BOOT] python={sys.version.split()[0]} file={__file__}", flush=True)
    print(f"[BOOT] cwd={os.getcwd()}", flush=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="+", default=["arc","csqa"])
    ap.add_argument("--pair", nargs="+", default=["ALL"], help="T0 T1 또는 ALL")
    ap.add_argument("--tokens", nargs="+", default=["T0","T1","T2"])
    ap.add_argument("--mode", choices=["correct","pred_idx","pred_letter"], default="correct")
    ap.add_argument("--metric", choices=["jaccard","overlap","inter"], default="jaccard",
                    help="※ correct 모드에서만 사용됨")
    ap.add_argument("--token_map", default="T0:ABCD,T1:abcd,T2:1234",
                    help="pred_letter 모드용 토큰-문자 매핑. 예: 'T0:ABCD,T1:abcd,T2:1234'")
    ap.add_argument("--out_png", default="viz_out/answer_match_matrix.png")
    ap.add_argument("--out_csv", default="viz_out/answer_match_matrix.csv")
    ap.add_argument("--min_models", type=int, default=1)
    ap.add_argument("--models", nargs="*", default=None, help="부분 문자열로 모델 필터")
    ap.add_argument("--write_debug", default=None, help="중간 결과 JSON 저장 경로")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # 절대경로 로그
    out_png_abs = str(Path(args.out_png).resolve())
    out_csv_abs = str(Path(args.out_csv).resolve())
    token_map = parse_token_map(args.token_map)

    print(f"[ARGS] root={args.root} datasets={args.datasets} pair={args.pair} "
          f"tokens={args.tokens} mode={args.mode} metric={args.metric} token_map={args.token_map} "
          f"out_png={args.out_png} out_csv={args.out_csv} "
          f"models_filter={args.models} verbose={args.verbose} dry_run={args.dry_run}",
          flush=True)

    # 모델 수집
    model_set=set(); per_ds_models={}
    for ds in args.datasets:
        dsd=os.path.join(args.root, ds)
        print(f"[SCAN] dataset_dir={dsd} exists={os.path.isdir(dsd)}", flush=True)
        if not os.path.isdir(dsd):
            continue
        ms=[m for m in os.listdir(dsd) if os.path.isdir(os.path.join(dsd,m))]
        per_ds_models[ds]=sorted(ms); model_set.update(ms)
        print(f"[SCAN] dataset={ds} models={per_ds_models[ds]}", flush=True)

    models=sorted(model_set)
    if args.models:
        models = filter_models(models, args.models)
        print(f"[FILTER] models after filter={models}", flush=True)
    print(f"[INFO] total_models={len(models)} → {models}", flush=True)
    if not models:
        die(f"No models under {args.root}/<dataset>/* (after filter)")

    # 토큰쌍
    if len(args.pair)==1 and args.pair[0].upper()=="ALL":
        pairs = pairwise_pairs(args.tokens)
        pair_name = "mean(T-pairs)"
    else:
        if len(args.pair)!=2:
            die("--pair must be two tokens or 'ALL'")
        pairs = [tuple(args.pair)]
        pair_name = f"{args.pair[0]} vs {args.pair[1]}"
    print(f"[INFO] pairs={pairs} ({pair_name})", flush=True)

    # 매트릭스
    M = np.full((len(models), len(args.datasets)), np.nan, float)
    token_file_counts = {t:0 for t in args.tokens}

    for j, ds in enumerate(args.datasets):
        dsd=os.path.join(args.root, ds)
        if not os.path.isdir(dsd): 
            continue
        mods=[m for m in os.listdir(dsd) if os.path.isdir(os.path.join(dsd,m))]
        if len(mods)<args.min_models:
            print(f"[WARN] skip dataset={ds} (models<{args.min_models})", flush=True)
            continue

        for i, model in enumerate(models):
            dir_model=os.path.join(dsd, model)
            if not os.path.isdir(dir_model):
                if args.verbose:
                    print(f"[MISS] model_dir not found here: {dir_model}", flush=True)
                continue

            need_tokens = sorted(set(sum(([a,b] for (a,b) in pairs), [])))
            if args.verbose:
                print(f"[MODEL] ds={ds} model={model} need_tokens={need_tokens}", flush=True)

            cache = {}
            for t in need_tokens:
                if os.path.exists(os.path.join(dir_model, t, "P0.jsonl")):
                    token_file_counts[t] += 1
                cache[t] = load_for_token(dir_model, t, args.mode, token_map, verbose=args.verbose)

            vals=[]
            for a,b in pairs:
                A, B = cache.get(a, set() if args.mode=="correct" else {}), \
                       cache.get(b, set() if args.mode=="correct" else {})
                val = compute_pair_value(A, B, args.mode, args.metric)
                vals.append(val)
                if args.verbose:
                    print(f"  [PAIR] {a} vs {b} | {args.mode} -> {val}", flush=True)

            if vals:
                M[i,j]=float(np.nanmean(vals))
                if args.verbose:
                    print(f"  [FILL] M[{i},{j}]={M[i,j]}", flush=True)

    # 정렬
    row_means=np.nanmean(M, axis=1)
    order=np.argsort(-row_means)
    M=M[order,:]; models_sorted=[models[k] for k in order]
    print(f"[ORDER] row_means(desc)={[(models[k], float(row_means[k])) for k in order]}", flush=True)
    print(f"[INFO] token file counts: {token_file_counts}", flush=True)

    # 디버그 JSON
    if args.write_debug:
        dbg = {
            "root": args.root, "cwd": os.getcwd(),
            "datasets": args.datasets, "tokens": args.tokens,
            "pairs": pairs, "pair_name": pair_name,
            "mode": args.mode, "metric": args.metric,
            "token_map": token_map, "models_sorted": models_sorted,
            "matrix": M.tolist(),
        }
        Path(args.write_debug).parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_debug, "w", encoding="utf-8") as w:
            json.dump(dbg, w, ensure_ascii=False, indent=2)
        print(f"[DEBUG] wrote {args.write_debug}", flush=True)

    if args.dry_run:
        print("[DRY] skip saving PNG/CSV", flush=True)
        return

    # 저장
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    # CSV
    with open(args.out_csv, "w", encoding="utf-8") as w:
        w.write("Model," + ",".join(args.datasets) + "\n")
        for i, m in enumerate(models_sorted):
            row=[simple_model_name(m)] + [("" if np.isnan(M[i,j]) else f"{M[i,j]:.4f}") for j in range(M.shape[1])]
            w.write(",".join(row) + "\n")
    if not (os.path.isfile(args.out_csv) and os.path.getsize(args.out_csv)>0):
        die(f"CSV not written: {args.out_csv}")
    print(f"[DONE] CSV: {args.out_csv} (abs={out_csv_abs})", flush=True)

    # 그림
    title = f"{'Correct-Overlap' if args.mode=='correct' else args.mode} " \
            f"({args.metric if args.mode=='correct' else 'equality'}) — {pair_name}"
    plt.figure(figsize=(8, max(3, 0.35*len(models_sorted)+1)))
    plt.imshow(M, vmin=0.0, vmax=1.0, aspect="auto")
    plt.title(title, fontsize=14, pad=8)
    plt.xticks(range(len(args.datasets)), [ds.upper() for ds in args.datasets])
    plt.yticks(range(len(models_sorted)), [simple_model_name(m) for m in models_sorted])
    plt.colorbar()
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i,j]):
                plt.text(j, i, f"{100*M[i,j]:.1f}%", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=180)
    plt.close()
    if not (os.path.isfile(args.out_png) and os.path.getsize(args.out_png)>0):
        die(f"PNG not written: {args.out_png}")
    print(f"[DONE] PNG: {args.out_png} (abs={out_png_abs})", flush=True)
    print(f"[INFO] Root={args.root} Mode={args.mode} Pair={pair_name} Metric={args.metric}", flush=True)

if __name__ == "__main__":
    main()
