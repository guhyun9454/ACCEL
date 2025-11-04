#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Answer-match matrix (rows=models, cols=datasets).

- --mode:
  * correct     : "정답 맞춘 qid 집합" 간 겹침(Jaccard/overlap/inter)
  * pred_idx    : 같은 qid에서 예측 인덱스가 일치한 비율
  * pred_letter : 같은 qid에서 예측 '문자(A/B/C/D, a/b/c/d, 1/2/3/4 등)'가 일치한 비율
- 입력: <root>/<dataset>/<model>/<T*/P0.jsonl>
- --pair T0 T1  또는 --pair ALL (T0,T1,T2 모든 쌍 평균)
- --metric: jaccard/overlap/inter   ※ correct 모드에서만 사용
- --token_map "T0:ABCD,T1:abcd,T2:1234"
- --auto_token_map: P0.jsonl의 choices를 스캔해 토큰별 레터 자동추론
"""

import os, sys, json, argparse, numpy as np
import traceback
from pathlib import Path

# --- Matplotlib 백엔드 (헤드리스 안전) ---
import matplotlib
if os.environ.get("MPLBACKEND") is None:
    os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt

# ----------------- 유틸 -----------------
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

# --------------- 지표 -------------------
def jaccard(A,B):
    if not A and not B: return np.nan
    return len(A & B)/len(A | B) if (A|B) else np.nan

def overlap_coeff(A,B):
    if not A or not B: return np.nan
    return len(A & B)/min(len(A),len(B))

def inter_ratio(A,B):
    if not A: return np.nan
    return len(A & B)/len(A)

# ----------- 레터 정규화 & 추출 ----------
def normalize_letter(x: str):
    """'a'->'A', ' A.'->'A', '(b'->'B', '1)'->'1' 등 첫 글자 표준화"""
    if x is None:
        return None
    if not isinstance(x, str):
        x = str(x)
    x = x.strip()
    if not x:
        return None
    ch = x[0]
    if ch in "([{":
        ch = x[1] if len(x) > 1 else ch
    if ch.isalpha():
        ch = ch.upper()
    return ch

def get_first_str(rec, keys):
    for k in keys:
        if k in rec and rec[k] is not None:
            v = rec[k]
            if isinstance(v, (str,int)):
                return str(v)
    return None

def get_pred_letter_from_rec(rec):
    keys = ("pred_letter","prediction_letter","answer","model_answer",
            "choice","selected","output","final_answer","pred","prediction")
    s = get_first_str(rec, keys)
    return normalize_letter(s) if s else None

def get_gold_letter_from_rec(rec):
    keys = ("gold_letter","label_letter","answer","gold","label","solution",
            "target","correct_letter")
    s = get_first_str(rec, keys)
    return normalize_letter(s) if s else None

# ----------- 레코드 파싱 ----------------
def parse_pred_gold(rec):
    """
    반환: (qid, pred_idx, gold_idx, pred_letter, gold_letter)
    - 인덱스/문자 모두 시도 (음수 인덱스는 None 처리)
    """
    qid = rec.get("qid", rec.get("id", rec.get("uid", rec.get("question_id"))))
    if qid is None:
        return None, None, None, None, None

    idx_keys_pred = ("pred_idx","prediction_idx","answer_idx","choice_idx",
                     "selected_idx","output_idx","model_pred_idx","pred","prediction")
    idx_keys_gold = ("gold_idx","label_idx","answer_idx","correct_idx",
                     "solution_idx","target_idx","gold","label","answer")

    def to_int_or_none(v):
        try:
            iv = int(v)
            return iv if iv >= 0 else None
        except Exception:
            return None

    pred_idx = None
    gold_idx = None
    for k in idx_keys_pred:
        if k in rec:
            pred_idx = to_int_or_none(rec[k])
            if pred_idx is not None: break
    for k in idx_keys_gold:
        if k in rec:
            gold_idx = to_int_or_none(rec[k])
            if gold_idx is not None: break

    p_letter = get_pred_letter_from_rec(rec)
    g_letter = get_gold_letter_from_rec(rec)

    return str(qid), pred_idx, gold_idx, p_letter, g_letter

# ------- 토큰-문자 매핑 관련 ------------
def parse_token_map(arg_str, default_map=None):
    """
    "T0:ABCD,T1:abcd,T2:1234" -> {"T0": ["A","B","C","D"], ...}
    """
    if default_map is None:
        default_map = {"T0": list("ABCD"), "T1": list("abcd"), "T2": list("1234")}
    if not arg_str:
        return dict({k:[normalize_letter(c) for c in v] for k,v in default_map.items()})
    m = {}
    for chunk in arg_str.split(","):
        if not chunk.strip(): continue
        if ":" not in chunk: continue
        t, letters = chunk.split(":", 1)
        t = t.strip()
        letters = [normalize_letter(ch) for ch in list(letters.strip())]
        letters = [ch for ch in letters if ch]
        if letters:
            m[t] = letters
    base = dict({k:[normalize_letter(c) for c in v] for k,v in default_map.items()})
    base.update(m)  # 명시값 우선
    return base

def _extract_letters_from_choices_field(choices):
    """choices의 각 항목에서 label/letter/문자 접두로 레터 추출"""
    letters = []
    for item in choices:
        ch = None
        if isinstance(item, dict):
            for k in ("label","letter","option","id","name","key","text"):
                if k in item and isinstance(item[k], str) and item[k]:
                    ch = normalize_letter(item[k]); break
        elif isinstance(item, str) and item:
            ch = normalize_letter(item)
        if ch:
            letters.append(ch)
    letters = [c for c in letters if isinstance(c, str) and len(c)==1]
    return letters or None

def infer_token_letters_from_rows(rows):
    for rec in rows[:100]:  # 조금 더 넉넉히
        for key in ("choices","options","option_list"):
            if key in rec and isinstance(rec[key], list) and rec[key]:
                got = _extract_letters_from_choices_field(rec[key])
                if got:
                    return got
    return None

def idx_to_letter(token_name, idx, token_map):
    letters = token_map.get(token_name)
    if letters is None or not isinstance(idx, int):
        return None
    return letters[idx] if 0 <= idx < len(letters) else None

# --------------- 로딩 -------------------
def load_for_token(dir_model, token_name, mode, token_map, auto_token_map=False, verbose=False):
    """
    mode별 반환:
      - "correct"     -> set[qid]
      - "pred_idx"    -> dict[qid] = int
      - "pred_letter" -> dict[qid] = str
    """
    jf = os.path.join(dir_model, token_name, "P0.jsonl")
    if verbose:
        print(f"  [SCAN] {jf} exists={os.path.exists(jf)}", flush=True)
    if not os.path.exists(jf):
        return set() if mode=="correct" else {}

    rows = read_jsonl(jf)

    # 필요시 토큰 레터 자동 추정
    if auto_token_map and token_name not in token_map:
        guessed = infer_token_letters_from_rows(rows)
        if guessed:
            token_map[token_name] = guessed
            if verbose:
                print(f"  [AUTOMAP] token={token_name} letters={guessed}", flush=True)
        elif verbose:
            print(f"  [AUTOMAP] token={token_name} letters NOT inferred", flush=True)

    if mode == "correct":
        ok = set()
        c_total = 0
        for rec in rows:
            qid, pred_idx, gold_idx, p_letter, g_letter = parse_pred_gold(rec)
            if qid is None:
                continue
            decided = None
            if pred_idx is not None and gold_idx is not None:
                decided = (pred_idx == gold_idx)
            else:
                if p_letter and g_letter:
                    decided = (normalize_letter(p_letter) == normalize_letter(g_letter))
                elif (pred_idx is not None) and (g_letter is not None) and (token_name in token_map):
                    # gold가 문자, pred는 인덱스 → 문자로 비교
                    ch = idx_to_letter(token_name, pred_idx, token_map)
                    decided = (ch is not None and normalize_letter(ch) == normalize_letter(g_letter))
            if decided:
                ok.add(qid)
                c_total += 1
        if verbose:
            print(f"  [OKSET] token={token_name} correct={len(ok)}", flush=True)
        return ok

    out = {}
    took_idx = took_letter = 0
    for rec in rows:
        qid, pred_idx, _, p_letter, _ = parse_pred_gold(rec)
        if qid is None:
            continue
        if mode == "pred_idx":
            if pred_idx is not None:
                out[qid] = int(pred_idx); took_idx += 1
            elif p_letter and token_name in token_map:
                letters = token_map[token_name]
                ch = normalize_letter(p_letter)
                if ch in letters:
                    out[qid] = int(letters.index(ch)); took_idx += 1
        elif mode == "pred_letter":
            if p_letter:
                out[qid] = normalize_letter(p_letter); took_letter += 1
            elif pred_idx is not None and token_name in token_map:
                ch = idx_to_letter(token_name, pred_idx, token_map)
                if ch is not None:
                    out[qid] = ch; took_letter += 1
    if verbose:
        print(f"  [PRED] token={token_name} loaded={len(out)} mode={mode}", flush=True)
        if token_name in token_map:
            print(f"  [PRED] token_map[{token_name}]={token_map[token_name]}", flush=True)
        if mode == "pred_idx":
            print(f"  [PRED] via idx/letter = {took_idx}", flush=True)
        else:
            print(f"  [PRED] via letter/idx = {took_letter}", flush=True)
    return out

# ----------- 페어 계산 ------------------
def pairwise_pairs(tokens):
    out=[]
    for i in range(len(tokens)):
        for j in range(i+1,len(tokens)):
            out.append((tokens[i],tokens[j]))
    return out

def compute_pair_value(objA, objB, mode, metric):
    if mode == "correct":
        if metric == "jaccard":      return jaccard(objA, objB)
        elif metric == "overlap":    return overlap_coeff(objA, objB)
        elif metric == "inter":      return inter_ratio(objA, objB)
        else: return np.nan
    keys = set(objA.keys()) & set(objB.keys())
    if not keys:
        return np.nan
    same = sum(1 for k in keys if objA[k] == objB[k])
    return same / float(len(keys))

# ---------------- 메인 -------------------
def main():
    print(f"[BOOT] python={sys.version.split()[0]} file={__file__}", flush=True)
    print(f"[BOOT] cwd={os.getcwd()}", flush=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="+", default=["arc","csqa"])
    ap.add_argument("--pair", nargs="+", default=["ALL"])
    ap.add_argument("--tokens", nargs="+", default=["T0","T1","T2"])
    ap.add_argument("--mode", choices=["correct","pred_idx","pred_letter"], default="correct")
    ap.add_argument("--metric", choices=["jaccard","overlap","inter"], default="jaccard")
    ap.add_argument("--token_map", default="T0:ABCD,T1:abcd,T2:1234")
    ap.add_argument("--auto_token_map", action="store_true")
    ap.add_argument("--out_png", default="viz_out/answer_match_matrix.png")
    ap.add_argument("--out_csv", default="viz_out/answer_match_matrix.csv")
    ap.add_argument("--min_models", type=int, default=1)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--write_debug", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_png_abs = str(Path(args.out_png).resolve())
    out_csv_abs = str(Path(args.out_csv).resolve())
    token_map = parse_token_map(args.token_map)

    print(f"[ARGS] root={args.root} datasets={args.datasets} pair={args.pair} "
          f"tokens={args.tokens} mode={args.mode} metric={args.metric} "
          f"token_map={args.token_map} auto_token_map={args.auto_token_map} "
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
                cache[t] = load_for_token(
                    dir_model, t, args.mode, token_map,
                    auto_token_map=args.auto_token_map, verbose=args.verbose
                )

            vals=[]
            for a,b in pairs:
                A = cache.get(a, set() if args.mode=="correct" else {})
                B = cache.get(b, set() if args.mode=="correct" else {})
                val = compute_pair_value(A, B, args.mode, args.metric)
                vals.append(val)
                if args.verbose:
                    print(f"  [PAIR] {a} vs {b} | mode={args.mode} -> {val}", flush=True)

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
            "token_map_effective": {k:v for k,v in token_map.items()},
            "models_sorted": models_sorted,
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
