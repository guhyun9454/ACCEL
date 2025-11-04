#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Answer-match matrix (rows=models, cols=datasets).

- --mode:
  * correct     : "정답 맞춘 qid 집합" 간 겹침(Jaccard/overlap/inter)
  * pred_idx    : 같은 qid에서 예측 인덱스가 일치한 비율
  * pred_letter : 같은 qid에서 예측 '문자(A/B/C/D, a/b/c/d, 1/2/3/4 등)'가 일치한 비율

- 입력 구조(예시):
  <root>/<dataset>/<model>/<T*/P0.jsonl>

- JSONL 지원 포맷:
  1) 기존 평면형 키(rec["qid"], "gold"/"pred", ... )
  2) 최신 결과형: {"type":"result","data":{ "idx","ideal","sampled","correct","probs", ... }}

- --pair T0 T1  또는 --pair ALL (T0,T1,T2 모든 쌍 평균)
- --metric: jaccard/overlap/inter   ※ correct 모드에서만 사용
- --token_map "T0:ABCD,T1:abcd,T2:1234"
- --auto_token_map: P0.jsonl의 choices를 스캔해 토큰별 레터 자동추론(가능한 경우)
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
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    # 방어적 파싱
                    continue
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
    """'a'->'A', ' A.'->'A', '(b'->'B', '1)'->'1' 등 첫 글자 표준화 (한글은 원형 유지)"""
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

# ----------- 최신(JSONL data) 대응 헬퍼 ----------
def flatten_record(rec: dict) -> dict:
    """{"type":"result","data":{...}} → {...} 로 평탄화"""
    if isinstance(rec, dict) and "data" in rec and isinstance(rec["data"], dict):
        flat = dict(rec["data"])
        # qid 보정: 있으면 그대로, 없으면 idx로 대체
        if "qid" not in flat and "idx" in flat:
            flat["qid"] = str(flat["idx"])
        return flat
    # 기존 포맷: 그대로
    return rec

def probs_to_idx_and_letter(probs, letters):
    """probs(list or 2D list)을 평균 후 argmax → (idx, letter)"""
    if probs is None or letters is None or len(letters) == 0:
        return None, None
    p = np.array(probs, dtype=np.float64)
    if p.ndim == 2:  # cyclic/perm: [num_views, num_options]
        p = p.mean(axis=0)
    # 드문 케이스: 길이가 2*num_options (공백/비공백)인 경우 → fold
    if letters and p.ndim == 1 and p.size == 2*len(letters):
        p = p.reshape(2, len(letters)).sum(axis=0)
    if p.ndim != 1 or p.size != len(letters):
        return None, None
    idx = int(np.argmax(p))
    return idx, letters[idx]

# ----------- 레코드 파싱 ----------------
def parse_pred_gold(rec, token_letters=None):
    """
    반환: (qid, pred_idx, gold_idx, pred_letter, gold_letter, correct_flag_or_None)

    - JSONL이 result/data 형태면 평탄화 후 키를 탐색
    - pred는 우선순위: sampled → pred/pred_letter → probs(argmax)
    - gold는 ideal → gold/gold_letter 등
    - correct가 있으면 그대로 반환
    """
    r = flatten_record(rec)

    # qid
    qid = r.get("qid") or r.get("id") or r.get("uid") or r.get("question_id")
    if qid is None and "idx" in r:
        qid = str(r["idx"])
    if qid is None:
        return None, None, None, None, None, None

    # gold (letter 우선)
    g_letter = None
    g_letter = ( get_first_str(r, ("ideal","gold_letter","label_letter","answer","gold","label","solution","target","correct_letter")) )
    g_letter = normalize_letter(g_letter) if g_letter else None

    # pred (letter/idx/확률 역산)
    p_letter = None
    p_idx = None

    # 1) 명시적 정오표기
    correct_flag = r.get("correct", None)
    if isinstance(correct_flag, str):
        if correct_flag.lower() in ("true","yes","1"):
            correct_flag = True
        elif correct_flag.lower() in ("false","no","0"):
            correct_flag = False
        else:
            correct_flag = None

    # 2) 샘플된 문자
    p_letter = ( get_first_str(r, ("sampled","pred_letter","prediction_letter","answer","model_answer",
                                   "choice","selected","output","final_answer","pred","prediction")) )
    p_letter = normalize_letter(p_letter) if p_letter else None

    # 3) probs → argmax
    if p_letter is None and "probs" in r and token_letters:
        idx, ch = probs_to_idx_and_letter(r["probs"], token_letters)
        if ch is not None:
            p_idx = idx
            p_letter = normalize_letter(ch)

    # 4) pred_idx 직접
    if p_idx is None:
        for k in ("pred_idx","prediction_idx","answer_idx","choice_idx","selected_idx",
                  "output_idx","model_pred_idx","pred","prediction"):
            if k in r:
                try:
                    iv = int(r[k])
                    if iv >= 0:
                        p_idx = iv
                except Exception:
                    pass
                break

    # gold_idx (거의 없겠지만 호환)
    g_idx = None
    for k in ("gold_idx","label_idx","answer_idx","correct_idx","solution_idx","target_idx","gold","label","answer"):
        if k in r:
            try:
                iv = int(r[k]); 
                if iv >= 0: g_idx = iv
            except Exception:
                pass
            break

    return str(qid), p_idx, g_idx, p_letter, g_letter, correct_flag

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
    # 최신 포맷은 options가 텍스트라서 자동추론 실패 가능 → best-effort
    for rec in rows[:100]:
        r = flatten_record(rec)
        for key in ("choices","options","option_list"):
            if key in r and isinstance(r[key], list) and r[key]:
                got = _extract_letters_from_choices_field(r[key])
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

    letters = token_map.get(token_name)

    if mode == "correct":
        ok = set()
        for raw in rows:
            qid, p_idx, g_idx, p_letter, g_letter, correct_flag = parse_pred_gold(raw, token_letters=letters)
            if qid is None:
                continue

            decided = None
            # 1) correct 플래그 우선
            if isinstance(correct_flag, bool):
                decided = correct_flag
            # 2) 인덱스 vs 인덱스
            if decided is None and (p_idx is not None) and (g_idx is not None):
                decided = (p_idx == g_idx)
            # 3) 문자 vs 문자
            if decided is None and p_letter and g_letter:
                decided = (normalize_letter(p_letter) == normalize_letter(g_letter))
            # 4) idx→letter 매핑 후 비교
            if decided is None and (p_idx is not None) and letters and g_letter:
                ch = idx_to_letter(token_name, p_idx, token_map)
                decided = (ch is not None and normalize_letter(ch) == normalize_letter(g_letter))

            if decided:
                ok.add(qid)

        if verbose:
            print(f"  [OKSET] token={token_name} correct={len(ok)}", flush=True)
        return ok

    out = {}
    took_idx = took_letter = 0
    for raw in rows:
        qid, p_idx, _, p_letter, _, _ = parse_pred_gold(raw, token_letters=letters)
        if qid is None:
            continue

        if mode == "pred_idx":
            if p_idx is not None:
                out[qid] = int(p_idx); took_idx += 1
            elif p_letter and letters:
                ch = normalize_letter(p_letter)
                if ch in letters:
                    out[qid] = int(letters.index(ch)); took_idx += 1

        elif mode == "pred_letter":
            if p_letter:
                out[qid] = normalize_letter(p_letter); took_letter += 1
            elif (p_idx is not None) and letters:
                ch = idx_to_letter(token_name, p_idx, token_map)
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
