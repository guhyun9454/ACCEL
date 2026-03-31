from __future__ import annotations

import os
import json
import argparse
import glob
import subprocess
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

from utils import normalize_pretrained_model_path_for_fs

# 논문용 깔끔한 스타일 세팅
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def _rotations(k):
    return [tuple((i + s) % k for i in range(k)) for s in range(k)]


def _is_factorial(n: int, m: int) -> bool:
    if n <= 1:
        return m == 1
    prod = 1
    for i in range(2, n + 1):
        prod *= i
        if prod > m:
            return False
    return prod == m


def _cyclic_view_indices(k: int, n_views: int) -> list[int]:
    """
    Return indices of views to use as cyclic rotations (k of them) for aggregation.

    - If n_views == k: assume the stored sequence is already cyclic rotations in order [0..k-1].
    - If n_views == k!: assume eval_clm.py used lexicographically sorted permutations of range(k).
      Then map rotation tuples to their indices in that sorted permutation list.
    - Else: fall back to first min(k, n_views) views (best-effort).
    """
    if n_views <= 0:
        return []
    if n_views == k:
        return list(range(k))
    if _is_factorial(k, n_views):
        # Match eval_clm.py: perm_list = list(sorted(permutations(range(k))))
        from itertools import permutations

        perm_list = list(sorted(permutations(range(k))))
        perm_to_idx = {p: i for i, p in enumerate(perm_list)}
        rots = _rotations(k)
        idxs: list[int] = []
        for r in rots:
            idx = perm_to_idx.get(tuple(r))
            if idx is not None:
                idxs.append(int(idx))
        # Should be length k, but keep it safe
        if len(idxs) > 0:
            return idxs
    # Fallback
    return list(range(min(k, n_views)))

def _gap(probs):
    vals = np.sort(probs)[::-1]
    return vals[0] - vals[1] if len(vals) > 1 else 0.0

def _is_curve_jsonl(path: str) -> bool:
    p = str(path).replace("\\", "/").lower()
    return p.endswith("_curve.jsonl") or p.endswith("_pride_curve.jsonl")


def _expand_inputs(input_path: str) -> list[str]:
    """
    input can be:
    - a jsonl file
    - a directory (collect *.jsonl under it)
    - a glob pattern
    """
    if not input_path:
        return []

    raw = os.path.expandvars(os.path.expanduser(str(input_path)))
    if os.path.isdir(raw):
        cand = glob.glob(os.path.join(raw, "*.jsonl"))
    else:
        if any(ch in raw for ch in ["*", "?", "["]):
            cand = glob.glob(raw)
        else:
            cand = [raw]

    out: list[str] = []
    for p in cand:
        if not str(p).lower().endswith(".jsonl"):
            continue
        if _is_curve_jsonl(p):
            continue
        if os.path.isfile(p):
            out.append(p)
    out = sorted(set(out))
    return out


def _expand_results_dir_jsonl_recursive(save_path: str) -> list[str]:
    """
    eval_clm 결과 폴더(과목별 jsonl이 하위에 있을 수 있음)에서 *.jsonl 재귀 수집.
    """
    raw = os.path.expandvars(os.path.expanduser(str(save_path)))
    if not os.path.isdir(raw):
        return []
    cand = glob.glob(os.path.join(raw, "**", "*.jsonl"), recursive=True)
    out: list[str] = []
    for p in cand:
        if _is_curve_jsonl(p):
            continue
        if os.path.isfile(p):
            out.append(os.path.abspath(p))
    return sorted(set(out))


def _collect_jsonl_from_pretrained_paths(
    code_dir: str,
    results_eval_name: str,
    pretrained_like_paths: list[str],
    option_id_set: str | None,
) -> tuple[list[str], list[str]]:
    """
    각 pretrained 경로에 대해 _eval_save_path 로 디렉터리를 구한 뒤 재귀적으로 jsonl 수집.
    Returns (file_paths, warnings).
    """
    warnings: list[str] = []
    all_files: list[str] = []
    for raw in pretrained_like_paths:
        s = str(raw).strip()
        if not s:
            continue
        try:
            save_path, task, num_few_shot, setting = _eval_save_path(
                code_dir, results_eval_name, s, option_id_set=option_id_set
            )
        except Exception as e:
            warnings.append(f"[skip] {s}: {e}")
            continue
        if not os.path.isdir(save_path):
            warnings.append(f"[skip] no dir for {s}: {save_path}")
            continue
        got = _expand_results_dir_jsonl_recursive(save_path)
        if not got:
            warnings.append(f"[skip] no jsonl under: {save_path} ({s})")
            continue
        print(f"[ok] {s} -> {save_path} ({len(got)} jsonl)")
        all_files.extend(got)
    return sorted(set(all_files)), warnings


def _expand_inputs_multi(base_dirs: list[str], input_path: str | None) -> list[str]:
    """
    Try expanding input_path as-is, then relative to each base_dir (if relative).
    This makes CLI more robust to different current working directories.
    """
    if not input_path:
        return []
    tried: list[str] = []
    out: list[str] = []

    def _try(p: str) -> None:
        nonlocal out
        if not p or p in tried:
            return
        tried.append(p)
        out.extend(_expand_inputs(p))

    raw = str(input_path)
    _try(raw)

    if not os.path.isabs(os.path.expandvars(os.path.expanduser(raw))):
        for bd in base_dirs:
            _try(os.path.join(bd, raw))

    return sorted(set(out))


def _safe_makedirs_for_prefix(out_prefix: str) -> None:
    d = os.path.dirname(out_prefix)
    if d:
        os.makedirs(d, exist_ok=True)


def _sanitize_tag(s: str) -> str:
    s = str(s).strip().replace("\\", "_").replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _model_name_from_pretrained_path(pretrained_model_path: str) -> str:
    p = normalize_pretrained_model_path_for_fs(str(pretrained_model_path))
    return os.path.basename(p.rstrip("/\\")).strip()


def _infer_model_tag_from_inputs(input_files: list[str]) -> str | None:
    """
    Best-effort inference from paths like:
      .../results_csqa/0s_Olmo-3-7B-Instruct/.../*.jsonl
    """
    if not input_files:
        return None
    for p in input_files[:50]:
        parts = str(p).replace("\\", "/").split("/")
        for seg in parts:
            m = re.match(r"^\d+s_(.+)$", seg)
            if m:
                cand = m.group(1)
                if cand:
                    return cand
    return None


def _eval_save_path(
    code_dir: str,
    eval_name: str,
    pretrained_model_path: str,
    option_id_set: str | None = None,
) -> tuple[str, str, int, str | None]:
    parts = [p.strip() for p in str(eval_name).split(",")]
    if len(parts) < 2:
        raise ValueError(f"Invalid eval_name='{eval_name}'. Expected 'task,num_few_shot,setting(optional)'.")
    task = parts[0]
    num_few_shot = int(parts[1])
    setting = parts[2] if len(parts) > 2 and parts[2] else None

    p = normalize_pretrained_model_path_for_fs(pretrained_model_path)
    model_name = os.path.basename(p.rstrip("/\\")).strip()
    save_path = os.path.join(code_dir, f"results_{task}", f"{num_few_shot}s_{model_name}", f"{task}")
    if setting is not None:
        save_path += f"_{setting}"
    if option_id_set:
        save_path += f"_id-{option_id_set}"
    return save_path, task, num_few_shot, setting


def _extract_flag_value(argv: list[str], flag: str) -> str | None:
    """Return the next token after `flag` if present (no '=' support)."""
    try:
        i = argv.index(flag)
    except ValueError:
        return None
    if i + 1 >= len(argv):
        return None
    v = argv[i + 1]
    if isinstance(v, str) and v.startswith("--"):
        return None
    return str(v)


def _validate_option_id_set(eval_names: list[str], extra_args: list[str]) -> None:
    """
    eval_clm_utils.py enforces option_id_set length matches #options:
      - mmlu, arc -> 4 (ABCD)
      - csqa -> 5 (ABCDE)
    If user passes a single global --option_id_set, prevent obvious mismatch early.
    """
    opt = _extract_flag_value(extra_args, "--option_id_set")
    if not opt:
        return

    tasks = []
    for ev in eval_names:
        parts = [p.strip() for p in str(ev).split(",")]
        if len(parts) >= 1 and parts[0]:
            tasks.append(parts[0].lower())

    if "csqa" in tasks and len(opt) != 5:
        raise SystemExit(
            f"--option_id_set '{opt}' has length {len(opt)} but csqa requires 5. "
            f"Use '--option_id_set ABCDE' for csqa, or run csqa separately without a global option_id_set."
        )
    if any(t in ("mmlu", "arc") for t in tasks) and len(opt) != 4 and ("csqa" not in tasks):
        raise SystemExit(
            f"--option_id_set '{opt}' has length {len(opt)} but mmlu/arc require 4. "
            f"Use '--option_id_set ABCD' or omit it."
        )


def _collect_eval_outputs(
    code_dir: str,
    eval_names: list[str],
    pretrained_model_path: str,
    option_id_set: str | None = None,
) -> dict[str, list[str]]:
    """
    Returns mapping: label -> list of jsonl file paths
    label examples: 'mmlu,0,full'
    """
    out: dict[str, list[str]] = {}
    for ev in eval_names:
        save_path, task, num_few_shot, setting = _eval_save_path(
            code_dir, ev, pretrained_model_path, option_id_set=option_id_set
        )
        files = _expand_inputs(save_path)
        out[str(ev)] = files
    return out


def process_jsonl(filepath: str) -> pd.DataFrame:
    """jsonl 파일을 읽어 각 샘플의 Margin, Required T, T-Sensitivity 등을 계산합니다."""
    records: list[dict] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue

            if row.get("type") != "result":
                continue

            data = row.get("data", {}) or {}
            probs_seq = data.get("probs")
            options = data.get("options") or []
            ideal = data.get("ideal", None)
            if not probs_seq or not isinstance(options, list) or len(options) <= 1:
                continue

            k = int(len(options))
            probs_seq_np = np.asarray(probs_seq, dtype=np.float64)
            if probs_seq_np.ndim == 1:
                probs_seq_np = probs_seq_np.reshape(1, -1)
            if probs_seq_np.shape[-1] != k:
                continue

            # 1) Decide which views correspond to cyclic rotations
            view_indices = _cyclic_view_indices(k=k, n_views=int(len(probs_seq_np)))
            if len(view_indices) == 0:
                continue
            identity_idx = int(view_indices[0])

            # 2. 1-view Base 측정
            base_probs = np.asarray(probs_seq_np[identity_idx], dtype=np.float64)
            base_gap = float(_gap(base_probs))
            base_pred = int(np.argmax(base_probs))

            # 3. T-Sensitivity 계산 (k개의 cyclic view에서 gap의 std)
            individual_gaps: list[float] = []
            for idx in view_indices:
                if idx < 0 or idx >= len(probs_seq_np):
                    continue
                view_probs = np.asarray(probs_seq_np[idx], dtype=np.float64)
                individual_gaps.append(float(_gap(view_probs)))
            t_sensitivity = float(np.std(individual_gaps)) if len(individual_gaps) > 0 else 0.0
            mean_cyclic_margin = float(np.mean(individual_gaps)) if len(individual_gaps) > 0 else 0.0

            # 4. T 1~k 누적 평균으로 Required T 계산
            cum_probs = np.zeros(k, dtype=np.float64)
            preds_at_t: list[int] = []
            for t, idx in enumerate(view_indices, start=1):
                if idx < 0 or idx >= len(probs_seq_np):
                    break
                cum_probs += np.asarray(probs_seq_np[int(idx)], dtype=np.float64)
                pred_t = int(np.argmax(cum_probs / float(t)))
                preds_at_t.append(pred_t)

            if len(preds_at_t) == 0:
                continue

            final_pred = int(preds_at_t[-1])  # T=k(or last available) 예측

            required_t = int(len(preds_at_t))
            for t_idx in range(len(preds_at_t)):
                if all(p == final_pred for p in preds_at_t[t_idx:]):
                    required_t = int(t_idx + 1)
                    break

            flips = [int(p != base_pred) for p in preds_at_t]

            final_correct = None
            if ideal is not None and isinstance(ideal, str):
                try:
                    final_correct = bool(final_pred == int(options.index(ideal)))
                except Exception:
                    final_correct = None

            rec: dict = {
                "source_file": os.path.basename(filepath),
                "k": k,
                "base_margin": base_gap,
                "mean_cyclic_margin": mean_cyclic_margin,
                "t_sensitivity": t_sensitivity,
                "required_t": required_t,
                "final_correct": final_correct,
            }
            for t in range(1, len(flips) + 1):
                rec[f"flip_at_T{t}"] = flips[t - 1]
            records.append(rec)

    return pd.DataFrame(records)

def plot_heatmap(df, out_path, k):
    """1. Margin 구간별 T 증가에 따른 Flip Rate 히트맵"""
    # 마진을 10개 구간으로 나눔
    df['Margin Bin'] = pd.cut(df['base_margin'], bins=10, labels=[f"{i*10}-{(i+1)*10}%" for i in range(10)])
    
    heatmap_data = []
    for t in range(2, k + 1): # T=1은 항상 Flip 0이므로 제외
        grp = df.groupby('Margin Bin')[f'flip_at_T{t}'].mean() * 100
        heatmap_data.append(grp)
        
    heat_df = pd.DataFrame(heatmap_data, index=[f"T={t}" for t in range(2, k+1)])
    
    plt.figure(figsize=(10, 4), dpi=200)
    sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="coolwarm", cbar_kws={'label': 'Flip Rate (%)'})
    plt.title("Flip Rate by Margin Bins and Number of Views (T)")
    plt.xlabel("1-View Margin Bins (Low to High Confidence)")
    plt.ylabel("Number of Views (T)")
    plt.tight_layout()
    plt.savefig(f"{out_path}_1_heatmap.png", bbox_inches="tight")
    plt.close()

def plot_required_t(df, out_path):
    """2. 마진 vs Required T 산점도 및 박스플롯"""
    corr, p = spearmanr(df['base_margin'], df['required_t'])
    
    plt.figure(figsize=(8, 6), dpi=200)
    # T는 이산값이므로 Stripplot과 Boxplot 겹치기
    sns.boxplot(x='required_t', y='base_margin', data=df, color="lightblue", showfliers=False)
    sns.stripplot(x='required_t', y='base_margin', data=df, color="black", alpha=0.3, jitter=True)
    
    plt.title(f"Required Views for Stabilization vs. Initial Margin\nSpearman $r$ = {corr:.3f} ($p$ < {p:.1e})")
    plt.xlabel("Required T to Stabilize Prediction")
    plt.ylabel("1-View Margin (Confidence)")
    plt.tight_layout()
    plt.savefig(f"{out_path}_2_required_t.png", bbox_inches="tight")
    plt.close()


def plot_required_t_mean_cyclic(df, out_path):
    """2b. mean cyclic margin vs Required T (same layout as plot_required_t)."""
    corr, p = spearmanr(df["mean_cyclic_margin"], df["required_t"])

    plt.figure(figsize=(8, 6), dpi=200)
    sns.boxplot(x="required_t", y="mean_cyclic_margin", data=df, color="lightgreen", showfliers=False)
    sns.stripplot(x="required_t", y="mean_cyclic_margin", data=df, color="black", alpha=0.3, jitter=True)

    plt.title(
        f"Required Views for Stabilization vs. Mean Cyclic Margin\n"
        f"Spearman $r$ = {corr:.3f} ($p$ < {p:.1e})"
    )
    plt.xlabel("Required T to Stabilize Prediction")
    plt.ylabel("Mean margin across cyclic views")
    plt.tight_layout()
    plt.savefig(f"{out_path}_2_required_t_mean_cyclic.png", bbox_inches="tight")
    plt.close()


def plot_t_sensitivity(df, out_path):
    """3. 마진 vs T-Sensitivity (Sigma) 분산 산점도"""
    corr, p = spearmanr(df['base_margin'], df['t_sensitivity'])
    
    plt.figure(figsize=(8, 6), dpi=200)
    # 회귀선이 포함된 산점도
    sns.regplot(x='base_margin', y='t_sensitivity', data=df, 
                scatter_kws={'alpha':0.4, 'color': 'purple', 's': 20}, 
                line_kws={'color': 'red', 'linewidth': 2})
    
    plt.title(f"1-View Margin vs. T-Sensitivity ($\hat{{\sigma}}_i$)\nSpearman $r$ = {corr:.3f} ($p$ < {p:.1e})")
    plt.xlabel("1-View Margin ($g_i$)")
    plt.ylabel("T-Sensitivity ($\hat{{\sigma}}_i$) across views")
    
    # 임계값 예시 선 (선택 사항)
    plt.axvline(x=0.3, color='gray', linestyle='--', label="Routing $thr_1$ Example")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{out_path}_3_t_sensitivity.png", bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Path to JSONL file/dir/glob (e.g., results_mmlu/.../anatomy.jsonl or results_mmlu/.../mmlu_full)")
    parser.add_argument("--out_prefix", type=str, default="analysis_plots/margin_vs_t", help="Output path prefix for plots")
    parser.add_argument("--k", type=int, default=None, help="(Optional) Force number of options (k). If omitted, inferred from data.")

    parser.add_argument(
        "--pretrained_model_paths",
        nargs="+",
        default=None,
        metavar="HF_ID",
        help=(
            "eval_clm 의 --pretrained_model_path 와 같은 문자열을 여러 개 (공백 구분). "
            "code/results_{task}/{n}s_{model_name}/... 에서만 jsonl을 재귀 수집; 다른 0s_* 모델은 섞이지 않음. "
            "지정 시 --input 은 무시됩니다."
        ),
    )
    parser.add_argument(
        "--results_eval_name",
        type=str,
        default="mmlu,0,cyclic",
        help="--pretrained_model_paths 사용 시 결과 디렉터리를 잡는 eval 이름 (예: mmlu,0,cyclic, mmlu,0,full).",
    )
    parser.add_argument(
        "--option_id_set",
        type=str,
        default=None,
        help="--pretrained_model_paths 사용 시 경로 끝에 _id-XXXX (eval_clm 과 동일, 예: ABCD).",
    )
    parser.add_argument(
        "--code_dir",
        type=str,
        default=None,
        help=(
            "eval_clm 결과가 들어 있는 **code 디렉터리** (그 아래에 results_mmlu, results_arc 등). "
            "기본값은 plot_margin_t.py 가 있는 폴더. "
            "NAS 등 다른 위치에만 결과가 있으면 예: "
            "/nas2/data/.../LLM-MCQ-Bias/code"
        ),
    )

    parser.add_argument("--auto_eval", action="store_true", help="If set, run eval_clm.py to generate missing JSONLs before plotting.")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Passed to eval_clm.py when --auto_eval is used.")
    parser.add_argument("--eval_names", type=str, nargs="+", default=["mmlu,0,full", "arc,0,full", "csqa,0,full"],
                        help="Passed to eval_clm.py. Example: mmlu,0,full arc,0,full csqa,0,full")
    parser.add_argument("--eval_clm_args", nargs=argparse.REMAINDER, default=[],
                        help="Extra args to pass through to eval_clm.py. "
                             "Example: --eval_clm_args --option_id_set ABCD --force --wandb")
    args = parser.parse_args()

    if args.pretrained_model_paths and args.input:
        print("[info] --pretrained_model_paths 가 있어 --input 은 무시합니다.", file=sys.stderr)
    if args.pretrained_model_paths and bool(args.auto_eval):
        raise SystemExit("--pretrained_model_paths 와 --auto_eval 은 함께 쓸 수 없습니다.")

    _safe_makedirs_for_prefix(args.out_prefix)

    code_dir = os.path.dirname(os.path.abspath(__file__))
    results_code_dir = (
        os.path.abspath(os.path.expanduser(os.path.expandvars(str(args.code_dir))))
        if args.code_dir
        else code_dir
    )

    if bool(args.auto_eval):
        if not args.pretrained_model_path:
            raise SystemExit("--auto_eval requires --pretrained_model_path")

        extra = list(args.eval_clm_args or [])
        _validate_option_id_set(list(args.eval_names), extra)
        opt_id = _extract_flag_value(extra, "--option_id_set")

        cmd = [
            sys.executable,
            os.path.join(code_dir, "eval_clm.py"),
            "--pretrained_model_path",
            str(args.pretrained_model_path),
            "--eval_names",
            *list(args.eval_names),
            *extra,
        ]
        print("Running eval_clm.py to generate JSONLs...")
        print(" ".join([str(x) for x in cmd]))
        subprocess.run(cmd, cwd=code_dir, check=True)

        collected = _collect_eval_outputs(code_dir, list(args.eval_names), str(args.pretrained_model_path), option_id_set=opt_id)
        if len(collected) == 0:
            raise SystemExit("No eval_names provided or no outputs collected.")

        model_tag = _sanitize_tag(_model_name_from_pretrained_path(str(args.pretrained_model_path)))
        any_plotted = False
        for ev in list(args.eval_names):
            files = collected.get(str(ev), []) or []
            if len(files) == 0:
                print(f"[WARN] No JSONL files found for eval_name='{ev}' in expected save_path.")
                continue

            save_path, task, num_few_shot, setting = _eval_save_path(code_dir, ev, str(args.pretrained_model_path), option_id_set=opt_id)
            label = f"{task},{num_few_shot},{setting or 'base'}"
            is_mmlu = (str(task).lower() == "mmlu")
            group_tag = f"{task}_allsubjects" if is_mmlu else f"{task}"

            print(f"[{label}] Found {len(files)} JSONL file(s). Loading...")
            dfs: list[pd.DataFrame] = []
            for fp in files:
                df_i = process_jsonl(fp)
                if len(df_i) > 0:
                    dfs.append(df_i)
            if len(dfs) == 0:
                print(f"[{label}] No valid records found. Skipping.")
                continue

            df_all = pd.concat(dfs, ignore_index=True)
            if args.k is not None:
                df_all = df_all[df_all["k"] == int(args.k)].copy()
            if len(df_all) == 0:
                print(f"[{label}] No records after filtering. Skipping.")
                continue

            ks = sorted([int(x) for x in df_all["k"].dropna().unique().tolist()])
            print(f"[{label}] Loaded {len(df_all)} samples across k={ks}. Plotting...")
            for k in ks:
                dfk = df_all[df_all["k"] == k].copy()
                if len(dfk) == 0:
                    continue
                out_prefix = f"{args.out_prefix}_{model_tag}_{group_tag}"
                out_prefix_k = out_prefix if len(ks) == 1 else f"{out_prefix}_k{k}"
                plot_heatmap(dfk, out_prefix_k, k)
                plot_required_t(dfk, out_prefix_k)
                plot_required_t_mean_cyclic(dfk, out_prefix_k)
                plot_t_sensitivity(dfk, out_prefix_k)
                any_plotted = True

        if not any_plotted:
            raise SystemExit("No plots were generated. Check eval outputs and JSONL format.")

        print(f"Done! Plots saved with base prefix: {args.out_prefix}")
        return

    # ---- non-auto mode: plot whatever user provided (file/dir/glob) ----
    repo_root = os.path.abspath(os.path.join(code_dir, os.pardir))

    if args.pretrained_model_paths:
        input_files, pm_warns = _collect_jsonl_from_pretrained_paths(
            results_code_dir,
            str(args.results_eval_name),
            list(args.pretrained_model_paths),
            args.option_id_set,
        )
        for w in pm_warns:
            print(w, file=sys.stderr)
        if len(input_files) == 0:
            raise SystemExit(
                "No JSONL from --pretrained_model_paths. Check --code_dir, --results_eval_name, --option_id_set, "
                "and that eval outputs exist under <code_dir>/results_*/."
            )
        n_pm = len([x for x in args.pretrained_model_paths if str(x).strip()])
        if n_pm == 1:
            mt = _sanitize_tag(_model_name_from_pretrained_path(str(args.pretrained_model_paths[0])))
            base_out_prefix = f"{args.out_prefix}_{mt}"
        else:
            base_out_prefix = f"{args.out_prefix}_pooled{n_pm}models"
    else:
        base_dirs = [os.getcwd(), repo_root, code_dir]
        if args.code_dir:
            base_dirs.insert(0, results_code_dir)
        input_files = _expand_inputs_multi(base_dirs, args.input)
        if len(input_files) == 0:
            raise SystemExit(
                "No JSONL inputs found. Provide --input, or --pretrained_model_paths, or use --auto_eval."
            )
        inferred_model = _infer_model_tag_from_inputs(input_files)
        base_out_prefix = args.out_prefix
        if inferred_model:
            base_out_prefix = f"{args.out_prefix}_{_sanitize_tag(inferred_model)}"

    print(f"Found {len(input_files)} JSONL file(s). Loading...")
    dfs2: list[pd.DataFrame] = []
    for fp in input_files:
        df_i = process_jsonl(fp)
        if len(df_i) > 0:
            dfs2.append(df_i)
    if len(dfs2) == 0:
        print("No valid records found. Check if the JSONL contains 'probs'.")
        return

    df_all2 = pd.concat(dfs2, ignore_index=True)
    if args.k is not None:
        df_all2 = df_all2[df_all2["k"] == int(args.k)].copy()
    if len(df_all2) == 0:
        print("No records after filtering. (k mismatch or empty inputs)")
        return

    ks2 = sorted([int(x) for x in df_all2["k"].dropna().unique().tolist()])
    print(f"Loaded {len(df_all2)} valid samples across k={ks2}. Generating plots...")

    for k in ks2:
        dfk = df_all2[df_all2["k"] == k].copy()
        if len(dfk) == 0:
            continue
        out_prefix_k = base_out_prefix if len(ks2) == 1 else f"{base_out_prefix}_k{k}"
        plot_heatmap(dfk, out_prefix_k, k)
        plot_required_t(dfk, out_prefix_k)
        plot_required_t_mean_cyclic(dfk, out_prefix_k)
        plot_t_sensitivity(dfk, out_prefix_k)

    print(f"Done! Plots saved with prefix: {base_out_prefix}")

if __name__ == "__main__":
    main()