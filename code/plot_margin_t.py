import os
import json
import argparse
import glob
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

# 논문용 깔끔한 스타일 세팅
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def _rotations(k):
    return [tuple((i + s) % k for i in range(k)) for s in range(k)]

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


def _safe_makedirs_for_prefix(out_prefix: str) -> None:
    d = os.path.dirname(out_prefix)
    if d:
        os.makedirs(d, exist_ok=True)


def _eval_save_path(code_dir: str, eval_name: str, pretrained_model_path: str) -> tuple[str, str, int, str | None]:
    parts = [p.strip() for p in str(eval_name).split(",")]
    if len(parts) < 2:
        raise ValueError(f"Invalid eval_name='{eval_name}'. Expected 'task,num_few_shot,setting(optional)'.")
    task = parts[0]
    num_few_shot = int(parts[1])
    setting = parts[2] if len(parts) > 2 and parts[2] else None

    model_name = os.path.basename(pretrained_model_path.rstrip("/\\")).strip()
    save_path = os.path.join(code_dir, f"results_{task}", f"{num_few_shot}s_{model_name}", f"{task}")
    if setting is not None:
        save_path += f"_{setting}"
    return save_path, task, num_few_shot, setting


def _collect_eval_outputs(code_dir: str, eval_names: list[str], pretrained_model_path: str) -> dict[str, list[str]]:
    """
    Returns mapping: label -> list of jsonl file paths
    label examples: 'mmlu,0,full'
    """
    out: dict[str, list[str]] = {}
    for ev in eval_names:
        save_path, task, num_few_shot, setting = _eval_save_path(code_dir, ev, pretrained_model_path)
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

            # 1. Cyclic indices 추출 (eval_clm.py 방식과 동일)
            identity_idx = 0
            cyc_perms = _rotations(k)

            # 2. 1-view Base 측정
            base_probs = np.asarray(probs_seq_np[identity_idx], dtype=np.float64)
            base_gap = float(_gap(base_probs))
            base_pred = int(np.argmax(base_probs))

            # 3. T-Sensitivity 계산 (k개의 cyclic view에서 gap의 std)
            individual_gaps: list[float] = []
            for i in range(k):
                # 결과가 rotation k개만 있을 땐 앞의 k개가 cyclic이라고 가정
                idx = i if len(probs_seq_np) == k else int(cyc_perms[i][0])
                if idx < 0 or idx >= len(probs_seq_np):
                    continue
                view_probs = np.asarray(probs_seq_np[idx], dtype=np.float64)
                individual_gaps.append(float(_gap(view_probs)))
            t_sensitivity = float(np.std(individual_gaps)) if len(individual_gaps) > 0 else 0.0

            # 4. T 1~k 누적 평균으로 Required T 계산
            cum_probs = np.zeros(k, dtype=np.float64)
            preds_at_t: list[int] = []
            for t in range(1, k + 1):
                idx = (t - 1) if len(probs_seq_np) == k else int(cyc_perms[t - 1][0])
                if idx < 0 or idx >= len(probs_seq_np):
                    break
                cum_probs += np.asarray(probs_seq_np[idx], dtype=np.float64)
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

    parser.add_argument("--auto_eval", action="store_true", help="If set, run eval_clm.py to generate missing JSONLs before plotting.")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Passed to eval_clm.py when --auto_eval is used.")
    parser.add_argument("--eval_names", type=str, nargs="+", default=["mmlu,0,full", "arc,0,full", "csqa,0,full"],
                        help="Passed to eval_clm.py. Example: mmlu,0,full arc,0,full csqa,0,full")
    args = parser.parse_args()

    _safe_makedirs_for_prefix(args.out_prefix)

    code_dir = os.path.dirname(os.path.abspath(__file__))

    input_files: list[str] = []

    if bool(args.auto_eval):
        if not args.pretrained_model_path:
            raise SystemExit("--auto_eval requires --pretrained_model_path")

        cmd = [
            sys.executable,
            os.path.join(code_dir, "eval_clm.py"),
            "--pretrained_model_path",
            str(args.pretrained_model_path),
            "--eval_names",
            *list(args.eval_names),
        ]
        print("Running eval_clm.py to generate JSONLs...")
        print(" ".join([str(x) for x in cmd]))
        subprocess.run(cmd, cwd=code_dir, check=True)

        collected = _collect_eval_outputs(code_dir, list(args.eval_names), str(args.pretrained_model_path))
        for ev, files in collected.items():
            if len(files) == 0:
                print(f"[WARN] No JSONL files found for eval_name='{ev}' in expected save_path.")
            input_files.extend(files)
    else:
        input_files = _expand_inputs(args.input)

    input_files = sorted(set(input_files))
    if len(input_files) == 0:
        raise SystemExit("No JSONL inputs found. Provide --input or use --auto_eval.")

    print(f"Found {len(input_files)} JSONL file(s). Loading...")
    dfs: list[pd.DataFrame] = []
    for fp in input_files:
        df_i = process_jsonl(fp)
        if len(df_i) > 0:
            dfs.append(df_i)
    if len(dfs) == 0:
        print("No valid records found. Check if the JSONL contains 'probs'.")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    # Optional: force k
    if args.k is not None:
        df_all = df_all[df_all["k"] == int(args.k)].copy()

    if len(df_all) == 0:
        print("No records after filtering. (k mismatch or empty inputs)")
        return

    # Split by k (4-choice vs 5-choice etc)
    ks = sorted([int(x) for x in df_all["k"].dropna().unique().tolist()])
    print(f"Loaded {len(df_all)} valid samples across k={ks}. Generating plots...")

    for k in ks:
        dfk = df_all[df_all["k"] == k].copy()
        if len(dfk) == 0:
            continue
        out_prefix_k = args.out_prefix if len(ks) == 1 else f"{args.out_prefix}_k{k}"
        plot_heatmap(dfk, out_prefix_k, k)
        plot_required_t(dfk, out_prefix_k)
        plot_t_sensitivity(dfk, out_prefix_k)

    print(f"Done! Plots saved with prefix: {args.out_prefix}")

if __name__ == "__main__":
    main()