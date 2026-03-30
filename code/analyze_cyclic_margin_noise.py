#!/usr/bin/env python3
"""
Cyclic rotation별 top1-top2 margin 분포 / 잔차 / 샘플 내 분산 분석.

eval_clm 캐시 jsonl (`data.probs`: permutation별 letter-space 확률)을 읽어,
각 샘플에 대해 cyclic rotations (k개)마다 gap = top1-top2 를 구하고:
  - 회전별 margin M_r
  - 샘플 내 평균 M̄, (표본)분산/표준편차
  - 잔차 ε_r = M_r - M̄  (합이 0이 되도록)

풀링:
  - 모든 샘플×회전 margin
  - 모든 잔차 (N×k 개)
  - 샘플별 std (N개)

선택: 잔차에 대해 정규성 QQ plot, 히스토그램 vs 적합 N(μ,σ²), binned KL(P||Q).

사용 예:
  python code/analyze_cyclic_margin_noise.py \\
    --jsonl_paths /path/to/results.jsonl \\
    --out_dir ./margin_noise_plots

  python code/analyze_cyclic_margin_noise.py \\
    --results_dir /path/to/results_mmlu/0s_MODEL/mmlu_cyclic_id-ABCD \\
    --jsonl_glob "*.jsonl"

  # results_mmlu 아래 0s_* 모델 폴더를 전부 돌려서 한 번에 풀링 + 모델별 요약 CSV:
  python code/analyze_cyclic_margin_noise.py \\
    --mmlu_root /nas2/data/jihye4118/g/LLM-MCQ-Bias/code/results_mmlu \\
    --out_dir ./margin_noise_mmlu_all_0s

  # 특정 모델만 (폴더 basename 부분 문자열 또는 전체 이름):
  python code/analyze_cyclic_margin_noise.py \\
    --mmlu_root .../results_mmlu --models 0s_Meta-Llama-3-8B-Instruct Llama \\
    --out_dir ./margin_noise_subset

  # HF model_path(eval_clm 의 --pretrained_model_path 와 동일 규칙: 마지막 세그먼트가 폴더명):
  #   .../results_mmlu/0s_Llama-3.1-8B-Instruct/mmlu_cyclic_id-ABCD/
  python code/analyze_cyclic_margin_noise.py \\
    --mmlu_root .../results_mmlu \\
    --model_path meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct \\
    --out_dir ./margin_noise_two_models
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None  # type: ignore


def _iter_result_rows(jsonl_path: str) -> Iterator[dict]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") != "result":
                continue
            data = obj.get("data", {}) or {}
            if isinstance(data, dict):
                yield data


def _rotations(k: int) -> List[Tuple[int, ...]]:
    return [tuple((i + s) % k for i in range(k)) for s in range(k)]


def _infer_perm_list(k: int, perm_count: int) -> List[Tuple[int, ...]]:
    from itertools import permutations

    if perm_count == math.factorial(k):
        return list(sorted(permutations(range(k))))
    if perm_count == k:
        return _rotations(k)
    full = list(sorted(permutations(range(k))))
    if perm_count <= len(full):
        return full[:perm_count]
    return full


def _gap_top1_top2(bp: np.ndarray) -> float:
    s = np.sort(np.asarray(bp, dtype=np.float64))[::-1]
    if s.size <= 1:
        return 0.0
    return float(s[0] - s[1])


def _discover_jsonl_files(results_dir: str, jsonl_glob: str = "*.jsonl") -> List[str]:
    pats = [
        os.path.join(results_dir, str(jsonl_glob)),
        os.path.join(results_dir, "**", str(jsonl_glob)),
    ]
    files: List[str] = []
    for pat in pats:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files))
    files = [
        p
        for p in files
        if p.endswith(".jsonl")
        and (not p.endswith("_curve.jsonl"))
        and (not p.endswith("_pride_curve.jsonl"))
    ]
    return files


def _discover_shot_model_dirs(mmlu_root: str, shot_prefix: str = "0s") -> List[str]:
    """
    results_mmlu/0s_<model_name>/ 형태의 디렉터리만 모은다.
    """
    out: List[str] = []
    if not os.path.isdir(mmlu_root):
        return out
    prefix = f"{str(shot_prefix).strip()}_"
    for name in sorted(os.listdir(mmlu_root)):
        path = os.path.join(mmlu_root, name)
        if not os.path.isdir(path):
            continue
        if not name.startswith(prefix):
            continue
        out.append(path)
    return out


def _task_results_subdir(task: str, setting: str, option_id_set: str) -> str:
    return f"{task}_{setting}_id-{option_id_set}"


def _discover_jsonl_from_mmlu_models(
    mmlu_root: str,
    task: str = "mmlu",
    setting: str = "cyclic",
    option_id_set: str = "ABCD",
    shot_prefix: str = "0s",
    model_glob: Optional[str] = None,
    models: Optional[Sequence[str]] = None,
    jsonl_glob: str = "*.jsonl",
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Returns:
      tagged: list of (jsonl_path, model_dir_basename) e.g. (.../run.jsonl, "0s_llama-7b")
      warnings: human-readable skip reasons
    """
    warnings: List[str] = []
    model_dirs = _discover_shot_model_dirs(mmlu_root, shot_prefix=shot_prefix)
    if not model_dirs:
        warnings.append(f"No subdirs like {shot_prefix}_* under {mmlu_root}")
        return [], warnings

    subdir = _task_results_subdir(task, setting, option_id_set)
    tagged: List[Tuple[str, str]] = []
    want = [str(x).strip() for x in (models or []) if str(x).strip()]

    for md in model_dirs:
        base = os.path.basename(md)
        if model_glob and not fnmatch.fnmatch(base, model_glob):
            continue
        if want:
            if base not in want and not any(w in base for w in want):
                continue
        tr = os.path.join(md, subdir)
        if not os.path.isdir(tr):
            warnings.append(f"[skip] no dir {tr}")
            continue
        files = _discover_jsonl_files(tr, jsonl_glob=jsonl_glob)
        if not files:
            warnings.append(f"[skip] no jsonl under {tr}")
            continue
        for fp in files:
            tagged.append((os.path.abspath(fp), base))

    if not tagged and not warnings:
        warnings.append("No jsonl after filters; check --task/--setting/--option_id_set/--models/--model_glob")
    return tagged, warnings


def _model_name_from_pretrained_like(s: str) -> str:
    """
    eval_clm_utils 와 동일: pretrained_model_path 의 마지막 '/' 뒤를 model_name 으로 씀.
    슬래시 없이 `Llama-3.1-8B-Instruct` 만 주면 그대로 model_name.
    """
    s = str(s).strip()
    if not s:
        return ""
    return s.split("/")[-1] if "/" in s else s


def _discover_jsonl_for_model_paths(
    mmlu_root: str,
    pretrained_like_paths: Sequence[str],
    task: str,
    setting: str,
    option_id_set: str,
    shot_prefix: str,
    jsonl_glob: str,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    --model_path 로 받은 HF id(또는 짧은 model_name)마다
    ``mmlu_root / {shot_prefix}_{model_name} / {task}_{setting}_id-{opt}/`` 에서 jsonl 수집.
    """
    warnings: List[str] = []
    tagged: List[Tuple[str, str]] = []
    subdir = _task_results_subdir(task, setting, option_id_set)
    mroot = os.path.abspath(mmlu_root)
    sp = str(shot_prefix).strip()

    for raw in pretrained_like_paths:
        raw = str(raw).strip()
        if not raw:
            continue
        model_name = _model_name_from_pretrained_like(raw)
        if not model_name:
            continue
        dir_base = f"{sp}_{model_name}"
        model_dir = os.path.join(mroot, dir_base)
        tr = os.path.join(model_dir, subdir)
        if not os.path.isdir(model_dir):
            warnings.append(f"[skip] no model dir: {model_dir}  (--model_path {raw})")
            continue
        if not os.path.isdir(tr):
            warnings.append(f"[skip] no task dir: {tr}  (--model_path {raw})")
            continue
        files = _discover_jsonl_files(tr, jsonl_glob=jsonl_glob)
        if not files:
            warnings.append(f"[skip] no jsonl under {tr}  (--model_path {raw})")
            continue
        for fp in files:
            tagged.append((os.path.abspath(fp), dir_base))

    if not tagged and not warnings:
        warnings.append("No jsonl from --model_path list; check paths and task folder name.")
    return tagged, warnings


def _kl_discrete_normal_binned(
    x: np.ndarray, n_bins: int = 80, eps: float = 1e-12
) -> Optional[Tuple[float, float, float]]:
    """
    히스토그램으로 경험분포 P, 동일 구간에서 N(mu_hat, sigma_hat) 이산화 Q.
    KL(P || Q). mu_hat, sigma_hat 은 x의 MLE (표본 평균/분산).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 10:
        return None
    mu = float(np.mean(x))
    sig = float(np.std(x, ddof=1))
    if not np.isfinite(sig) or sig <= 0:
        sig = float(np.std(x, ddof=0)) or 1e-6
    lo, hi = float(np.min(x)), float(np.max(x))
    pad = max(1e-6, 0.02 * (hi - lo))
    edges = np.linspace(lo - pad, hi + pad, int(n_bins) + 1)
    hist, _ = np.histogram(x, bins=edges)
    p = hist.astype(np.float64)
    p_sum = p.sum()
    if p_sum <= 0:
        return None
    p = p / p_sum
    # Q: bin 확률 = Phi((e_{i+1}-mu)/sig) - Phi((e_i-mu)/sig)
    if scipy_stats is None:
        return None
    cdf = scipy_stats.norm.cdf
    q = cdf(edges[1:], loc=mu, scale=sig) - cdf(edges[:-1], loc=mu, scale=sig)
    q = np.asarray(q, dtype=np.float64)
    q = np.clip(q, eps, 1.0)
    q = q / q.sum()
    p = np.clip(p, eps, 1.0)
    kl = float(np.sum(p * (np.log(p) - np.log(q))))
    return kl, mu, sig


def _load_cyclic_gaps_from_file(
    fp: str,
    max_samples: Optional[int] = None,
) -> Optional[
    Tuple[
        int,
        List[str],
        np.ndarray,  # margins_all: (N, k) cyclic order
        np.ndarray,  # mean per sample (N,)
        np.ndarray,  # std per sample (N,)
        np.ndarray,  # var per sample (N,)
        np.ndarray,  # residuals flat (N*k,)
        np.ndarray,  # margins flat (N*k,)
    ]
]:
    """
    한 jsonl에서 cyclic k개 회전 각각의 letter-space top1-top2 gap 행렬 반환.
    """
    k_local: Optional[int] = None
    option_ids_local: Optional[List[str]] = None
    perm_list: Optional[List[Tuple[int, ...]]] = None
    identity_idx: Optional[int] = None
    cyc_indices: Optional[List[int]] = None

    margins_rows: List[np.ndarray] = []
    means: List[float] = []
    stds: List[float] = []
    vars_: List[float] = []
    margins_flat: List[float] = []
    residuals_flat: List[float] = []

    n_seen = 0
    for d in _iter_result_rows(fp):
        if max_samples is not None and n_seen >= max_samples:
            break
        probs = d.get("probs", None)
        if not isinstance(probs, list) or len(probs) == 0:
            continue
        if not isinstance(probs[0], list):
            continue
        row0 = probs[0]
        if not isinstance(row0, list) or len(row0) == 0:
            continue

        k = int(len(row0))
        perm_count = int(len(probs))
        if k_local is None:
            k_local = k
            option_ids_local = list("ABCDE"[:k_local]) if k_local in (4, 5) else [str(i) for i in range(k_local)]
            perm_list = _infer_perm_list(k_local, perm_count)
            identity = tuple(range(k_local))
            identity_idx = perm_list.index(identity) if identity in perm_list else 0
            cyc_perms = _rotations(k_local)
            cyc_indices = []
            for p in cyc_perms:
                if p in perm_list:
                    cyc_indices.append(perm_list.index(p))
            if len(cyc_indices) != k_local:
                return None

        if k != k_local or perm_list is None or cyc_indices is None:
            continue

        gaps_rot: List[float] = []
        for pi in cyc_indices:
            lp = np.asarray(probs[pi], dtype=np.float64)
            if lp.ndim != 1 or lp.size != k_local:
                gaps_rot = []
                break
            gaps_rot.append(_gap_top1_top2(lp))
        if len(gaps_rot) != k_local:
            continue

        g = np.asarray(gaps_rot, dtype=np.float64)
        mu = float(np.mean(g))
        var = float(np.var(g, ddof=1)) if k_local > 1 else 0.0
        std = float(np.std(g, ddof=1)) if k_local > 1 else 0.0
        res = g - mu

        margins_rows.append(g)
        means.append(mu)
        stds.append(std)
        vars_.append(var)
        margins_flat.extend(g.tolist())
        residuals_flat.extend(res.tolist())
        n_seen += 1

    if k_local is None or not margins_rows or option_ids_local is None:
        return None

    margins_all = np.stack(margins_rows, axis=0)
    return (
        int(k_local),
        option_ids_local,
        margins_all,
        np.asarray(means, dtype=np.float64),
        np.asarray(stds, dtype=np.float64),
        np.asarray(vars_, dtype=np.float64),
        np.asarray(residuals_flat, dtype=np.float64),
        np.asarray(margins_flat, dtype=np.float64),
    )


def _plot_hist(ax, data: np.ndarray, title: str, xlabel: str, bins: int = 60):
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    ax.hist(data, bins=bins, density=True, alpha=0.75, color="#4F46E5", edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.35)


def _plot_qq(ax, data: np.ndarray, title: str):
    data = np.asarray(data, dtype=np.float64).ravel()
    data = data[np.isfinite(data)]
    if scipy_stats is None or data.size < 8:
        ax.text(0.5, 0.5, "scipy unavailable or too few points", ha="center", va="center")
        ax.set_title(title)
        return
    scipy_stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)


def _aggregate_tagged_jsonl(
    tagged: List[Tuple[str, str]],
    max_samples: Optional[int],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    Dict[str, Dict[str, Any]],
]:
    """
    tagged: (jsonl_path, model_dir_basename)
    Returns pooled arrays + meta lines + per-model raw lists for CSV summary.
    """
    all_margins: List[float] = []
    all_residuals: List[float] = []
    all_sample_std: List[float] = []
    all_sample_var: List[float] = []
    meta_lines: List[str] = []
    per_model: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "margins": [],
            "residuals": [],
            "sample_std": [],
            "sample_var": [],
            "n_files_ok": 0,
            "n_files_skip": 0,
        }
    )

    for fp, mkey in tagged:
        loaded = _load_cyclic_gaps_from_file(fp, max_samples=max_samples)
        if loaded is None:
            meta_lines.append(f"[skip] {fp} (no full cyclic probs or mismatch)")
            per_model[mkey]["n_files_skip"] += 1
            continue
        k, opt_ids, margins_all, means, stds, vars_, residuals_flat, margins_flat = loaded
        all_margins.extend(margins_flat.tolist())
        all_residuals.extend(residuals_flat.tolist())
        all_sample_std.extend(stds.tolist())
        all_sample_var.extend(vars_.tolist())
        pm = per_model[mkey]
        pm["margins"].extend(margins_flat.tolist())
        pm["residuals"].extend(residuals_flat.tolist())
        pm["sample_std"].extend(stds.tolist())
        pm["sample_var"].extend(vars_.tolist())
        pm["n_files_ok"] += 1
        meta_lines.append(
            f"[ok] {fp}  model={mkey}  k={k}  n_samples={margins_all.shape[0]}  option_ids={''.join(opt_ids)}"
        )

    mar = np.asarray(all_margins, dtype=np.float64)
    res = np.asarray(all_residuals, dtype=np.float64)
    sstd = np.asarray(all_sample_std, dtype=np.float64)
    svar = np.asarray(all_sample_var, dtype=np.float64)
    return mar, res, sstd, svar, meta_lines, dict(per_model)


def _write_per_model_csv(path: str, per_model: Dict[str, Dict[str, Any]]) -> None:
    rows = []
    for model_key in sorted(per_model.keys()):
        pm = per_model[model_key]
        m = np.asarray(pm.get("margins", []), dtype=np.float64)
        r = np.asarray(pm.get("residuals", []), dtype=np.float64)
        ss = np.asarray(pm.get("sample_std", []), dtype=np.float64)
        sv = np.asarray(pm.get("sample_var", []), dtype=np.float64)
        n_samp = int(ss.size)
        rows.append(
            {
                "model_dir": model_key,
                "n_jsonl_ok": int(pm.get("n_files_ok", 0)),
                "n_jsonl_skip": int(pm.get("n_files_skip", 0)),
                "n_rows_samples": n_samp,
                "n_margins_pool": int(m.size),
                "mean_margin": float(np.nanmean(m)) if m.size else float("nan"),
                "std_margin": float(np.nanstd(m, ddof=1)) if m.size > 1 else float("nan"),
                "mean_residual": float(np.nanmean(r)) if r.size else float("nan"),
                "std_residual": float(np.nanstd(r, ddof=1)) if r.size > 1 else float("nan"),
                "mean_per_sample_std": float(np.nanmean(ss)) if ss.size else float("nan"),
                "mean_per_sample_var": float(np.nanmean(sv)) if sv.size else float("nan"),
            }
        )
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_summary_and_plots(
    out_dir: str,
    mar: np.ndarray,
    res: np.ndarray,
    sstd: np.ndarray,
    svar: np.ndarray,
    meta_lines: List[str],
    bins: int,
    kl_bins: int,
    per_model: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines) + "\n\n")
        f.write(f"n_margins (N×k) = {mar.size}\n")
        f.write(f"n_residuals   = {res.size}\n")
        f.write(f"n_samples     = {sstd.size}\n")
        if mar.size:
            f.write(f"margin: mean={np.mean(mar):.6f} std={np.std(mar, ddof=1):.6f}\n")
        if res.size:
            f.write(f"residual: mean={np.mean(res):.6f} std={np.std(res, ddof=1):.6f}\n")
        if sstd.size:
            f.write(f"per-sample std: mean={np.mean(sstd):.6f} std={np.std(sstd, ddof=1):.6f}\n")
        if svar.size:
            f.write(f"per-sample var: mean={np.mean(svar):.6f}\n")

        if scipy_stats is not None and res.size >= 20:
            sub = res
            if sub.size > 5000:
                rng = np.random.default_rng(0)
                sub = rng.choice(sub, size=5000, replace=False)
            w, pval = scipy_stats.shapiro(sub)
            f.write(f"Shapiro-Wilk (n={sub.size}): W={w:.6f} p={pval:.6e}\n")
            k2, p_nt = scipy_stats.normaltest(res)
            f.write(f"D'Agostino-Pearson normaltest: K2={k2:.6f} p={p_nt:.6e}\n")

        kl_out = _kl_discrete_normal_binned(res, n_bins=int(kl_bins))
        if kl_out is not None:
            kl, mu_q, sig_q = kl_out
            f.write(f"Binned KL(P_emp || N(mu_hat,sig_hat)): KL={kl:.6f}  mu_hat={mu_q:.6f} sig_hat={sig_q:.6f}\n")
        else:
            f.write("Binned KL: N/A (need scipy)\n")

        if per_model:
            f.write("\n-- Per-model means (see per_model_summary.csv) --\n")
            for mk in sorted(per_model.keys()):
                pm = per_model[mk]
                m = np.asarray(pm.get("margins", []), dtype=np.float64)
                if m.size:
                    f.write(f"{mk}: n_samples={len(pm.get('sample_std', []))} mean_margin={np.mean(m):.6f}\n")

    if plt is None:
        return summary_path

    fig1, ax1 = plt.subplots(figsize=(9, 5), dpi=140)
    _plot_hist(ax1, mar, "Pooled cyclic margins (top1−top2, letter-space)", "Margin", bins=bins)
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "pooled_margins.png"))

    fig2, ax2 = plt.subplots(figsize=(9, 5), dpi=140)
    _plot_hist(ax2, res, "Pooled residuals (M_r − mean_r M)", "Residual", bins=bins)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "pooled_residuals.png"))

    fig3, ax3 = plt.subplots(figsize=(9, 5), dpi=140)
    _plot_hist(ax3, sstd, "Per-sample std across cyclic rotations", "Std(M_1..M_k)", bins=min(bins, 50))
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "per_sample_std.png"))

    fig4, ax4 = plt.subplots(figsize=(6, 6), dpi=140)
    _plot_qq(ax4, res, "QQ: residuals vs normal")
    fig4.tight_layout()
    fig4.savefig(os.path.join(out_dir, "qq_residuals.png"))
    plt.close("all")
    return summary_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Cyclic margin / residual noise analysis for cached eval jsonl.")
    ap.add_argument("--jsonl_paths", type=str, nargs="*", default=None, help="Explicit jsonl files.")
    ap.add_argument("--results_dir", type=str, default="", help="Directory with *.jsonl (recursive glob).")
    ap.add_argument(
        "--mmlu_root",
        type=str,
        default="",
        help="results_mmlu 루트 (예: .../code/results_mmlu). 하위 0s_*/<task>_<setting>_id-OPT/ 에서 jsonl 수집.",
    )
    ap.add_argument("--task", type=str, default="mmlu", help="결과 하위 폴더 이름용 (기본 mmlu).")
    ap.add_argument("--setting", type=str, default="cyclic", help="예: cyclic, full")
    ap.add_argument("--option_id_set", type=str, default="ABCD", help="폴더 접미사 id-XXXX")
    ap.add_argument("--shot_prefix", type=str, default="0s", help="모델 폴더 접두 (기본 0s → 0s_* 만 포함).")
    ap.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="포함할 모델 폴더 basename (0s_...) 전체 또는 부분 문자열. 비우면 전부.",
    )
    ap.add_argument(
        "--model_glob",
        type=str,
        default=None,
        help="모델 폴더 basename에 대한 fnmatch 패턴 (예: 0s_*Llama*).",
    )
    ap.add_argument(
        "--model_path",
        action="append",
        nargs="+",
        default=None,
        dest="model_paths_hf_groups",
        metavar="HF_ID",
        help=(
            "eval_clm 의 --pretrained_model_path 와 같은 문자열 (예: meta-llama/Llama-3.1-8B-Instruct). "
            "마지막 '/' 뒤가 model_name 이 되고, 결과 폴더는 {shot_prefix}_{model_name} (기본 0s_*). "
            "여러 개는 한 줄에 공백으로: --model_path A B C 또는 플래그를 반복. "
            "--model_path 가 있으면 --models / --model_glob 은 무시됩니다."
        ),
    )
    ap.add_argument("--jsonl_glob", type=str, default="*.jsonl")
    ap.add_argument("--out_dir", type=str, default="cyclic_margin_noise_out")
    ap.add_argument("--max_samples", type=int, default=None, help="Max result rows per file (debug).")
    ap.add_argument("--bins", type=int, default=80, help="Histogram bins.")
    ap.add_argument("--kl_bins", type=int, default=80, help="Bins for binned KL to fitted normal.")
    ap.add_argument(
        "--no_per_model_csv",
        action="store_true",
        help="mmlu_root 배치일 때 per_model_summary.csv 를 쓰지 않음.",
    )
    args = ap.parse_args()

    tagged: List[Tuple[str, str]] = []
    batch_warnings: List[str] = []

    mroot = str(args.mmlu_root).strip()
    model_paths_hf: List[str] = []
    for grp in args.model_paths_hf_groups or []:
        for x in grp:
            s = str(x).strip()
            if s:
                model_paths_hf.append(s)

    if model_paths_hf:
        if args.models or args.model_glob:
            print("[info] --model_path 가 지정되어 --models / --model_glob 은 사용하지 않습니다.", file=sys.stderr)
        if not mroot:
            print("--model_path 는 results 위치인 --mmlu_root 와 함께 지정해야 합니다.", file=sys.stderr)
            sys.exit(1)
        tagged, batch_warnings = _discover_jsonl_for_model_paths(
            mmlu_root=os.path.abspath(mroot),
            pretrained_like_paths=model_paths_hf,
            task=str(args.task),
            setting=str(args.setting),
            option_id_set=str(args.option_id_set),
            shot_prefix=str(args.shot_prefix),
            jsonl_glob=str(args.jsonl_glob),
        )
    elif mroot:
        tagged, batch_warnings = _discover_jsonl_from_mmlu_models(
            mmlu_root=os.path.abspath(mroot),
            task=str(args.task),
            setting=str(args.setting),
            option_id_set=str(args.option_id_set),
            shot_prefix=str(args.shot_prefix),
            model_glob=args.model_glob,
            models=args.models,
            jsonl_glob=str(args.jsonl_glob),
        )

    files: List[str] = []
    if args.jsonl_paths:
        files = [os.path.abspath(p) for p in args.jsonl_paths]
    rd = str(args.results_dir).strip()
    if rd:
        files.extend(_discover_jsonl_files(rd, jsonl_glob=str(args.jsonl_glob)))
    files = sorted(set(files))
    files = [f for f in files if os.path.isfile(f)]

    if mroot and files:
        print("[warn] --mmlu_root 와 --jsonl_paths/--results_dir 가 동시에 지정됨. 둘 다 풀에 합칩니다.", file=sys.stderr)

    for fp in files:
        tagged.append((fp, "_single_dir"))

    if not tagged:
        print("No jsonl files found. Use --mmlu_root, or --results_dir, or --jsonl_paths.", file=sys.stderr)
        for w in batch_warnings:
            print(w, file=sys.stderr)
        sys.exit(1)

    if plt is None:
        print("matplotlib is required for plots.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    if batch_warnings:
        warn_path = os.path.join(args.out_dir, "batch_warnings.txt")
        with open(warn_path, "w", encoding="utf-8") as wf:
            wf.write("\n".join(batch_warnings) + "\n")

    mar, res, sstd, svar, meta_lines, per_model = _aggregate_tagged_jsonl(tagged, max_samples=args.max_samples)

    if not mar.size:
        print("No usable samples. Need jsonl with probs length = k (cyclic) or k! (full) containing all rotations.", file=sys.stderr)
        for m in meta_lines:
            print(m, file=sys.stderr)
        for w in batch_warnings:
            print(w, file=sys.stderr)
        sys.exit(1)

    pm_for_summary = per_model if (mroot and not args.no_per_model_csv) else None
    if mroot and not args.no_per_model_csv:
        csv_path = os.path.join(args.out_dir, "per_model_summary.csv")
        _write_per_model_csv(csv_path, per_model)
        print(f"Wrote per-model summary: {csv_path}")

    summary_path = _write_summary_and_plots(
        args.out_dir,
        mar,
        res,
        sstd,
        svar,
        meta_lines,
        bins=int(args.bins),
        kl_bins=int(args.kl_bins),
        per_model=pm_for_summary,
    )

    print(f"Wrote: {args.out_dir}/ (summary.txt, *.png)")
    print(f"Read summary: {summary_path}")


if __name__ == "__main__":
    main()
