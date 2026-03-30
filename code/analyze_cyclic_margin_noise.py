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
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Cyclic margin / residual noise analysis for cached eval jsonl.")
    ap.add_argument("--jsonl_paths", type=str, nargs="*", default=None, help="Explicit jsonl files.")
    ap.add_argument("--results_dir", type=str, default="", help="Directory with *.jsonl (recursive glob).")
    ap.add_argument("--jsonl_glob", type=str, default="*.jsonl")
    ap.add_argument("--out_dir", type=str, default="cyclic_margin_noise_out")
    ap.add_argument("--max_samples", type=int, default=None, help="Max result rows per file (debug).")
    ap.add_argument("--bins", type=int, default=80, help="Histogram bins.")
    ap.add_argument("--kl_bins", type=int, default=80, help="Bins for binned KL to fitted normal.")
    args = ap.parse_args()

    files: List[str] = []
    if args.jsonl_paths:
        files = [os.path.abspath(p) for p in args.jsonl_paths]
    rd = str(args.results_dir).strip()
    if rd:
        files.extend(_discover_jsonl_files(rd, jsonl_glob=str(args.jsonl_glob)))
    files = sorted(set(files))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print("No jsonl files found. Pass --jsonl_paths or --results_dir.", file=sys.stderr)
        sys.exit(1)

    if plt is None:
        print("matplotlib is required for plots.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    all_margins: List[float] = []
    all_residuals: List[float] = []
    all_sample_std: List[float] = []
    all_sample_var: List[float] = []
    meta_lines: List[str] = []

    for fp in files:
        loaded = _load_cyclic_gaps_from_file(fp, max_samples=args.max_samples)
        if loaded is None:
            meta_lines.append(f"[skip] {fp} (no full cyclic probs or mismatch)")
            continue
        k, opt_ids, margins_all, means, stds, vars_, residuals_flat, margins_flat = loaded
        all_margins.extend(margins_flat.tolist())
        all_residuals.extend(residuals_flat.tolist())
        all_sample_std.extend(stds.tolist())
        all_sample_var.extend(vars_.tolist())
        meta_lines.append(
            f"[ok] {fp}  k={k}  n_samples={margins_all.shape[0]}  option_ids={''.join(opt_ids)}"
        )

    if not all_margins:
        print("No usable samples. Need jsonl with probs length = k (cyclic) or k! (full) containing all rotations.", file=sys.stderr)
        for m in meta_lines:
            print(m, file=sys.stderr)
        sys.exit(1)

    mar = np.asarray(all_margins, dtype=np.float64)
    res = np.asarray(all_residuals, dtype=np.float64)
    sstd = np.asarray(all_sample_std, dtype=np.float64)
    svar = np.asarray(all_sample_var, dtype=np.float64)

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines) + "\n\n")
        f.write(f"n_margins (N×k) = {mar.size}\n")
        f.write(f"n_residuals   = {res.size}\n")
        f.write(f"n_samples     = {sstd.size}\n")
        f.write(f"margin: mean={np.mean(mar):.6f} std={np.std(mar, ddof=1):.6f}\n")
        f.write(f"residual: mean={np.mean(res):.6f} std={np.std(res, ddof=1):.6f}\n")
        f.write(f"per-sample std: mean={np.mean(sstd):.6f} std={np.std(sstd, ddof=1):.6f}\n")
        f.write(f"per-sample var: mean={np.mean(svar):.6f}\n")

        if scipy_stats is not None and res.size >= 20:
            # Shapiro on subsample (max 5000; Shapiro is sensitive for huge n)
            sub = res
            if sub.size > 5000:
                rng = np.random.default_rng(0)
                sub = rng.choice(sub, size=5000, replace=False)
            w, pval = scipy_stats.shapiro(sub)
            f.write(f"Shapiro-Wilk (n={sub.size}): W={w:.6f} p={pval:.6e}\n")
            k2, p_nt = scipy_stats.normaltest(res)
            f.write(f"D'Agostino-Pearson normaltest: K2={k2:.6f} p={p_nt:.6e}\n")

        kl_out = _kl_discrete_normal_binned(res, n_bins=int(args.kl_bins))
        if kl_out is not None:
            kl, mu_q, sig_q = kl_out
            f.write(f"Binned KL(P_emp || N(mu_hat,sig_hat)): KL={kl:.6f}  mu_hat={mu_q:.6f} sig_hat={sig_q:.6f}\n")
        else:
            f.write("Binned KL: N/A (need scipy)\n")

    # Figures
    fig1, ax1 = plt.subplots(figsize=(9, 5), dpi=140)
    _plot_hist(ax1, mar, "Pooled cyclic margins (top1−top2, letter-space)", "Margin", bins=args.bins)
    fig1.tight_layout()
    fig1.savefig(os.path.join(args.out_dir, "pooled_margins.png"))

    fig2, ax2 = plt.subplots(figsize=(9, 5), dpi=140)
    _plot_hist(ax2, res, "Pooled residuals (M_r − mean_r M)", "Residual", bins=args.bins)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.out_dir, "pooled_residuals.png"))

    fig3, ax3 = plt.subplots(figsize=(9, 5), dpi=140)
    _plot_hist(ax3, sstd, "Per-sample std across cyclic rotations", "Std(M_1..M_k)", bins=min(args.bins, 50))
    fig3.tight_layout()
    fig3.savefig(os.path.join(args.out_dir, "per_sample_std.png"))

    fig4, ax4 = plt.subplots(figsize=(6, 6), dpi=140)
    _plot_qq(ax4, res, "QQ: residuals vs normal")
    fig4.tight_layout()
    fig4.savefig(os.path.join(args.out_dir, "qq_residuals.png"))
    plt.close("all")

    print(f"Wrote: {args.out_dir}/ (summary.txt, *.png)")
    print(f"Read summary: {summary_path}")


if __name__ == "__main__":
    main()
