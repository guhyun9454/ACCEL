#!/usr/bin/env python3
import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _try_import_wandb():
    try:
        import wandb  # type: ignore
        return wandb
    except Exception:
        return None


def _set_font():
    # Keep it simple: rely on OS defaults.
    try:
        import matplotlib.pyplot as plt  # type: ignore
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _compute_results_dir(code_dir: str, eval_name: str, pretrained_model_path: str, option_id_set: Optional[str]):
    eval_args = str(eval_name).split(",")
    task = str(eval_args[0]).strip()
    num_few_shot = int(eval_args[1])
    setting = str(eval_args[2]).strip() if len(eval_args) > 2 and str(eval_args[2]).strip() else None
    model_name = str(pretrained_model_path).split("/")[-1]
    save_path = f"results_{task}/{num_few_shot}s_{model_name}/{task}"
    if setting is not None:
        save_path += f"_{setting}"
    if option_id_set:
        save_path += f"_id-{option_id_set}"
    return os.path.join(code_dir, save_path)


def _discover_jsonl_files(results_dir: str, jsonl_glob: str = "*.jsonl", run_idx: Optional[int] = None) -> List[str]:
    pats = [
        os.path.join(results_dir, str(jsonl_glob)),
        os.path.join(results_dir, "**", str(jsonl_glob)),
    ]
    files: List[str] = []
    for pat in pats:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files))

    # exclude derived curve files
    files = [
        p for p in files
        if p.endswith(".jsonl")
        and (not p.endswith("_curve.jsonl"))
        and (not p.endswith("_pride_curve.jsonl"))
    ]

    if run_idx is not None:
        tok = f"run{int(run_idx)}"
        files = [p for p in files if tok in os.path.basename(p)]
    return files


def _iter_result_rows(jsonl_path: str):
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


def _rotations(k: int):
    return [tuple((i + s) % k for i in range(k)) for s in range(k)]


def _infer_perm_list(k: int, perm_count: int):
    from itertools import permutations

    if perm_count == math.factorial(k):
        return list(sorted(permutations(range(k))))
    if perm_count == k:
        return _rotations(k)
    full = list(sorted(permutations(range(k))))
    if perm_count <= len(full):
        return full[:perm_count]
    return full


def _extract_base_probs_and_pred(data: dict) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Supports multiple cache shapes:
    - base cache: probs = [k] (float list)
    - perm/full cache: probs = [n_perm][k] (nested list); we use identity row in inferred perm_list
    - cyclic cache: probs = [k][k] (k rotations); identity is index 0
    """
    probs = data.get("probs", None)
    if not isinstance(probs, list) or len(probs) == 0:
        return None, None
    # nested?
    if isinstance(probs[0], list):
        row0 = probs[0]
        if not isinstance(row0, list) or len(row0) == 0:
            return None, None
        k = int(len(row0))
        perm_count = int(len(probs))
        perm_list = _infer_perm_list(k, perm_count)
        identity = tuple(range(k))
        identity_idx = perm_list.index(identity) if identity in perm_list else 0
        if identity_idx < 0 or identity_idx >= perm_count:
            identity_idx = 0
        bp = np.asarray(probs[identity_idx], dtype=np.float64)
    else:
        bp = np.asarray(probs, dtype=np.float64)
    if bp.ndim != 1 or bp.size == 0:
        return None, None
    pred_idx = int(np.argmax(bp))
    return bp, pred_idx


def _gap_top1_top2(bp: np.ndarray) -> float:
    s = np.sort(np.asarray(bp, dtype=np.float64))[::-1]
    if s.size <= 1:
        return 0.0
    return float(s[0] - s[1])


def _quantile_bin_edges(x: np.ndarray, n_bins: int) -> List[float]:
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = [float(np.quantile(x, q)) for q in qs]
    # ensure monotonic (quantile can repeat when many ties)
    for i in range(1, len(edges)):
        edges[i] = max(edges[i], edges[i - 1])
    return edges


def _bin_indices(conf: np.ndarray, n_bins: int, mode: str) -> Tuple[List[np.ndarray], Optional[List[float]]]:
    """
    Returns:
      - list of index arrays (one per bin, in increasing-confidence order)
      - optional edges (only for quantile mode)

    mode:
      - "equal_count": sort by conf, split into exactly equal-sized bins (±1 due to remainder)
      - "quantile": threshold by quantiles (can create empty bins when many ties)
    """
    conf = np.asarray(conf, dtype=np.float64)
    N = int(conf.size)
    if N <= 0:
        return [], None
    nb = int(max(1, int(n_bins)))
    m = str(mode or "equal_count").strip().lower()

    if m == "quantile":
        edges = _quantile_bin_edges(conf, nb)
        idx_bins: List[np.ndarray] = []
        for i in range(nb):
            lo, hi = edges[i], edges[i + 1]
            if i < nb - 1:
                mask = (conf >= lo) & (conf < hi)
            else:
                mask = (conf >= lo) & (conf <= hi)
            idx = np.nonzero(mask)[0]
            idx_bins.append(idx)
        return idx_bins, edges

    # equal_count (default): stable sort, then split indices
    order = np.argsort(conf, kind="mergesort")
    chunks = np.array_split(order, nb)
    return [c.astype(np.int64) for c in chunks], None


def _recall_by_class(y_true: np.ndarray, y_pred: np.ndarray, k: int, mask: np.ndarray) -> np.ndarray:
    rec = np.full((k,), np.nan, dtype=np.float64)
    for c in range(k):
        m = mask & (y_true == c)
        denom = int(np.sum(m))
        if denom <= 0:
            continue
        rec[c] = float(np.mean((y_pred[m] == c).astype(np.float64)))
    return rec


def _analyze(
    files: List[str],
    option_id_set: Optional[str],
    n_bins: int,
    min_bin_n: int,
    aggregate_mode: str = "pooled",  # "pooled" | "mean_over_files"
    binning_mode: str = "equal_count",  # "equal_count" | "quantile"
    binning_scope: str = "global",  # "global" | "per_file" (only meaningful for mean_over_files)
):
    def _load_one_file(fp: str):
        confs1: List[float] = []
        yt1: List[int] = []
        yp1: List[int] = []
        k_local: Optional[int] = None
        option_ids_local: Optional[List[str]] = None
        for d in _iter_result_rows(fp):
            ideal = d.get("ideal", None)
            if not isinstance(ideal, str):
                continue
            options = d.get("options", None)
            if isinstance(options, list) and len(options) > 0:
                k = int(len(options))
            else:
                k = -1
            bp, pred_idx = _extract_base_probs_and_pred(d)
            if bp is None or pred_idx is None:
                continue
            if k <= 0:
                k = int(bp.size)
            if k <= 0:
                continue

            if k_local is None:
                k_local = k
                if option_id_set is not None:
                    option_ids_local = list(str(option_id_set))
                    if len(option_ids_local) != k_local:
                        raise ValueError(f"--option_id_set length must be {k_local}, got {len(option_ids_local)}: {option_id_set}")
                else:
                    option_ids_local = list("ABCDE"[:k_local]) if k_local in (4, 5) else [str(i) for i in range(k_local)]
            if k_local != k:
                continue
            assert option_ids_local is not None
            if ideal not in option_ids_local:
                continue
            confs1.append(_gap_top1_top2(bp))
            yt1.append(int(option_ids_local.index(ideal)))
            yp1.append(int(pred_idx))
        if k_local is None or len(confs1) == 0:
            return None
        return {
            "k": int(k_local),
            "option_ids": option_ids_local,
            "conf": np.asarray(confs1, dtype=np.float64),
            "yt": np.asarray(yt1, dtype=np.int64),
            "yp": np.asarray(yp1, dtype=np.int64),
            "file": fp,
        }

    per_file = []
    for fp in files:
        obj = _load_one_file(fp)
        if obj is not None:
            per_file.append(obj)
    if not per_file:
        raise ValueError("No usable samples found. Check results_dir and (if used) --option_id_set.")

    k_seen = int(per_file[0]["k"])
    option_ids = per_file[0]["option_ids"]
    for o in per_file[1:]:
        if int(o["k"]) != k_seen:
            raise ValueError("Mixed k across files; pass a cleaner selection.")
        if list(o["option_ids"]) != list(option_ids):
            raise ValueError("Mixed option id sets across files; pass --option_id_set or use consistent runs.")

    if str(aggregate_mode) == "mean_over_files":
        scope = str(binning_scope or "global").strip().lower()
        mode = str(binning_mode or "equal_count").strip().lower()
        # For global scope, we compute ONE set of percentile edges over ALL samples (all files/models),
        # then apply those edges to each file and average per-bin metrics across files.
        # This ensures bins correspond to p10, p20, ..., p100 over the combined population.
        global_edges: Optional[List[float]] = None
        if scope == "global":
            conf_all = np.concatenate([o["conf"] for o in per_file], axis=0)
            global_edges = _quantile_bin_edges(conf_all, int(n_bins))

        bins_by_i: Dict[int, List[dict]] = {i: [] for i in range(int(n_bins))}
        n_total = 0
        for o in per_file:
            conf = o["conf"]
            yt = o["yt"]
            yp = o["yp"]
            n_total += int(conf.size)
            if scope == "global" and global_edges is not None:
                idx_bins = []
                edges = global_edges
                for i in range(int(n_bins)):
                    lo, hi = edges[i], edges[i + 1]
                    if i < int(n_bins) - 1:
                        mask = (conf >= lo) & (conf < hi)
                    else:
                        mask = (conf >= lo) & (conf <= hi)
                    idx_bins.append(np.nonzero(mask)[0].astype(np.int64))
            else:
                # per_file binning (each file has its own deciles)
                idx_bins, _ = _bin_indices(conf, int(n_bins), mode)

            for i, idx in enumerate(idx_bins):
                N = int(idx.size)
                if N < int(min_bin_n):
                    continue
                acc = float(np.mean((yp[idx] == yt[idx]).astype(np.float64)))
                m = np.zeros((int(conf.size),), dtype=bool)
                m[idx] = True
                rec = _recall_by_class(yt, yp, int(k_seen), m)
                rstd = float(np.nanstd(rec))
                bins_by_i[i].append(
                    {
                        "n": N,
                        "conf_mean": float(np.mean(conf[idx])),
                        "acc": acc,
                        "recall_std": rstd,
                        "recalls": rec,
                    }
                )

        rows = []
        for i in range(int(n_bins)):
            items = bins_by_i.get(i, [])
            if not items:
                continue
            accs = np.asarray([it["acc"] for it in items], dtype=np.float64)
            rstds = np.asarray([it["recall_std"] for it in items], dtype=np.float64)
            cms = np.asarray([it["conf_mean"] for it in items], dtype=np.float64)
            recs = np.asarray([it["recalls"] for it in items], dtype=np.float64)  # (n_files_used, k)
            Ns = np.asarray([it["n"] for it in items], dtype=np.float64)
            rows.append(
                {
                    "bin": int(i),
                    "n_files": int(len(items)),
                    "n_mean": float(np.mean(Ns)),
                    "conf_mean": float(np.mean(cms)),
                    "conf_mean_std": float(np.std(cms)) if len(cms) > 1 else 0.0,
                    "acc": float(np.mean(accs)),
                    "acc_std": float(np.std(accs)) if len(accs) > 1 else 0.0,
                    "recall_std": float(np.mean(rstds)),
                    "recall_std_std": float(np.std(rstds)) if len(rstds) > 1 else 0.0,
                    "recalls": np.nanmean(recs, axis=0),
                    "recalls_std": np.nanstd(recs, axis=0) if recs.shape[0] > 1 else np.zeros((k_seen,), dtype=np.float64),
                }
            )
        rows = sorted(rows, key=lambda r: int(r["bin"]))
        return {
            "aggregate_mode": "mean_over_files",
            "k": int(k_seen),
            "option_ids": option_ids,
            "n_total": int(n_total),
            "n_files": int(len(per_file)),
            "conf_edges": global_edges,
            "bins": rows,
        }

    # Default: pooled across all samples
    conf = np.concatenate([o["conf"] for o in per_file], axis=0)
    yt = np.concatenate([o["yt"] for o in per_file], axis=0)
    yp = np.concatenate([o["yp"] for o in per_file], axis=0)

    idx_bins, edges = _bin_indices(conf, int(n_bins), str(binning_mode))
    rows = []
    for i, idx in enumerate(idx_bins):
        N = int(idx.size)
        if N < int(min_bin_n):
            continue
        acc = float(np.mean((yp[idx] == yt[idx]).astype(np.float64)))
        m = np.zeros((int(conf.size),), dtype=bool)
        m[idx] = True
        rec = _recall_by_class(yt, yp, int(k_seen), m)
        rstd = float(np.nanstd(rec))
        rows.append(
            {
                "bin": int(i),
                "n": int(N),
                "conf_min": float(np.min(conf[idx])) if N > 0 else float("nan"),
                "conf_max": float(np.max(conf[idx])) if N > 0 else float("nan"),
                "conf_mean": float(np.mean(conf[idx])),
                "acc": float(acc),
                "recall_std": float(rstd),
                "recalls": rec,
            }
        )
    return {
        "aggregate_mode": "pooled",
        "k": int(k_seen),
        "option_ids": option_ids,
        "n_total": int(conf.size),
        "n_files": int(len(per_file)),
        "conf_edges": edges,
        "bins": rows,
    }


def _print_table(rep: dict, table_mode: str = "summary"):
    option_ids = rep["option_ids"]
    mode = rep.get("aggregate_mode", "pooled")
    table_mode = str(table_mode or "summary").strip().lower()
    n_bins = int(len(rep.get("bins", []) or [])) or 10

    if mode == "mean_over_files":
        # If global edges exist, bins correspond to p10, p20, ..., p100 over pooled samples.
        edges = rep.get("conf_edges")
        has_edges = isinstance(edges, list) and len(edges) >= 2
        if table_mode == "full":
            hdr = "pct_hi\tbin\tn_files\tN_mean\tconf_mean\tacc_mean±std\trecall_std_mean±std\t" + "\t".join([f"R_{o}" for o in option_ids])
            print(hdr)
            for r in rep["bins"]:
                rec = r["recalls"]
                rec_str = "\t".join([f"{float(x):.3f}" if np.isfinite(x) else "nan" for x in rec])
                pct_hi = int((int(r["bin"]) + 1) * (100 / max(1, int(n_bins))))
                print(
                    f"{pct_hi}%\t{r['bin']}\t{r['n_files']}\t{r['n_mean']:.1f}\t{r['conf_mean']:.4f}\t"
                    f"{r['acc']:.4f}±{r['acc_std']:.4f}\t{r['recall_std']:.4f}±{r['recall_std_std']:.4f}\t{rec_str}"
                )
        else:
            print("pct_hi\tbin\tn_files\tN_mean\tconf_mean\tacc_mean±std\trecall_std_mean±std")
            for r in rep["bins"]:
                pct_hi = int((int(r["bin"]) + 1) * (100 / max(1, int(n_bins))))
                print(
                    f"{pct_hi}%\t{r['bin']}\t{r['n_files']}\t{r['n_mean']:.1f}\t{r['conf_mean']:.4f}\t"
                    f"{r['acc']:.4f}±{r['acc_std']:.4f}\t{r['recall_std']:.4f}±{r['recall_std_std']:.4f}"
                )
    else:
        if table_mode == "full":
            print("bin\tN\tconf_mean\tacc\trecall_std\t" + "\t".join([f"R_{o}" for o in option_ids]))
            for r in rep["bins"]:
                rec = r["recalls"]
                rec_str = "\t".join([f"{float(x):.3f}" if np.isfinite(x) else "nan" for x in rec])
                print(
                    f"{r['bin']}\t{r['n']}\t{r['conf_mean']:.4f}\t{r['acc']:.4f}\t{r['recall_std']:.4f}\t{rec_str}"
                )
        else:
            print("bin\tN\tconf_mean\tacc\trecall_std")
            for r in rep["bins"]:
                print(f"{r['bin']}\t{r['n']}\t{r['conf_mean']:.4f}\t{r['acc']:.4f}\t{r['recall_std']:.4f}")


def _save_plot(rep: dict, out_path: str, title: str, plot_mode: str = "rstd_only"):
    plt = _try_import_matplotlib()
    if plt is None:
        print("[warn] matplotlib not available; skipping plot.")
        return None
    _set_font()

    bins = rep["bins"]
    if not bins:
        print("[warn] no bins to plot.")
        return None

    option_ids = rep["option_ids"]
    xs = np.asarray([b["conf_mean"] for b in bins], dtype=np.float64)
    rstd = np.asarray([b["recall_std"] for b in bins], dtype=np.float64)
    acc = np.asarray([b["acc"] for b in bins], dtype=np.float64)
    recalls = np.asarray([b["recalls"] for b in bins], dtype=np.float64)  # (n_bins, k)
    mode = rep.get("aggregate_mode", "pooled")
    rstd_std = np.asarray([b.get("recall_std_std", 0.0) for b in bins], dtype=np.float64) if mode == "mean_over_files" else None
    acc_std = np.asarray([b.get("acc_std", 0.0) for b in bins], dtype=np.float64) if mode == "mean_over_files" else None

    plot_mode = str(plot_mode or "rstd_only").strip().lower()
    if plot_mode == "full":
        nrows = 2
    elif plot_mode == "rstd_and_acc":
        nrows = 2
    else:
        nrows = 1

    fig, axes = plt.subplots(nrows, 1, figsize=(9, 4.8 if nrows == 1 else 8), dpi=160, sharex=True)
    if nrows == 1:
        axes = [axes]

    ax1 = axes[0]
    ax1.plot(xs, rstd, marker="o", linewidth=1.8, color="#8E44AD", label="recall_std")
    if mode == "mean_over_files" and rstd_std is not None:
        ax1.fill_between(xs, rstd - rstd_std, rstd + rstd_std, color="#8E44AD", alpha=0.15, linewidth=0)
    ax1.set_ylabel("Recall std")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="best", fontsize=9)

    if nrows == 2:
        ax2 = axes[1]
        ax2.plot(xs, acc, marker="o", linewidth=1.6, color="#2E86C1", label="accuracy")
        if mode == "mean_over_files" and acc_std is not None:
            ax2.fill_between(xs, acc - acc_std, acc + acc_std, color="#2E86C1", alpha=0.12, linewidth=0)
        if plot_mode == "full":
            for ci, lab in enumerate(option_ids):
                ax2.plot(xs, recalls[:, ci], marker=".", linewidth=1.2, alpha=0.85, label=f"recall {lab}")
            ax2.legend(loc="best", fontsize=8, ncol=2)
        else:
            ax2.legend(loc="best", fontsize=9)
        ax2.set_ylabel("Accuracy")
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.set_xlabel("Confidence gap mean (top1 - top2) in bin")
    else:
        ax1.set_xlabel("Confidence gap mean (top1 - top2) in bin")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--results_dir", type=str, default="", help="Directory containing cached *.jsonl results.")
    ap.add_argument("--jsonl_paths", type=str, nargs="+", default=None, help="Explicit jsonl file paths to analyze (overrides --results_dir).")
    ap.add_argument("--jsonl_glob", type=str, default="*.jsonl", help="Glob to select jsonl files under results_dir (default: *.jsonl).")
    ap.add_argument("--run_idx", type=int, default=None, help="If set, only analyze files whose name contains 'run{idx}'.")
    ap.add_argument("--models", type=str, nargs="+", default=None,
                    help="Model names/ids. Auto-collect jsonl under computed results_dir for each model, then aggregate into ONE table.")
    ap.add_argument("--results_root", type=str, default="",
                    help="Root directory that contains results_* folders (default: this script's directory).")
    ap.add_argument("--eval_name", type=str, default="",
                    help="Eval name like 'arc,0,full' used to locate results when --models is set.")
    ap.add_argument("--out", type=str, default="conf_recall_bias.png", help="Output PNG path (relative paths saved into results_dir when available).")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--n_bins", type=int, default=10, help="Quantile bins (default: 10).")
    ap.add_argument("--min_bin_n", type=int, default=50, help="Skip bins with <N samples (default: 50).")
    ap.add_argument("--binning_mode", type=str, default="equal_count", choices=["equal_count", "quantile"],
                    help="equal_count: sort by confidence gap then split into equal-sized bins. quantile: use np.quantile thresholds.")
    ap.add_argument("--binning_scope", type=str, default="global", choices=["global", "per_file"],
                    help="For mean_over_files only. global: one set of percentile edges over ALL samples then apply to each file. per_file: compute deciles within each file.")
    ap.add_argument("--aggregate_mode", type=str, default="pooled", choices=["pooled", "mean_over_files"],
                    help="pooled: concatenate all samples. mean_over_files: compute bins per file then average metrics across files.")
    ap.add_argument("--table_mode", type=str, default="summary", choices=["summary", "full"],
                    help="summary: print only bin/conf/acc/recall_std. full: include per-option recalls.")
    ap.add_argument("--plot_mode", type=str, default="rstd_only", choices=["rstd_only", "rstd_and_acc", "full"],
                    help="rstd_only: only recall_std trend. rstd_and_acc: add accuracy. full: also show per-option recalls.")
    ap.add_argument("--save_plot", action="store_true", help="Save plot PNG.")
    ap.add_argument("--wandb", action="store_true", help="Log table scalars + plot to W&B.")
    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default="capde")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    # One-shot runner support (optional)
    ap.add_argument("--eval_clm_path", type=str, default="", help="Path to eval_clm.py (run if cache missing).")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--pretrained_model_path", type=str, default="")
    ap.add_argument("--eval_names", type=str, nargs="+", default=[])
    ap.add_argument("--option_id_set", type=str, default=None)
    ap.add_argument("--n_runs", type=int, default=1)
    ap.add_argument("--force", action="store_true")

    args, unknown = ap.parse_known_args()

    results_dir = str(args.results_dir).strip()
    files: List[str] = []

    # Auto-collect from multiple models → aggregate into ONE table
    if args.models:
        results_root = str(args.results_root).strip() or os.path.dirname(os.path.abspath(__file__))
        eval_name = str(args.eval_name).strip() or (str(args.eval_names[0]).strip() if args.eval_names else "")
        if not eval_name:
            raise SystemExit("When using --models, provide --eval_name (e.g., arc,0,full) or --eval_names.")
        model_list = [str(m).strip() for m in (args.models or []) if str(m).strip()]
        if not model_list:
            raise SystemExit("--models provided but empty.")

        by_model: Dict[str, List[str]] = {}
        for m in model_list:
            rdir = _compute_results_dir(results_root, eval_name, m, args.option_id_set)
            f = _discover_jsonl_files(rdir, jsonl_glob=str(args.jsonl_glob), run_idx=args.run_idx)
            by_model[m] = f
            files.extend(f)

        files = sorted(set(files))
        print(f"[info] selected models={len(model_list)}, jsonl_files={len(files)}")
        for m in model_list:
            print(f"[info]  - {m.split('/')[-1]}: {len(by_model.get(m, []))} files")

        # choose a default directory for saving relative outputs
        if not results_dir:
            for m in model_list:
                rdir = _compute_results_dir(results_root, eval_name, m, args.option_id_set)
                if os.path.isdir(rdir):
                    results_dir = rdir
                    break

    # One-shot runner (single model) only when not using --models
    if (not files) and (not results_dir) and str(args.pretrained_model_path).strip() and args.eval_names:
        eval_clm_path = str(args.eval_clm_path).strip() or os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_clm.py")
        code_dir = os.path.dirname(os.path.abspath(eval_clm_path))
        eval_name = str(args.eval_names[0])
        results_dir = _compute_results_dir(code_dir, eval_name, str(args.pretrained_model_path), args.option_id_set)

        files = _discover_jsonl_files(results_dir, jsonl_glob=str(args.jsonl_glob), run_idx=args.run_idx)
        if (not files) and (not bool(args.skip_eval)):
            cmd = [sys.executable, os.path.abspath(eval_clm_path)]
            cmd += ["--pretrained_model_path", str(args.pretrained_model_path)]
            cmd += ["--eval_names", eval_name]
            if args.option_id_set:
                cmd += ["--option_id_set", str(args.option_id_set)]
            if int(args.n_runs) != 1:
                cmd += ["--n_runs", str(int(args.n_runs))]
            if bool(args.force):
                cmd += ["--force"]
            if bool(args.wandb):
                cmd += ["--wandb"]
            if args.wandb_project:
                cmd += ["--wandb_project", str(args.wandb_project)]
            if args.wandb_entity:
                cmd += ["--wandb_entity", str(args.wandb_entity)]
            if args.wandb_run_name:
                cmd += ["--wandb_run_name", str(args.wandb_run_name)]
            cmd += list(unknown)
            print("==== Running eval_clm.py ====")
            print("cwd:", code_dir)
            print("cmd:", " ".join(cmd))
            subprocess.run(cmd, cwd=code_dir, check=True)

    # Resolve input files
    if not files:
        if args.jsonl_paths:
            files = [os.path.abspath(p) for p in (args.jsonl_paths or [])]
        else:
            if not results_dir:
                raise SystemExit("Provide --models, --jsonl_paths, --results_dir, or eval_clm args to infer it.")
            files = _discover_jsonl_files(results_dir, jsonl_glob=str(args.jsonl_glob), run_idx=args.run_idx)

    if not files:
        raise SystemExit("No jsonl files selected. Check --results_dir/--jsonl_glob/--run_idx or --jsonl_paths.")

    # For relative --out, prefer saving into results_dir when available.
    save_root = results_dir
    if not save_root and files:
        save_root = os.path.dirname(os.path.abspath(files[0]))

    if not results_dir:
        results_dir = save_root

    if not results_dir:
        raise SystemExit("Could not infer results_dir for saving outputs.")

    rep = _analyze(
        files=files,
        option_id_set=args.option_id_set,
        n_bins=int(args.n_bins),
        min_bin_n=int(args.min_bin_n),
        aggregate_mode=str(args.aggregate_mode),
        binning_mode=str(args.binning_mode),
        binning_scope=str(args.binning_scope),
    )

    _print_table(rep, table_mode=str(args.table_mode))

    out_plot = None
    out_path = str(args.out)
    if results_dir and (not os.path.isabs(out_path)):
        out_path = os.path.join(results_dir, out_path)
    if bool(args.save_plot):
        title = str(args.title) if args.title else f"Confidence gap vs recall bias ({platform.system()})"
        out_plot = _save_plot(rep, out_path, title, plot_mode=str(args.plot_mode))
        if out_plot:
            print(f"Saved plot: {out_plot}")

    if bool(args.wandb):
        wandb = _try_import_wandb()
        if wandb is None:
            print("[warn] wandb not available; skipping W&B logging.")
            return
        run = wandb.init(
            project=getattr(args, "wandb_project", None) or None,
            entity=getattr(args, "wandb_entity", None) or None,
            name=getattr(args, "wandb_run_name", None) or None,
            job_type="conf_recall_bias",
            reinit=True,
        )
        try:
            # log scalars per bin
            for r in rep["bins"]:
                n_val = float(r.get("n", r.get("n_mean", 0.0)))
                wandb.log({
                    "bin": int(r["bin"]),
                    "n": n_val,
                    "conf_mean": float(r["conf_mean"]),
                    "acc": float(r["acc"]),
                    "recall_std": float(r["recall_std"]),
                })
            if out_plot:
                wandb.log({"plots/conf_recall_bias": wandb.Image(out_plot)})
        except Exception as e:
            print(f"[warn] W&B logging failed: {e}")
        try:
            run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()

