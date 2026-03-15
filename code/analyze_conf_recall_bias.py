#!/usr/bin/env python3
import argparse
import glob
import json
import os
import platform
import subprocess
import sys
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


def _discover_jsonl_files(results_dir: str) -> List[str]:
    pats = [
        os.path.join(results_dir, "*.jsonl"),
        os.path.join(results_dir, "**", "*.jsonl"),
    ]
    files = []
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


def _extract_base_probs_and_pred(data: dict) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Supports multiple cache shapes:
    - base cache: probs = [k] (float list)
    - perm/full cache: probs = [n_perm][k] (nested list); we use probs[0] as identity row
    - cyclic cache: probs = [k][k] (k rotations); we use probs[0]
    """
    probs = data.get("probs", None)
    if not isinstance(probs, list) or len(probs) == 0:
        return None, None
    # nested?
    if isinstance(probs[0], list):
        row0 = probs[0]
        if not isinstance(row0, list) or len(row0) == 0:
            return None, None
        bp = np.asarray(row0, dtype=np.float64)
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
):
    confs: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    k_seen: Optional[int] = None
    option_ids: Optional[List[str]] = None

    for fp in files:
        for d in _iter_result_rows(fp):
            ideal = d.get("ideal", None)
            if not isinstance(ideal, str):
                continue
            options = d.get("options", None)
            if isinstance(options, list) and len(options) > 0:
                k = int(len(options))
            else:
                # fallback: infer from probs length later
                k = -1

            bp, pred_idx = _extract_base_probs_and_pred(d)
            if bp is None or pred_idx is None:
                continue

            if k <= 0:
                k = int(bp.size)
            if k <= 0:
                continue

            if k_seen is None:
                k_seen = k
                if option_id_set is not None:
                    option_ids = list(str(option_id_set))
                    if len(option_ids) != k_seen:
                        raise ValueError(f"--option_id_set length must be {k_seen}, got {len(option_ids)}: {option_id_set}")
                else:
                    option_ids = list("ABCDE"[:k_seen]) if k_seen in (4, 5) else [str(i) for i in range(k_seen)]

            if k != k_seen:
                continue
            assert option_ids is not None
            if ideal not in option_ids:
                # likely custom option_id_set; user should pass it explicitly
                continue

            confs.append(_gap_top1_top2(bp))
            y_true.append(int(option_ids.index(ideal)))
            y_pred.append(int(pred_idx))

    if k_seen is None or len(confs) == 0:
        raise ValueError("No usable samples found. Check results_dir and (if used) --option_id_set.")

    conf = np.asarray(confs, dtype=np.float64)
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)

    edges = _quantile_bin_edges(conf, int(n_bins))

    rows = []
    for i in range(int(n_bins)):
        lo, hi = edges[i], edges[i + 1]
        if i < int(n_bins) - 1:
            m = (conf >= lo) & (conf < hi)
        else:
            m = (conf >= lo) & (conf <= hi)
        N = int(np.sum(m))
        if N < int(min_bin_n):
            continue
        acc = float(np.mean((yp[m] == yt[m]).astype(np.float64)))
        rec = _recall_by_class(yt, yp, int(k_seen), m)
        rstd = float(np.nanstd(rec))
        rows.append(
            {
                "bin": int(i),
                "n": int(N),
                "conf_min": float(lo),
                "conf_max": float(hi),
                "conf_mean": float(np.mean(conf[m])),
                "acc": float(acc),
                "recall_std": float(rstd),
                "recalls": rec,
            }
        )

    return {
        "k": int(k_seen),
        "option_ids": option_ids,
        "n_total": int(conf.size),
        "conf_edges": edges,
        "bins": rows,
    }


def _print_table(rep: dict):
    option_ids = rep["option_ids"]
    print("bin\tN\tconf_mean\tacc\trecall_std\t" + "\t".join([f"R_{o}" for o in option_ids]))
    for r in rep["bins"]:
        rec = r["recalls"]
        rec_str = "\t".join([f"{float(x):.3f}" if np.isfinite(x) else "nan" for x in rec])
        print(
            f"{r['bin']}\t{r['n']}\t{r['conf_mean']:.4f}\t{r['acc']:.4f}\t{r['recall_std']:.4f}\t{rec_str}"
        )


def _save_plot(rep: dict, out_path: str, title: str):
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), dpi=160, sharex=True)

    ax1.plot(xs, rstd, marker="o", linewidth=1.8, color="#8E44AD", label="recall_std (across options)")
    ax1.set_ylabel("Recall std")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="best", fontsize=9)

    ax2.plot(xs, acc, marker="o", linewidth=1.6, color="#2E86C1", label="accuracy")
    for ci, lab in enumerate(option_ids):
        ax2.plot(xs, recalls[:, ci], marker=".", linewidth=1.2, alpha=0.85, label=f"recall {lab}")
    ax2.set_xlabel("Confidence gap mean (top1 - top2) in bin")
    ax2.set_ylabel("Accuracy / Recall")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="best", fontsize=8, ncol=2)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--results_dir", type=str, default="", help="Directory containing cached *.jsonl results.")
    ap.add_argument("--out", type=str, default="conf_recall_bias.png", help="Output PNG path (relative paths saved into results_dir when available).")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--n_bins", type=int, default=10, help="Quantile bins (default: 10).")
    ap.add_argument("--min_bin_n", type=int, default=50, help="Skip bins with <N samples (default: 50).")
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
    if (not results_dir) and str(args.pretrained_model_path).strip() and args.eval_names:
        eval_clm_path = str(args.eval_clm_path).strip() or os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_clm.py")
        code_dir = os.path.dirname(os.path.abspath(eval_clm_path))
        eval_name = str(args.eval_names[0])
        results_dir = _compute_results_dir(code_dir, eval_name, str(args.pretrained_model_path), args.option_id_set)

        files = _discover_jsonl_files(results_dir)
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

    if not results_dir:
        raise SystemExit("Provide --results_dir, or provide eval_clm args to infer it.")

    files = _discover_jsonl_files(results_dir)
    if not files:
        raise SystemExit(f"No jsonl files found under: {results_dir}")

    rep = _analyze(
        files=files,
        option_id_set=args.option_id_set,
        n_bins=int(args.n_bins),
        min_bin_n=int(args.min_bin_n),
    )

    _print_table(rep)

    out_plot = None
    out_path = str(args.out)
    if results_dir and (not os.path.isabs(out_path)):
        out_path = os.path.join(results_dir, out_path)
    if bool(args.save_plot):
        title = str(args.title) if args.title else f"Confidence gap vs recall bias ({platform.system()})"
        out_plot = _save_plot(rep, out_path, title)
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
                wandb.log({
                    "bin": int(r["bin"]),
                    "n": int(r["n"]),
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

