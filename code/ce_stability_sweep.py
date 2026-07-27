"""Tally ACCEL vs CalibraEval at several optimizer stopping points.

Why this exists: the calibrator's objective keeps decreasing past any practical
epoch budget and the resulting RStd is not monotone in the epoch count, so a
tally read at one stopping point is not reproducible. Reporting the same tally
at several stopping points makes the dependence visible — if the counts barely
move, the conclusion is safe to state; if they swing, it is not.

One fit per (model, dataset, run) covers every checkpoint, so the cost is the
longest checkpoint rather than the sum.

Usage:
    python ce_stability_sweep.py --results_root . --tasks arc csqa mmlu \
        --models A B C --checkpoints 500 2000 10000 --out ce_stability.json
"""

import argparse
import json
import os

import numpy as np

from calibraeval_mcq import (
    OrderPreservingCalibrator,
    group_by_run_index,
    load_cached_run,
    recall_std,
)
from compare_cost_axis import find_sweep, load_sweep


def run_files_for(results_root, task, model):
    sweep = find_sweep(results_root, task, model)
    if not sweep:
        return []
    d = os.path.dirname(sweep)
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if f.endswith(".jsonl") and "run" in f and "curve" not in f]


def cell_recall_stds(files, checkpoints, lam):
    """{checkpoint: macro recall_std} for CalibraEval@1, averaged over runs."""
    per_ckpt = {c: [] for c in checkpoints}
    for _, run_files in sorted(group_by_run_index(files).items()):
        parts = [load_cached_run(p) for p in run_files]
        Q = np.concatenate([q for q, _ in parts], axis=0)
        y = np.concatenate([t for _, t in parts], axis=0)
        sizes = [len(q) for q, _ in parts]
        k = Q.shape[2]
        cal = OrderPreservingCalibrator(lam=lam)
        for epoch, fitted in cal.fit_with_checkpoints(Q, checkpoints):
            preds = np.argmax(fitted.calibrate(Q[:, 0, :]), axis=1)
            # macro over subject blocks, matching eval_clm.py for MMLU
            start, blocks = 0, []
            for size in sizes:
                r = recall_std(y[start:start + size], preds[start:start + size], k)
                if np.isfinite(r):
                    blocks.append(r)
                start += size
            per_ckpt[epoch].append(float(np.mean(blocks)) if blocks else float("nan"))
    return {c: float(np.mean(v)) for c, v in per_ckpt.items() if v}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default=".")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[500, 2000, 10000])
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cells, missing = [], []
    for model in args.models:
        for task in args.tasks:
            files = run_files_for(args.results_root, task, model)
            sweep = find_sweep(args.results_root, task, model)
            if not files or not sweep:
                missing.append(f"{model}/{task}")
                continue
            accel = load_sweep(sweep).get("accel")
            if not accel:
                missing.append(f"{model}/{task}: no accel point")
                continue
            ce = cell_recall_stds(files, args.checkpoints, args.lam)
            cells.append({"model": model, "task": task,
                          "accel_recall_std": accel["recall_std"],
                          "accel_cost": accel["cost"],
                          "calibraeval_recall_std": {str(c): v for c, v in ce.items()}})
            print(f"{model:<26}{task:<6} ACCEL={accel['recall_std']:.4f}  " +
                  "  ".join(f"CE@{c}={ce.get(c, float('nan')):.4f}" for c in args.checkpoints),
                  flush=True)

    print(f"\nACCEL wins (lower RStd) out of {len(cells)} cells, by stopping point:")
    for c in args.checkpoints:
        wins = sum(1 for cell in cells
                   if cell["accel_recall_std"] < cell["calibraeval_recall_std"].get(str(c), np.inf))
        print(f"  epoch {c:>6}: {wins}/{len(cells)}")
    if missing:
        print("\nMISSING:")
        for m in missing:
            print(f"  - {m}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"checkpoints": args.checkpoints, "cells": cells, "missing": missing},
                      f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
