"""Write a CalibraEval-calibrated copy of a cached results directory.

Purpose: test whether ACCEL and CalibraEval are complementary. Rather than
reimplementing ACCEL's routing (residual marginalization, empirical prior,
Latin-square scheduling all live in eval_clm.py), this calibrates the *input*
and leaves the pipeline untouched: fit CalibraEval's per-slot map on the cached
observed distributions, write a calibrated copy of the cache under a new
result_tag, then rerun eval_clm.py against it. eval_clm reuses a complete cached
run (`maybe_use_cached`), so no inference is repeated.

The calibration is label-free and fitted per (model, dataset) on the whole test
set, matching the paper protocol used in calibraeval_mcq.py.

Usage:
    python calibrate_cache.py \
        --src results_arc/0s_Llama-3.1-8B/arc_full_id-ABCD \
        --dst results_arc/0s_Llama-3.1-8B/arc_full_id-ABCD__ce_calibrated

Then rerun with `--result_tag ce_calibrated`.
"""

import argparse
import json
import os
import shutil

import numpy as np

from calibraeval_mcq import OrderPreservingCalibrator, group_by_run_index


def read_records(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def probs_of(rec):
    """Return the (views, k) block if this record carries a usable one."""
    if rec.get("type") != "result":
        return None
    arr = (rec.get("data") or {}).get("probs")
    if not arr:
        return None
    arr = np.asarray(arr, dtype=np.float64)
    return arr if arr.ndim == 2 and arr.shape[0] >= 2 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="cached results dir to read")
    parser.add_argument("--dst", required=True, help="dir to write the calibrated copy into")
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    run_files = [os.path.join(args.src, f) for f in sorted(os.listdir(args.src))
                 if f.endswith(".jsonl") and "run" in f and "curve" not in f]
    if not run_files:
        raise SystemExit(f"no run files under {args.src}")
    os.makedirs(args.dst, exist_ok=True)

    for run_key, files in sorted(group_by_run_index(run_files).items()):
        # One map per run, fitted on every item of that run across all subjects —
        # the same estimation set calibraeval_mcq.py uses under --estimation full.
        blocks = []
        for path in files:
            for rec in read_records(path):
                arr = probs_of(rec)
                if arr is not None:
                    blocks.append(arr)
        if not blocks:
            print(f"run{run_key}: nothing usable, skipped")
            continue
        views = min(b.shape[0] for b in blocks)
        Q = np.stack([b[:views] for b in blocks])
        cal = OrderPreservingCalibrator(lam=args.lam, max_epochs=args.max_epochs,
                                        tol=args.tol).fit(Q)

        written = 0
        for path in files:
            out_path = os.path.join(args.dst, os.path.basename(path))
            with open(out_path, "w", encoding="utf-8") as out:
                for rec in read_records(path):
                    arr = probs_of(rec)
                    if arr is not None:
                        # calibrate() renormalizes each view, so the record stays a
                        # distribution over option IDs and everything downstream
                        # reads it exactly as before.
                        rec["data"]["probs"] = cal.calibrate(arr).tolist()
                        written += 1
                    out.write(json.dumps(rec) + "\n")
        print(f"run{run_key}: {len(files)} files, {written} records calibrated, "
              f"epochs={cal.n_epochs_run_} converged={cal.converged_}")

    # eval_clm regenerates curves/plots; only the run caches need to be present.
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
