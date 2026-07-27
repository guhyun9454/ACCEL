"""Put Baseline / PriDe / ACCEL / cyclic / CalibraEval on one cost axis.

The rebuttal claim is about the cost-accuracy-RStd trade-off, but the numbers
live in two places: ACCEL and PriDe in the sweep that `eval_clm.py` writes to
`<task>_three_curves_points.json`, and CalibraEval in the json produced by
`calibraeval_mcq.py`. This merges them per (model, dataset).

Two unit traps in the sweep file, both handled here:
* `acc` is stored as a **percentage** (78.88) while `recall_std` is a fraction
  (0.0300). Everything is emitted as fractions.
* `curves.ours_pride.by_alpha[<prefix %>][<variant>]` is indexed by the PriDe
  prefix, and within it `p` is the th1 percentile — so the operating point the
  logs call `ours_pride_th12_α0.5_2%` is `by_alpha["2"]["th1/2"]` at `p == 0.5`.

Usage:
    python compare_cost_axis.py --results_root . --tasks arc csqa \
        --models Llama-3.1-8B Qwen2.5-7B-Instruct \
        --calibraeval_dir calibraeval_out_paper
"""

import argparse
import glob
import json
import os
from typing import Optional

# ACCEL = the empirical-residual + Latin-square method, which lives in
# `curves.empirical_pride` -- NOT in `curves.ours_pride`. The latter holds the
# threshold-cascade variants (th1/2, th1/sqrt2, online sqrt) from the earlier ARR
# line; they carry the word "ours" but are a different method.
#
# Identified by matching the run command
# (--empirical_pride --empirical_residual_model empirical
#  --empirical_stage_schedule flat --empirical_transition_mode latin)
# and paper §3.5 ("flat schedule ... online running beta-th percentile") against
# the log family `empirical_pride_pct_flat_online_latin_a{alpha}_{beta}%`.
# Values were then cross-checked: the json curve and those log lines agree to 4
# decimals at every beta, so this is confirmed by two independent sources rather
# than by assumption.
#
# `empirical_pride.by_alpha` has a single key "2" (the calibration prefix
# alpha=2% of paper Fig. 1), and `p` inside `primary` is beta.
PRIDE_PREFIX_PCT = 2.0        # PriDe's own alpha sweep, read from default_pride.p
ACCEL_PREFIX_KEY = "2"        # calibration prefix alpha = 2%
ACCEL_BLOCK = "primary"
ACCEL_BETA = 2.0              # cheapest point of the paper's beta sweep


def _as_fraction(acc) -> float:
    """Sweep files store accuracy in percent; CalibraEval json stores fractions."""
    acc = float(acc)
    return acc / 100.0 if acc > 1.0 else acc


def _nearest_index(values, target) -> int:
    return min(range(len(values)), key=lambda i: abs(float(values[i]) - float(target)))


def load_sweep(path: str) -> dict:
    """Extract the operating points of interest from one three_curves_points.json."""
    with open(path, "r", encoding="utf-8") as f:
        curves = json.load(f)["curves"]
    out = {}

    cyclic = curves.get("cyclic")
    if cyclic and cyclic.get("cost"):
        # cost 1.0 is the undebiased single view; the last point is full cyclic.
        first, last = 0, len(cyclic["cost"]) - 1
        out["baseline"] = {
            "cost": float(cyclic["cost"][first]),
            "acc": _as_fraction(cyclic["acc"][first]),
            "recall_std": float(cyclic["recall_std"][first]),
        }
        out["cyclic"] = {
            "cost": float(cyclic["cost"][last]),
            "acc": _as_fraction(cyclic["acc"][last]),
            "recall_std": float(cyclic["recall_std"][last]),
        }

    pride = curves.get("default_pride")
    if pride and pride.get("p"):
        i = _nearest_index(pride["p"], PRIDE_PREFIX_PCT)
        out["pride"] = {
            "cost": float(pride["cost"][i]),
            "acc": _as_fraction(pride["acc"][i]),
            "recall_std": float(pride["recall_std"][i]),
            "alpha_pct": float(pride["p"][i]),
        }

    emp = curves.get("empirical_pride")
    if emp:
        by_alpha = emp.get("by_alpha") or {}
        block = None
        for key in (ACCEL_PREFIX_KEY, str(float(ACCEL_PREFIX_KEY))):
            if key in by_alpha:
                block = by_alpha[key].get(ACCEL_BLOCK)
                break
        if block and block.get("p"):
            i = _nearest_index(block["p"], ACCEL_BETA)
            out["accel"] = {
                "cost": float(block["cost"][i]),
                "acc": _as_fraction(block["acc"][i]),
                "recall_std": float(block["recall_std"][i]),
                "prefix_pct": float(ACCEL_PREFIX_KEY),
                "beta": float(block["p"][i]),
                "schedule": emp.get("threshold_schedule"),
                "percentile_mode": emp.get("percentile_mode"),
                "residual_model": emp.get("residual_model"),
                "transition_mode": emp.get("transition_mode"),
            }
    return out


def load_calibraeval(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        mean = json.load(f)["mean"]
    out = {}
    for name, m in mean.items():
        if name.startswith("calibraeval@"):
            out[name] = {"cost": float(m["cost"]), "acc": float(m["acc"]),
                         "recall_std": float(m["recall_std"])}
    return out


def find_sweep(results_root: str, task: str, model: str,
               require_empirical: bool = True) -> Optional[str]:
    """Locate the sweep file that actually holds the paper's method.

    The untagged `<task>_full_id-*` directory is an older run whose
    three_curves_points.json has no `empirical_pride` curve at all; only the
    `__<result_tag>` run does. I previously filtered the tagged directories out
    on the assumption that untagged was canonical, which meant the file
    containing ACCEL was never opened. Selection is now by content -- whichever
    file carries the curve -- rather than by directory naming.
    """
    pattern = os.path.join(results_root, f"results_{task}", f"0s_{model}",
                           f"{task}_full_id-*", f"{task}_three_curves_points.json")
    hits = sorted(glob.glob(pattern))
    if not require_empirical:
        return hits[0] if hits else None
    for path in hits:
        try:
            with open(path, "r", encoding="utf-8") as f:
                if "empirical_pride" in (json.load(f).get("curves") or {}):
                    return path
        except (ValueError, OSError):
            continue
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default=".")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--calibraeval_dir", default="calibraeval_out_paper")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows, missing = [], []
    for model in args.models:
        for task in args.tasks:
            sweep_path = find_sweep(args.results_root, task, model)
            if not sweep_path:
                missing.append(f"{model}/{task}: no three_curves_points.json")
                continue
            entry = {"model": model, "task": task, "methods": load_sweep(sweep_path)}

            ce_path = os.path.join(args.calibraeval_dir, f"{model}_{task}.json")
            if os.path.exists(ce_path):
                entry["methods"].update(load_calibraeval(ce_path))
            else:
                missing.append(f"{model}/{task}: no CalibraEval json at {ce_path}")
            rows.append(entry)

    order = ["baseline", "calibraeval@1", "pride", "accel", "calibraeval@4",
             "calibraeval@5", "cyclic"]
    header = f"{'model':<26} {'task':<6} {'method':<15} {'cost':>6} {'acc':>8} {'RStd':>8}"
    print(header)
    print("-" * len(header))
    for entry in rows:
        for name in order:
            m = entry["methods"].get(name)
            if not m:
                continue
            print(f"{entry['model']:<26} {entry['task']:<6} {name:<15} "
                  f"{m['cost']:>6.3f} {m['acc']:>8.4f} {m['recall_std']:>8.4f}")
        print()

    if missing:
        print("MISSING (not silently dropped):")
        for item in missing:
            print(f"  - {item}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "missing": missing}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
