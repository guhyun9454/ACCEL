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

# Pinned to the configuration the paper reports, not to whatever a log line
# happened to show. Figure 1: "For our proposed framework (ACCEL), we fix the
# calibration prefix ratio at alpha = 2% and vary the threshold percentile
# beta in {2,5,10,20,30,40,50,60,70,80}."
#
# Mapping into three_curves_points.json, verified against eval_clm.py:
#   curves.ours_pride.by_alpha is keyed by the PriDe prefix (pride_alphas),
#   and `p` inside each block is the threshold percentile -- it is passed to
#   get_heur_stats_by_th1_p(cobjs, th1_p) with sweep_key="th1_p".
# So the paper's setting is by_alpha["2"] at p = beta.
#
# Variant: eval_clm.py defines PRIMARY_OURS_LABEL = "th1/sqrt2" and
# LEGACY_OURS_LABEL = "th1/2". The primary one is what curves.ours carries, so
# that is the method the paper plots; "th1/2" is a superseded variant.
PRIDE_PREFIX_PCT = 2.0        # PriDe's own alpha sweep, read from default_pride.p
ACCEL_VARIANT = "th1/sqrt2"   # PRIMARY_OURS_LABEL
ACCEL_PREFIX_KEY = "2"        # by_alpha key = calibration prefix alpha = 2%
ACCEL_TH1_PCT = 2.0           # beta, the cheapest point of the paper's sweep


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

    by_alpha = (curves.get("ours_pride") or {}).get("by_alpha") or {}
    variant = None
    for key in (ACCEL_PREFIX_KEY, str(float(ACCEL_PREFIX_KEY))):
        if key in by_alpha:
            variant = by_alpha[key].get(ACCEL_VARIANT)
            break
    if variant and variant.get("p"):
        i = _nearest_index(variant["p"], ACCEL_TH1_PCT)
        out["accel"] = {
            "cost": float(variant["cost"][i]),
            "acc": _as_fraction(variant["acc"][i]),
            "recall_std": float(variant["recall_std"][i]),
            "prefix_pct": float(ACCEL_PREFIX_KEY),
            "th1_pct": float(variant["p"][i]),
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


def find_sweep(results_root: str, task: str, model: str) -> Optional[str]:
    pattern = os.path.join(results_root, f"results_{task}", f"0s_{model}",
                           f"{task}_full_id-*", f"{task}_three_curves_points.json")
    hits = [p for p in glob.glob(pattern) if "__" not in os.path.basename(os.path.dirname(p))]
    return hits[0] if hits else None


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
