import os
import argparse
import json
from glob import glob
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve_dir", type=str, default=None,
                        help="Directory containing *_beta_curve.jsonl files "
                             "(e.g., results_arc/0s_MODEL/arc_full[_id-ABCD])")
    parser.add_argument("--task", type=str, default="arc", choices=["arc", "mmlu", "csqa"])
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--model_name", type=str, default=None,
                        help="Leaf model name (e.g., Llama-3.2-1B-Instruct)")
    parser.add_argument("--option_id_set", type=str, default=None,
                        help="Optional ID set suffix (e.g., ABCD or ABCDE)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path. Defaults to '<curve_dir>/beta_curve.png'")
    return parser.parse_args()


def resolve_curve_dir(args) -> str:
    if args.curve_dir:
        return args.curve_dir
    if not args.model_name:
        raise ValueError("Either --curve_dir or --model_name must be provided.")
    base = f"results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full"
    if args.option_id_set:
        base += f"_id-{args.option_id_set}"
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Curve directory not found: {base}")
    return base


def load_curves(curve_dir: str) -> List[Dict[str, Any]]:
    paths = sorted(glob(os.path.join(curve_dir, "*_beta_curve.jsonl")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No curve files found in: {curve_dir}")
    curves = []
    for p in paths:
        with open(p, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and "betas" in obj and "cyclic" in obj and "full" in obj:
                    curves.append(obj)
    if len(curves) == 0:
        raise RuntimeError(f"No valid curve objects found in: {curve_dir}")
    return curves


def aggregate_curves(curves: List[Dict[str, Any]]) -> Tuple[List[float], List[float], List[float], float]:
    # Assume all curves share same betas length/order
    betas = curves[0]["betas"]
    num_pts = len(betas)
    cyc_costs = np.zeros(num_pts, dtype=np.float64)
    cyc_accs = np.zeros(num_pts, dtype=np.float64)
    full_costs = np.zeros(num_pts, dtype=np.float64)
    full_accs = np.zeros(num_pts, dtype=np.float64)
    defaults = []

    for obj in curves:
        defaults.append(float(obj.get("default_accuracy", float("nan"))))
        c_costs = np.asarray(obj["cyclic"]["costs"], dtype=np.float64)
        c_accs = np.asarray(obj["cyclic"]["accuracies"], dtype=np.float64)
        f_costs = np.asarray(obj["full"]["costs"], dtype=np.float64)
        f_accs = np.asarray(obj["full"]["accuracies"], dtype=np.float64)
        cyc_costs += c_costs
        cyc_accs += c_accs
        full_costs += f_costs
        full_accs += f_accs

    n = float(len(curves))
    cyc_costs /= n
    cyc_accs /= n
    full_costs /= n
    full_accs /= n
    default_acc = float(np.nanmean(np.asarray(defaults, dtype=np.float64)))
    return betas, cyc_costs.tolist(), cyc_accs.tolist(), default_acc, full_costs.tolist(), full_accs.tolist()


def plot_curves(betas: List[float],
                cyc_costs: List[float], cyc_accs: List[float],
                full_costs: List[float], full_accs: List[float],
                default_acc: float,
                out_path: str):
    plt.figure(figsize=(7.5, 5.0), dpi=160)
    # Lines with markers (11 points each)
    plt.plot(cyc_costs, cyc_accs, marker='o', label='Cyclic (k rotations)')
    plt.plot(full_costs, full_accs, marker='o', label='Full (k! permutations)')
    # Default single point (cost=1)
    plt.scatter([1.0], [default_acc], marker='*', s=180, c='black', label='Default')
    plt.xlabel("Computational Cost (× of default)")
    plt.ylabel("Accuracy")
    plt.xscale('log', base=2)
    plt.title("Accuracy vs. Computational Cost")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    args = parse_args()
    curve_dir = resolve_curve_dir(args)
    curves = load_curves(curve_dir)
    betas, cyc_costs, cyc_accs, default_acc, full_costs, full_accs = aggregate_curves(curves)
    out_path = args.output or os.path.join(curve_dir, "beta_curve.png")
    plot_curves(betas, cyc_costs, cyc_accs, full_costs, full_accs, default_acc, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


