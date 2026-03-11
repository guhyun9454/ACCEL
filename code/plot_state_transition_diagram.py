#!/usr/bin/env python3
import argparse
import json
import math
import os
import platform
import subprocess
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def _try_import_wandb():
    try:
        import wandb  # type: ignore
        return wandb
    except Exception:
        return None


def _set_font():
    """
    Try to set a readable font across OSes.
    """
    try:
        sys_name = platform.system()
        # Use OS defaults; keep minus signs.
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def draw_state_column(ax, x_pos, title, blocks):
    """
    Draw one column of stacked blocks at x_pos.
    """
    # 컬럼 제목
    ax.text(
        x_pos + 0.4,
        8.5,
        title,
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

    total_val = sum([float(b["h"]) for b in blocks]) if blocks else 1.0
    y_curr = 2.0

    # 바닥에서부터 위로 쌓기 위해 역순으로 그림
    for block in reversed(blocks):
        h_norm = (float(block["h"]) / total_val) * 6.0

        rect = patches.Rectangle(
            (x_pos, y_curr),
            0.8,
            h_norm,
            linewidth=2,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)

        text_y = y_curr + (h_norm / 2)
        ax.text(x_pos + 1.0, text_y, str(block["text"]), va="center", fontsize=12)

        y_curr += h_norm


def _rotations(k: int):
    return [tuple((i + s) % k for i in range(k)) for s in range(k)]


def _aggregate_probs_over_permutations(probs_seq, permuted_indices, k: int):
    """
    probs_seq: list/array of length = (#permutations used)
      each element: length k (letter-space probs)
    permuted_indices: list of permutations p where p[j] is content-index at letter position j.
    Returns: agg (k,) content-space aggregated probs (mean over permutations)
    """
    import numpy as np

    agg = np.zeros(k, dtype=np.float64)
    for perm_idx, p in enumerate(permuted_indices):
        letter_probs = np.asarray(probs_seq[perm_idx], dtype=np.float64)
        for j in range(k):
            agg[p[j]] += letter_probs[j]
    if len(permuted_indices) > 0:
        agg /= float(len(permuted_indices))
    return agg


def _probe_shift_put_top2_into_top1_slot(base_probs, k: int):
    """shift s = (t2 - t1) mod k, with s!=0 when possible."""
    import numpy as np

    bp = np.asarray(base_probs, dtype=np.float64)
    sidx = np.argsort(bp)[::-1]
    t1 = int(sidx[0])
    t2 = int(sidx[1]) if len(sidx) > 1 else int(sidx[0])
    s = int((t2 - t1) % int(k))
    if s == 0:
        s = 1 if k > 1 else 0
    return s


def _infer_perm_list(k: int, perm_count: int):
    from itertools import permutations

    if perm_count == math.factorial(k):
        return list(sorted(permutations(range(k)))), True
    if perm_count == k:
        return _rotations(k), False
    # unknown: prefix of full
    full = list(sorted(permutations(range(k))))
    if perm_count <= len(full):
        return full[:perm_count], False
    raise ValueError(f"Unsupported perm_count={perm_count} for k={k}")


def _read_results_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") == "result":
                rows.append(obj.get("data", {}) or {})
    rows.sort(key=lambda d: int(d.get("idx", 0)))
    return rows


def _compute_results_dir(code_dir: str, eval_name: str, pretrained_model_path: str, option_id_set: str | None):
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


def _compute_transition_counts(results_dir: str, n_runs: int, option_id_set: str | None):
    """
    Compute counts for:
      - Initial: (base)
      - OnlyFlip: (base, probe2)
      - Cyclic: (base, probe2, cyclic)
    Aggregated over all subject files in results_dir.
    """
    # Pick one run by default to avoid mixing run distributions
    run_suffix = "_run0" if int(n_runs) > 1 else ""
    pattern = os.path.join(results_dir, f"*{run_suffix}.jsonl")
    files = [p for p in sorted(glob.glob(pattern)) if p.endswith(".jsonl") and not p.endswith("_curve.jsonl") and not p.endswith("_pride_curve.jsonl")]
    if not files:
        # fallback: any jsonl
        files = [p for p in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))) if p.endswith(".jsonl") and not p.endswith("_curve.jsonl") and not p.endswith("_pride_curve.jsonl")]
    if not files:
        raise FileNotFoundError(f"No result jsonl files found under: {results_dir}")

    # Determine k from the first file
    first = _read_results_jsonl(files[0])
    if not first:
        raise ValueError(f"Empty results: {files[0]}")
    k = int(len(first[0].get("options", []) or []))
    if k <= 0:
        raise ValueError("Cannot infer k from results.")
    option_ids = list(option_id_set) if option_id_set else (list("ABCDE"[:k]) if k in (4, 5) else [str(i) for i in range(k)])

    counts_init = {(0,): 0, (1,): 0}
    counts_flip = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    counts_cyc = {t: 0 for t in [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]}

    for fp in files:
        rows = _read_results_jsonl(fp)
        if not rows:
            continue
        probs0 = rows[0].get("probs", None)
        if not isinstance(probs0, list):
            continue
        perm_count = len(probs0)
        perm_list, _ = _infer_perm_list(k, perm_count)
        identity = tuple(range(k))
        identity_idx = perm_list.index(identity) if identity in perm_list else 0

        # cyclic indices within perm_list
        cyc_perms = _rotations(k)
        cyc_idxs = [perm_list.index(p) for p in cyc_perms if p in perm_list]
        cyc_perm_list = [perm_list[i] for i in cyc_idxs]

        for d in rows:
            probs_seq = d.get("probs", None)
            ideal = str(d.get("ideal"))
            if not isinstance(probs_seq, list) or len(probs_seq) != len(perm_list):
                continue
            if ideal not in option_ids:
                continue

            # base (identity prompt only)
            base_row = probs_seq[identity_idx]
            agg_base = _aggregate_probs_over_permutations([base_row], [perm_list[identity_idx]], k)
            pred_base = option_ids[int(__import__("numpy").argmax(agg_base))]
            base_correct = 1 if pred_base == ideal else 0

            # probe2 (mean of base + per-sample probe shift)
            shift = _probe_shift_put_top2_into_top1_slot(base_row, k)
            probe_perm = tuple((i + shift) % k for i in range(k))
            probe_idx = perm_list.index(probe_perm) if probe_perm in perm_list else identity_idx
            agg_probe = _aggregate_probs_over_permutations([probs_seq[probe_idx]], [perm_list[probe_idx]], k)
            mean_probs = (agg_base + agg_probe) / 2.0
            pred_probe2 = option_ids[int(__import__("numpy").argmax(mean_probs))]
            probe2_correct = 1 if pred_probe2 == ideal else 0

            # cyclic ensemble (all rotations)
            if cyc_idxs:
                probs_c = [probs_seq[i] for i in cyc_idxs]
                agg_cyc = _aggregate_probs_over_permutations(probs_c, cyc_perm_list, k)
                pred_cyc = option_ids[int(__import__("numpy").argmax(agg_cyc))]
            else:
                pred_cyc = pred_base
            cyc_correct = 1 if pred_cyc == ideal else 0

            counts_init[(base_correct,)] += 1
            counts_flip[(base_correct, probe2_correct)] += 1
            counts_cyc[(base_correct, probe2_correct, cyc_correct)] += 1

    return k, counts_init, counts_flip, counts_cyc


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--out", type=str, default="state_transition_diagram.png", help="Output PNG path (default saved into results_dir when eval args provided)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--width", type=float, default=14.0)
    ap.add_argument("--height", type=float, default=6.0)
    # Optional W&B upload (keep flags aligned with other scripts)
    ap.add_argument("--wandb", action="store_true", help="Upload PNG to W&B using wandb.Image.")
    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default="capde")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    # Optional: eval+plot wrapper (similar to analyze_perm_noise.py)
    ap.add_argument("--eval_clm_path", type=str, default="", help="Path to eval_clm.py (to run if cache missing).")
    ap.add_argument("--skip_eval", action="store_true", help="Do not run eval_clm even if cache missing.")
    ap.add_argument("--pretrained_model_path", type=str, default="", help="HF model id/path (required for eval+plot).")
    ap.add_argument("--eval_names", type=str, nargs="+", default=[], help="e.g., arc,0,full")
    ap.add_argument("--option_id_set", type=str, default=None)
    ap.add_argument("--n_runs", type=int, default=1)
    ap.add_argument("--force", action="store_true")

    args, unknown = ap.parse_known_args()

    _set_font()

    # If eval args are provided, ensure cache exists (run eval_clm if needed) and compute heights from data.
    results_dir = None
    if str(args.pretrained_model_path).strip() and args.eval_names:
        eval_clm_path = str(args.eval_clm_path).strip() or os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_clm.py")
        if not os.path.exists(eval_clm_path):
            raise SystemExit(f"eval_clm.py not found: {eval_clm_path}")
        code_dir = os.path.dirname(os.path.abspath(eval_clm_path))

        # Only use first eval_name for this diagram
        eval_name = str(args.eval_names[0])
        results_dir = _compute_results_dir(code_dir, eval_name, str(args.pretrained_model_path), args.option_id_set)

        # Check cache exists; if not, run eval_clm
        try:
            _ = _compute_transition_counts(results_dir, int(args.n_runs), args.option_id_set)
            cache_ok = True
        except Exception:
            cache_ok = False

        if (not cache_ok) and (not bool(args.skip_eval)):
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

        # Now compute transition counts
        k, counts_init, counts_flip, counts_cyc = _compute_transition_counts(results_dir, int(args.n_runs), args.option_id_set)
    else:
        # Fallback: static example heights
        k = 4
        counts_init = {(0,): 1, (1,): 1}
        counts_flip = {(0, 0): 2, (0, 1): 1, (1, 0): 1, (1, 1): 2}
        counts_cyc = {t: 1 for t in [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]}

    fig, ax = plt.subplots(figsize=(float(args.width), float(args.height)))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Build plot blocks from counts (bottom->top order handled by draw_state_column)
    col1_data = [
        {"h": float(counts_init.get((0,), 0)), "text": "(0)"},
        {"h": float(counts_init.get((1,), 0)), "text": "(1)"},
    ]
    col2_data = [
        {"h": float(counts_flip.get((0, 0), 0)), "text": "(0, 0)"},
        {"h": float(counts_flip.get((0, 1), 0)), "text": "(0, 1)"},
        {"h": float(counts_flip.get((1, 0), 0)), "text": "(1, 0)"},
        {"h": float(counts_flip.get((1, 1), 0)), "text": "(1, 1)"},
    ]
    col3_data = [{"h": float(counts_cyc.get(t, 0)), "text": str(t)} for t in [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]]

    draw_state_column(ax, 0.5, "Initial\n(Default)", col1_data)
    draw_state_column(ax, 3.5, "Only\nFlip (Cost=2)", col2_data)
    draw_state_column(ax, 7.5, f"Cyclic\n(Cost={int(k)})", col3_data)

    # Legend / note
    ax.text(
        9.2,
        1.1,
        "0 = incorrect\n1 = correct",
        ha="left",
        va="bottom",
        fontsize=12,
    )

    plt.tight_layout()

    out_path = str(args.out)
    if results_dir and (not os.path.isabs(out_path)) and (os.path.dirname(out_path) == ""):
        out_path = os.path.join(results_dir, out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    print(f"Saved: {out_path}")

    if bool(getattr(args, "wandb", False)):
        wandb = _try_import_wandb()
        if wandb is None:
            print("[warn] wandb not available; skipping upload.")
            return
        run = wandb.init(
            project=getattr(args, "wandb_project", None) or None,
            entity=getattr(args, "wandb_entity", None) or None,
            name=getattr(args, "wandb_run_name", None) or None,
            job_type="state_transition_diagram",
            reinit=True,
        )
        try:
            wandb.log({"diagram/state_transition": wandb.Image(out_path)})
        except Exception as e:
            print(f"[warn] failed to upload diagram image: {e}")
        try:
            run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()