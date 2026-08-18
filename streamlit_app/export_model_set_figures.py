"""Export per-dataset Cyclic / PriDe / ACCEL cost-benefit figures for model subsets.

Data source is W&B: every `eval_clm.py --wandb` run stores its sweep in
`summary["three_curves_points_v1"][<task>]`, the same payload that is written to
`<task>_three_curves_points.json` on disk.

Two things are NOT in the W&B config and have to be recovered from
`wandb-metadata.json` (the raw argv of the run): `--result_tag` and the
empirical-PriDe flags.  Only the exact result_tag identifies the canonical
sweep -- each model has a dozen `__<result_tag>` variants, several of which
carry an `empirical_pride` curve with identical flat/online/empirical/latin
settings (see code/compare_cost_axis.py).

The ACCEL curve is `curves.empirical_pride.by_alpha["2"]["primary"]`, i.e. the
log family `empirical_pride_pct_flat_online_latin_a2_<beta>%`.  It is NOT
`curves.ours` / `curves.ours_pride` -- those are the earlier ARR threshold
cascade (th1/2, th1/sqrt2).

Usage
-----
    # 1. what is on W&B?  (prints run paths you can paste into streamlit_app/app.py)
    python export_model_set_figures.py --discover

    # 2. render PNGs
    python export_model_set_figures.py --out_dir figures_model_sets
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# canonical run identity
# --------------------------------------------------------------------------

# task -> result_tag given by the run owner.  Matching on the empirical flags
# alone does not identify the run; only the tag does.
CANONICAL_TAG: Dict[str, str] = {
    "csqa": "empirical_latin_flat_0502",
    "mmlu": "empirical_latin_flat",
    "arc": "empirical_latin_flat_0502",
    "raceall": "empirical_latin_flat_0502",
}

# alpha = calibration-prefix ratio (%) of the ACCEL curve.  by_alpha key "2".
ACCEL_ALPHA_KEY = "2"
ACCEL_BLOCK = "primary"

# The settings that make the curve `empirical_pride_pct_flat_online_latin_a2_*`.
# Each entry is (payload meta key, CLI flag, required value).  Older runs do not
# record every key in the payload -- `percentile_mode` in particular was added
# later -- so the CLI flag is the authority and the payload is checked only when
# it is present.  The flag defaults come from code/eval_clm_utils.py.
REQUIRED_EMPIRICAL = [
    ("sweep_mode", "--empirical_sweep_mode", "percentile"),
    ("percentile_mode", "--empirical_percentile_mode", "online"),
    ("threshold_schedule", "--empirical_stage_schedule", "flat"),
    ("transition_mode", "--empirical_transition_mode", "latin"),
    ("residual_model", "--empirical_residual_model", "empirical"),
]

# The canonical sweep is zero-shot; the `*_5` projects are the 5-shot variant and
# must never be mixed in.  Option-label set is fixed per task.
REQUIRED_OPTION_ID_SET = {"csqa": "ABCDE", "mmlu": "ABCD", "arc": "ABCD", "raceall": "ABCD"}


# --------------------------------------------------------------------------
# model sets
# --------------------------------------------------------------------------

# the 15 models of the submitted paper (docs/results.md main table)
PAPER_15 = [
    "Olmo-3-7B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "gemma-3-4b-it",
    "Llama-3.1-8B",
    "Llama-3.1-8B-Instruct",
    "Llama-3.2-3B",
    "Llama-3.2-3B-Instruct",
    "Phi-4-mini-instruct",
    "Phi-3-mini-4k-instruct",
    "Qwen3-4B-Instruct-2507",
    "Qwen2.5-7B",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Ministral-8B-Instruct-2410",
    "DeepSeek-R1-Distill-Llama-8B",
]

# the large models added after submission
NEW_5 = [
    "gemma-4-31B-it",
    "Qwen2.5-32B-Instruct",
    "Qwen2.5-72B-Instruct",
    "DeepSeek-R1-Distill-Llama-70B",
    "Llama-3.3-70B-Instruct",
]

# Some large models were launched from a local snapshot directory, so W&B
# recorded the snapshot hash as `model_name`.  Map those back to readable names.
MODEL_ALIASES: Dict[str, str] = {
    "145dc2508c480a64b47242f160d286cff94a2343": "gemma-4-31B-it",
    "495f39366efef23836d0cfae4fbe635880d2be31": "Qwen2.5-72B-Instruct",
    "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd": "Qwen2.5-32B-Instruct",
}


MODEL_SETS: Dict[str, Tuple[str, List[str]]] = {
    "paper15": ("15 models (paper)", PAPER_15),
    "all20": ("20 models (paper 15 + 5 large)", PAPER_15 + NEW_5),
    "large5": ("5 large models", NEW_5),
}


def _display_model(name: str) -> str:
    s = str(name or "").strip()
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    return MODEL_ALIASES.get(s, s)


def _norm_model(name: str) -> str:
    """Canonical key for model matching: lowercase, strip org prefix and punctuation."""
    s = _display_model(name).lower()
    return re.sub(r"[^a-z0-9]+", "", s)


ALL_MODEL_KEYS = {_norm_model(m) for m in (PAPER_15 + NEW_5)}


# --------------------------------------------------------------------------
# W&B fetching (with an on-disk cache; summaries are large)
# --------------------------------------------------------------------------

DEFAULT_CACHE = os.path.expanduser("~/.cache/accel_wandb_curves")


def _cache_path(cache_dir: str, entity: str, project: str, run_id: str) -> str:
    return os.path.join(cache_dir, f"{entity}__{project}__{run_id}.json")


def _arg_value(argv: List[str], flag: str) -> Optional[str]:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return str(argv[i + 1])
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _to_plain(obj):
    """W&B summaries hand back SummarySubDict, which is NOT a dict subclass."""
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if hasattr(obj, "keys") and hasattr(obj, "__getitem__"):
        try:
            return {str(k): _to_plain(obj[k]) for k in obj.keys()}
        except Exception:
            return {}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _run_metadata_args(run) -> List[str]:
    """argv of the run.  `run.metadata` lazily downloads wandb-metadata.json."""
    try:
        meta = run.metadata or {}
    except Exception:
        return []
    args = meta.get("args") or []
    return [str(a) for a in args]


def _empirical_meta(payload: dict) -> Dict[str, Any]:
    emp = ((payload.get("curves") or {}).get("empirical_pride") or {})
    return {
        "sweep_mode": str(emp.get("sweep_mode", "")).strip().lower(),
        "percentile_mode": str(emp.get("percentile_mode", "")).strip().lower(),
        "threshold_schedule": str(emp.get("threshold_schedule", "")).strip().lower(),
        "transition_mode": str(emp.get("transition_mode", "")).strip().lower(),
        "residual_model": str(emp.get("residual_model", "")).strip().lower(),
        "alphas": sorted((emp.get("by_alpha") or {}).keys()),
    }


def _check_accel_curve(payload: dict, argv: List[str], task: str) -> Tuple[bool, str]:
    """Verify this run really is the canonical empirical_pride_pct_flat_online_latin_a2 sweep."""
    meta = _empirical_meta(payload)
    problems = []

    for meta_key, flag, want in REQUIRED_EMPIRICAL:
        got_flag = _arg_value(argv, flag)
        if got_flag is not None and str(got_flag).strip().lower() != want:
            problems.append(f"{flag}={got_flag}!={want}")
        got_meta = meta.get(meta_key, "")
        if got_meta and got_meta != want:
            problems.append(f"{meta_key}={got_meta}!={want}")

    if str(_arg_value(argv, "--plot_empirical_prefix_fractions") or "2").strip() != "2":
        problems.append("prefix_fractions!=2")
    if "--empirical_pride" not in argv:
        problems.append("--empirical_pride missing")

    eval_names = str(_arg_value(argv, "--eval_names") or "")
    if eval_names and eval_names != f"{task},0,full":
        problems.append(f"eval_names={eval_names}!={task},0,full")
    want_ids = REQUIRED_OPTION_ID_SET.get(task)
    got_ids = _arg_value(argv, "--option_id_set")
    if want_ids and got_ids and got_ids != want_ids:
        problems.append(f"option_id_set={got_ids}!={want_ids}")

    if ACCEL_ALPHA_KEY not in meta["alphas"]:
        problems.append(f"alpha 2 missing (has {meta['alphas']})")
    else:
        emp = ((payload.get("curves") or {}).get("empirical_pride") or {})
        block = (emp.get("by_alpha") or {}).get(ACCEL_ALPHA_KEY) or {}
        if ACCEL_BLOCK not in block:
            problems.append("by_alpha['2']['primary'] missing")
    return (not problems), "; ".join(problems)


def collect_runs(
    entity: str,
    projects: Optional[List[str]],
    tasks: List[str],
    cache_dir: str,
    refresh: bool = False,
    verbose: bool = True,
) -> List[dict]:
    """Return one record per (run, task) that matches the canonical result_tag."""
    import wandb

    api = wandb.Api(timeout=60)
    os.makedirs(cache_dir, exist_ok=True)

    if not projects:
        projects = [p.name for p in api.projects(entity)]
        if verbose:
            print(f"[scan] {len(projects)} projects under {entity}", file=sys.stderr)

    records: List[dict] = []

    for project in projects:
        try:
            runs = list(api.runs(f"{entity}/{project}"))
        except Exception as e:
            if verbose:
                print(f"[skip] {entity}/{project}: {e}", file=sys.stderr)
            continue
        if verbose:
            print(f"[scan] {entity}/{project}: {len(runs)} runs", file=sys.stderr)

        def _fetch(run):
            cpath = _cache_path(cache_dir, entity, project, run.id)
            if os.path.exists(cpath) and not refresh:
                try:
                    with open(cpath, "r", encoding="utf-8") as f:
                        return run, json.load(f)
                except Exception:
                    pass
            cfg = run.config or {}
            model_raw = cfg.get("model_name") or cfg.get("pretrained_model_path")
            if _norm_model(model_raw) not in ALL_MODEL_KEYS or run.state != "finished":
                # cheap reject: no metadata download for runs we would drop anyway
                cached = {"skip": True, "result_tag": None}
            else:
                argv = _run_metadata_args(run)
                tag = _arg_value(argv, "--result_tag")
                if True:
                    try:
                        summary = _to_plain(run.summary)
                    except Exception:
                        summary = {}
                    points = summary.get("three_curves_points_v1") or {}
                    if not isinstance(points, dict):
                        points = {}
                    cached = {
                        "skip": False,
                        "run_id": run.id,
                        "project": project,
                        "entity": entity,
                        "name": run.name,
                        "state": run.state,
                        "created_at": str(run.created_at),
                        "result_tag": tag,
                        "argv": argv,
                        "model_name": cfg.get("model_name"),
                        "pretrained_model_path": cfg.get("pretrained_model_path"),
                        "points_by_task": {str(k): v for k, v in points.items()},
                    }
            with open(cpath, "w", encoding="utf-8") as f:
                json.dump(cached, f)
            return run, cached

        with ThreadPoolExecutor(max_workers=8) as pool:
            fetched = list(pool.map(_fetch, runs))

        for run, cached in fetched:
            if cached.get("skip"):
                continue
            for task in tasks:
                payload = (cached.get("points_by_task") or {}).get(task)
                if not isinstance(payload, dict):
                    continue
                argv = cached.get("argv") or []
                ok, why = _check_accel_curve(payload, argv, task)
                model = _display_model(cached.get("model_name") or cached.get("pretrained_model_path") or cached.get("name"))
                records.append({
                    "task": task,
                    "model": str(model),
                    "model_key": _norm_model(model),
                    "run_path": f"{entity}/{cached['project']}/{cached['run_id']}",
                    "state": cached.get("state"),
                    "created_at": cached.get("created_at"),
                    "result_tag": cached.get("result_tag"),
                    "canonical_tag": (cached.get("result_tag") == CANONICAL_TAG.get(task)),
                    "n_runs": _arg_value(argv, "--n_runs"),
                    "option_id_set": _arg_value(argv, "--option_id_set"),
                    "accel_ok": ok,
                    "accel_why": why,
                    "payload": payload,
                    "empirical_meta": _empirical_meta(payload),
                })
    return records


def pick_one_per_model(records: List[dict], require_n_runs: Optional[str]) -> Dict[str, dict]:
    """Latest finished run per model.  Runs whose ACCEL curve fails the gate are dropped."""
    best: Dict[str, dict] = {}
    for r in records:
        if not r["accel_ok"]:
            continue
        if require_n_runs is not None and str(r.get("n_runs")) != str(require_n_runs):
            continue
        key = r["model_key"]
        cur = best.get(key)
        if cur is None:
            best[key] = r
            continue
        rank = lambda x: (bool(x.get("canonical_tag")), x.get("state") == "finished",
                          str(x.get("created_at") or ""))
        if rank(r) > rank(cur):
            best[key] = r
    return best


# --------------------------------------------------------------------------
# curve extraction / aggregation
# --------------------------------------------------------------------------

def _default_baseline(payload: dict, key: str) -> float:
    v = payload.get(f"default_{key}", float("nan"))
    try:
        v = float(v)
    except Exception:
        v = float("nan")
    if np.isfinite(v):
        return v
    cyc = ((payload.get("curves") or {}).get("cyclic") or {})
    vals = cyc.get("acc" if key == "acc" else "recall_std") or []
    return float(vals[0]) if vals else float("nan")


def series(payload: dict, curve: str, metric: str) -> Dict[float, Tuple[float, float]]:
    """curve in {cyclic, pride, accel}; metric in {delta_acc, delta_rstd}.

    Returns x -> (cost, y).  x is the sweep parameter (cyclic fraction / PriDe
    alpha / ACCEL beta) so that models are averaged at the same operating point.
    Δ Rstd is returned as (rstd - baseline) in percentage points: negative = better.
    """
    curves = payload.get("curves") or {}
    if curve == "cyclic":
        block, x_key = curves.get("cyclic") or {}, "fraction"
    elif curve == "pride":
        block, x_key = curves.get("default_pride") or {}, "p"
    elif curve == "accel":
        emp = curves.get("empirical_pride") or {}
        block = ((emp.get("by_alpha") or {}).get(ACCEL_ALPHA_KEY) or {}).get(ACCEL_BLOCK) or {}
        x_key = "p"
    else:
        raise ValueError(curve)

    xs = block.get(x_key) or []
    costs = block.get("cost") or []
    if metric == "delta_acc":
        ys = block.get("delta_acc") or []
        if len(ys) != len(xs):
            base = _default_baseline(payload, "acc")
            ys = [float(a) - base for a in (block.get("acc") or [])]
    elif metric == "delta_rstd":
        # payload stores delta_recall_std = baseline - value (higher = better);
        # plot the signed change instead, in percentage points.
        ys = block.get("delta_recall_std") or []
        if len(ys) != len(xs):
            base = _default_baseline(payload, "recall_std")
            ys = [base - float(r) for r in (block.get("recall_std") or [])]
        ys = [-float(v) * 100.0 for v in ys]
    else:
        raise ValueError(metric)

    out: Dict[float, Tuple[float, float]] = {}
    for i in range(min(len(xs), len(costs), len(ys))):
        try:
            x, c, y = float(xs[i]), float(costs[i]), float(ys[i])
        except (TypeError, ValueError):
            continue
        if np.isfinite(c) and np.isfinite(y):
            out[x] = (c, y)
    return out


def aggregate(series_list: List[Dict[float, Tuple[float, float]]],
              max_pct: Optional[float], min_pct: Optional[float] = None):
    """Mean over models at each sweep point.  Returns (cost, y, y_std, n)."""
    xs = sorted(set().union(*[set(s) for s in series_list])) if series_list else []
    rows = []
    for x in xs:
        if max_pct is not None and x > max_pct:
            continue
        if min_pct is not None and x < min_pct:
            continue
        cs = [s[x][0] for s in series_list if x in s]
        ys = [s[x][1] for s in series_list if x in s]
        if not cs:
            continue
        rows.append((float(np.mean(cs)), float(np.mean(ys)), float(np.std(ys)), len(ys)))
    rows.sort(key=lambda t: t[0])
    if not rows:
        return (np.array([]),) * 4
    c, y, s, n = zip(*rows)
    return np.array(c), np.array(y), np.array(s), np.array(n)


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

STYLE = {
    "cyclic": {"label": "Cyclic", "color": "#E69F00", "marker": "o"},
    "pride": {"label": "PriDe", "color": "#009E73", "marker": "s"},
    "accel": {"label": "ACCEL (Ours)", "color": "#0072B2", "marker": "X"},
}
METRICS = [("delta_acc", "Δ Accuracy (%)", "Δ Accuracy"),
           ("delta_rstd", "Δ Rstd (%)", "Δ Rstd")]


def make_figure(task: str, set_label: str, payloads: List[dict], max_pct: Optional[float],
                show_band: bool, min_pct: Optional[float] = None,
                overall_label: str = "Overall mean"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(21, 6.2), dpi=160)
    for ax, (metric, ylabel, title) in zip(axes, METRICS):
        for curve, st in STYLE.items():
            sl = [s for s in (series(p, curve, metric) for p in payloads) if s]
            if not sl:
                continue
            # min_pct trims the sweep parameter of the two swept curves; cyclic
            # keeps fraction 0 so the (cost 1, delta 0) origin stays on the plot.
            x, y, ystd, _ = aggregate(sl, max_pct, None if curve == "cyclic" else min_pct)
            if x.size == 0:
                continue
            if show_band and ystd.size == y.size:
                ax.fill_between(x, y - ystd, y + ystd, color=st["color"], alpha=0.10, linewidth=0)
            ax.plot(x, y, color=st["color"], linestyle=":", marker=st["marker"],
                    linewidth=2.5, markersize=8, alpha=0.95,
                    label=f"{overall_label} • {st['label']}")
        ax.set_xlabel("Computational Cost (× of default)", fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_title(f"{task} — {title}", fontsize=20)
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(labelsize=14)
        if metric == "delta_rstd":
            # lower Rstd is better, so flip the axis: improvement points up,
            # same reading direction as the accuracy panel.
            ax.invert_yaxis()
        ax.legend(loc="lower right", fontsize=18)
    fig.suptitle(f"{task} — {set_label} (n={len(payloads)})", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="capde")
    ap.add_argument("--projects", nargs="*", default=None,
                    help="restrict the scan; default = every project of the entity")
    ap.add_argument("--tasks", nargs="*", default=["csqa", "mmlu", "arc", "raceall"])
    ap.add_argument("--sets", nargs="*", default=list(MODEL_SETS))
    ap.add_argument("--out_dir", default="figures_model_sets")
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE)
    ap.add_argument("--refresh", action="store_true", help="ignore the on-disk run cache")
    ap.add_argument("--require_n_runs", default=None, help="e.g. 3 to keep only 3-run sweeps")
    ap.add_argument("--max_pct", type=float, default=80.0,
                    help="drop sweep points above this percentile (100 = keep all)")
    ap.add_argument("--min_pct", type=float, default=2.0,
                    help="drop PriDe alpha / ACCEL beta below this (cyclic is not trimmed)")
    ap.add_argument("--band", action="store_true", help="shade ±1 std across models")
    ap.add_argument("--discover", action="store_true", help="only print what was found")
    args = ap.parse_args()

    records = collect_runs(args.entity, args.projects, args.tasks, args.cache_dir, args.refresh)
    if not records:
        print("no matching runs found", file=sys.stderr)
        return 1

    manifest: Dict[str, Any] = {}
    os.makedirs(args.out_dir, exist_ok=True)

    for task in args.tasks:
        task_recs = [r for r in records if r["task"] == task]
        chosen = pick_one_per_model(task_recs, args.require_n_runs)
        print(f"\n=== {task}  (tag={CANONICAL_TAG.get(task)})  {len(chosen)} models ===")
        for r in sorted(task_recs, key=lambda r: r["model"]):
            mark = "OK " if r["accel_ok"] else "BAD"
            picked = "*" if chosen.get(r["model_key"]) is r else " "
            flag = "  " if r["canonical_tag"] else "! "
            print(f" {picked}{mark}{flag}{r['model']:<32} {r['run_path']:<42} "
                  f"tag={r['result_tag']:<26} n_runs={r['n_runs']}"
                  f"{'' if r['accel_ok'] else '  << ' + r['accel_why']}")

        manifest[task] = {}
        for set_key in args.sets:
            set_label, models = MODEL_SETS[set_key]
            picked, missing = [], []
            for m in models:
                rec = chosen.get(_norm_model(m))
                (picked.append(rec) if rec else missing.append(m))
            manifest[task][set_key] = {
                "label": set_label,
                "runs": [{"model": r["model"], "run_path": r["run_path"],
                          "result_tag": r["result_tag"], "canonical_tag": r["canonical_tag"],
                          "n_runs": r["n_runs"]} for r in picked],
                "missing": missing,
            }
            print(f"  [{set_key}] {len(picked)}/{len(models)} models"
                  + (f"  MISSING: {', '.join(missing)}" if missing else ""))
            if args.discover or not picked:
                continue
            fig = make_figure(task, set_label, [r["payload"] for r in picked],
                              None if args.max_pct >= 100 else args.max_pct, args.band,
                              min_pct=(None if args.min_pct <= 0 else args.min_pct))
            out = os.path.join(args.out_dir, f"{task}_{set_key}.png")
            fig.savefig(out, bbox_inches="tight")
            print(f"  -> {out}")

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nmanifest: {os.path.join(args.out_dir, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
