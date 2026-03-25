import os
from dataclasses import dataclass
import json
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import wandb


st.set_page_config(page_title="LLM-MCQ-Bias • W&B Curve Averager", layout="wide")

# Color palette: Okabe-Ito (colorblind-friendly)
_PAL = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "gray": "#6B7280",
}


CURVE_DEFS = {
    "cyclic": {
        "x_key": "fraction",
        "label": "Cyclic",
        "color": _PAL["orange"],
        "linestyle": "-",
        "marker": "o",
    },
    "default_pride": {
        "x_key": "p",
        "label": "PriDe",
        "color": _PAL["bluish_green"],
        "linestyle": "--",
        "marker": "s",
    },
    "ours": {
        "x_key": "p",
        "label": "Ours (th1/sqrt2, no PriDe)",
        "color": _PAL["blue"],
        "linestyle": "-.",
        "marker": "^",
    },
    "ours_pride_primary": {
        "x_key": "p",
        "label": "Ours+PriDe (th1/sqrt2)",
        "color": _PAL["reddish_purple"],
        "linestyle": "-",
        "marker": "D",
    },
    "ours_pride_th1_2": {
        "x_key": "p",
        "label": "Ours+PriDe (th1/2 legacy)",
        "color": _PAL["vermillion"],
        "linestyle": "--",
        "marker": "*",
    },
    "ours_pride_online_sqrt": {
        "x_key": "p",
        "label": "Ours+PriDe (Online Sqrt)",
        "color": _PAL["sky"],
        "linestyle": "-.",
        "marker": "D",
    },
}

# Ours+PRIDE 다중 α 곡선용 색상 팔레트 (α·variant별 구분)
ALPHA_COLOR_PALETTE = [
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    "#AEC7E8", "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5",
    "#C49C94", "#F7B6D2", "#C7C7C7", "#DBDB8D", "#9EDAE5",
]


@dataclass(frozen=True)
class RunRecord:
    run_path: str
    run_id: str
    project: str
    entity: str
    display_name: str
    model_name: Optional[str]
    pretrained_model_path: Optional[str]
    tasks: List[str]
    points_by_task: Dict[str, dict]  # task -> payload
    sigma_by_task: Dict[str, dict]   # task -> sigma summary payload


def _parse_run_paths(text: str) -> List[str]:
    out: List[str] = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        # accept "entity/project/runid" or full URL-like inputs (strip prefix)
        s = s.replace("https://wandb.ai/", "").strip("/")
        out.append(s)
    # de-dup preserving order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _safe_dict(x) -> dict:
    try:
        return dict(x) if isinstance(x, dict) else dict(x or {})
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=60)
def _fetch_run_record(run_path: str, refresh_token: int = 0) -> Tuple[Optional[RunRecord], Optional[str]]:
    try:
        api = wandb.Api()
        run = api.run(run_path)
        summary = _safe_dict(run.summary)
        config = _safe_dict(run.config)
        points_by_task = summary.get("three_curves_points_v1", {}) or {}
        if not isinstance(points_by_task, dict):
            points_by_task = {}
        sigma_by_task_raw = summary.get("sigma_analysis_v1", {}) or {}
        if not isinstance(sigma_by_task_raw, dict):
            sigma_by_task_raw = {}
        sigma_by_task: Dict[str, dict] = {}
        for task_key, payload in sigma_by_task_raw.items():
            if not isinstance(payload, dict):
                continue
            task_name = str(task_key).split("_")[0].strip()
            if task_name and task_name not in sigma_by_task:
                sigma_by_task[task_name] = payload

        # Fallback: if summary is missing, try to recover from logged artifacts (newer eval_clm.py logs them)
        if len(points_by_task) == 0:
            try:
                recovered: Dict[str, dict] = {}
                arts = list(run.logged_artifacts() or [])[:30]
                for art in arts:
                    try:
                        if getattr(art, "type", None) != "three_curves_points":
                            continue
                        with tempfile.TemporaryDirectory() as td:
                            d = art.download(root=td)
                            # pick any *_three_curves_points.json in the artifact
                            for fn in os.listdir(d):
                                if not (isinstance(fn, str) and fn.endswith("_three_curves_points.json")):
                                    continue
                                p = os.path.join(d, fn)
                                with open(p, "r", encoding="utf-8") as f:
                                    payload = json.load(f)
                                task = str(payload.get("task", "")).strip()
                                if task:
                                    recovered[task] = payload
                    except Exception:
                        continue
                if recovered:
                    points_by_task = recovered
            except Exception:
                pass
        tasks = sorted([str(k) for k in points_by_task.keys()])

        # lightweight identity fields
        parts = run_path.split("/")
        entity = parts[0] if len(parts) >= 1 else ""
        project = parts[1] if len(parts) >= 2 else ""
        run_id = parts[2] if len(parts) >= 3 else getattr(run, "id", "")
        display_name = getattr(run, "name", None) or run_id or run_path

        rec = RunRecord(
            run_path=run_path,
            run_id=str(run_id),
            project=str(project),
            entity=str(entity),
            display_name=str(display_name),
            model_name=config.get("model_name"),
            pretrained_model_path=config.get("pretrained_model_path"),
            tasks=tasks,
            points_by_task=points_by_task,
            sigma_by_task=sigma_by_task,
        )
        if len(tasks) == 0:
            return rec, "W&B summary에 `three_curves_points_v1`가 없어요. (이 기능을 넣은 이후의 run이어야 평균을 낼 수 있어요.)"
        return rec, None
    except Exception as e:
        return None, f"run 로드 실패: `{run_path}` ({e})"


def _get_default_baseline(payload: dict, curves: dict, key: str) -> float:
    """key in ('acc', 'recall_std'). V1 payload: infer from cyclic[0]. V2: use payload.default_*."""
    default = payload.get(f"default_{key}", float("nan"))
    if np.isfinite(default):
        return float(default)
    cyc = curves.get("cyclic", {}) or {}
    vals = cyc.get("acc" if key == "acc" else "recall_std", []) or []
    if vals:
        try:
            v = float(vals[0])
            if np.isfinite(v):
                return v
        except Exception:
            pass
    return float("nan")


def _curve_series_from_payload(
    payload: dict,
    curve_key: str,
    y_key: str,
    ours_pride_alpha: Optional[float] = None,
    ours_pride_variant: Optional[str] = None,
) -> Dict[float, Dict[str, float]]:
    """
    Returns: x(p or fraction) -> {'cost': float, 'y': float}
    y_key: 'acc'|'recall_std'|'delta_acc'|'delta_recall_std'
    ours_pride_alpha: for ours_pride*, which PriDe α (10,20,...,100). None = 첫 번째.
    ours_pride_variant: 'th1/2'|'th1/sqrt2'|'online_sqrt_all' for ours_pride_* curves.
    """
    if not isinstance(payload, dict):
        return {}
    curves = payload.get("curves", {}) or {}
    curve = curves.get(curve_key, {}) or {}
    if curve_key in ("ours_pride_primary", "ours_pride_th1_2", "ours_pride_online_sqrt"):
        ours_pride_data = curves.get("ours_pride", {}) or {}
        by_alpha = ours_pride_data.get("by_alpha") or {}
        if by_alpha:
            alpha_key = (f"{float(ours_pride_alpha):g}" if ours_pride_alpha is not None else (list(by_alpha.keys())[0] if by_alpha else None))
            if alpha_key and alpha_key in by_alpha:
                alpha_curves = by_alpha[alpha_key]
                if isinstance(alpha_curves, dict):
                    if ours_pride_variant is not None:
                        variant = ours_pride_variant
                    elif curve_key == "ours_pride_online_sqrt":
                        variant = "online_sqrt_all"
                    elif curve_key == "ours_pride_primary":
                        variant = "th1/sqrt2"
                    else:
                        variant = "th1/2"
                    if variant in alpha_curves:
                        curve = alpha_curves[variant]
                    else:
                        curve = (
                            alpha_curves.get("th1/sqrt2")
                            or alpha_curves.get("online_sqrt_all")
                            or alpha_curves.get("th1/2")
                            or alpha_curves
                        )
                else:
                    curve = alpha_curves
    x_key = CURVE_DEFS.get(curve_key, {}).get("x_key") or "p"

    xs = curve.get(x_key, []) or []
    costs = curve.get("cost", []) or []
    if y_key == "delta_acc":
        ys = curve.get("delta_acc", []) or []
        default = _get_default_baseline(payload, curves, "acc")
        if not ys or (len(ys) != len(xs) and np.isfinite(default)):
            accs = curve.get("acc", []) or []
            if len(accs) == len(xs):
                ys = [float(a) - float(default) if np.isfinite(a) else float("nan") for a in accs]
        if not ys or len(ys) != len(xs):
            ys = curve.get("acc", []) or []
    elif y_key == "delta_recall_std":
        ys = curve.get("delta_recall_std", []) or []
        default = _get_default_baseline(payload, curves, "recall_std")
        if not ys or (len(ys) != len(xs) and np.isfinite(default)):
            rstds = curve.get("recall_std", []) or []
            if len(rstds) == len(xs):
                ys = [float(default) - float(r) if np.isfinite(r) else float("nan") for r in rstds]
        if not ys or len(ys) != len(xs):
            ys = curve.get("recall_std", []) or []
    else:
        ys = curve.get(y_key, []) or []

    out: Dict[float, Dict[str, float]] = {}
    n = min(len(xs), len(costs), len(ys))
    for i in range(n):
        try:
            xf = float(xs[i])
            cf = float(costs[i])
            yf = float(ys[i])
        except Exception:
            continue
        out[xf] = {"cost": cf, "y": yf}
    return out


def _nanmean(xs: List[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def _nanstd(xs: List[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    return float(np.nanstd(arr)) if arr.size else float("nan")


def _aggregate_series(series_list: List[Dict[float, Dict[str, float]]]) -> Dict[float, Dict[str, float]]:
    """
    여러 run의 시리즈를 x 키로 맞춰서 평균. y_std = n개 run 값들의 표준편차 (Run 간 흩어짐).
    """
    if not series_list:
        return {}
    xs_all = sorted(set().union(*[set(s.keys()) for s in series_list if isinstance(s, dict)]))
    out: Dict[float, Dict[str, float]] = {}
    for x in xs_all:
        costs = []
        ys = []
        for s in series_list:
            if not isinstance(s, dict) or x not in s:
                continue
            c = s[x].get("cost")
            y = s[x].get("y")
            try:
                cf = float(c)
                yf = float(y)
            except (TypeError, ValueError):
                continue
            costs.append(cf)
            ys.append(yf)
        if not costs or not ys:
            continue
        out[float(x)] = {
            "cost_mean": _nanmean(costs),
            "y_mean": _nanmean(ys),
            "y_std": _nanstd(ys),
            "n": float(len([v for v in ys if np.isfinite(v)])),
        }
    return out


def _series_to_xy(agg: Dict[float, Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    items = []
    for p, v in (agg or {}).items():
        y = float(v.get("y_mean", float("nan")))
        ystd = float(v.get("y_std", float("nan")))
        x = float(v.get("cost_mean", float("nan")))
        if np.isfinite(x) and np.isfinite(y):
            # n=1일 때 nanstd→nan이므로 0으로 치환 (fill_between nan 시 잘못 렌더링됨)
            items.append((x, y, ystd if np.isfinite(ystd) else 0.0))
    items = sorted(items, key=lambda t: t[0])
    if not items:
        return np.asarray([]), np.asarray([]), np.asarray([])
    x, y, ystd = zip(*items)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(ystd, dtype=np.float64)


def _filter_series_by_max_pct(
    series: Dict[float, Dict[str, float]],
    max_pct: Optional[float],
    curve_key: str,
    min_pct: Optional[float] = None,
) -> Dict[float, Dict[str, float]]:
    """Filter points to [min_pct, max_pct] range. Applied per curve_key."""
    result = dict(series or {})
    if min_pct is not None:
        min_limit = float(min_pct)
        if min_limit > 0:
            result = {k: v for k, v in result.items() if k >= min_limit}
    if max_pct is None:
        return result
    pct_limit = float(max_pct)
    if pct_limit >= 100:
        return result
    return {k: v for k, v in result.items() if k <= pct_limit}


def _plot_groups(
    group_payloads: Dict[str, List[dict]],
    task: str,
    curve_keys: List[str],
    y_key: str,
    show_group_std: bool,
    overall_mode: Optional[str],
    overall_curve_label: str = "Overall mean",
    curve_label_overrides: Optional[Dict[str, str]] = None,
    plot_individual: bool = True,
    max_pct_by_curve: Optional[Dict[str, float]] = None,
    n_runs_flattened: int = 0,
    show_overall_band: bool = False,
    ours_pride_alphas: Optional[List[float]] = None,
    ours_pride_base_label: str = "Ours+PRIDE",
    min_pct_by_curve: Optional[Dict[str, float]] = None,
):
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=160)

    # plot per-model lines (each selected run is a "group" with a single payload)
    max_by = max_pct_by_curve or {}
    min_by = min_pct_by_curve or {}

    _pride_keys = ("ours_pride_primary", "ours_pride_th1_2", "ours_pride_online_sqrt")
    if plot_individual:
        for gname, payloads in group_payloads.items():
            if not payloads:
                continue
            for ck in curve_keys:
                if ck in _pride_keys:
                    alphas = ours_pride_alphas or [2]
                    _show_alpha = len(alphas) >= 2
                    for i, alpha in enumerate(alphas):
                        series_list = [_curve_series_from_payload(p, ck, y_key, ours_pride_alpha=alpha) for p in payloads]
                        m = max_by.get(ck)
                        mn = min_by.get(ck)
                        series_list = [_filter_series_by_max_pct(s, m, ck, min_pct=mn) for s in series_list if s]
                        series_list = [s for s in series_list if s]
                        if not series_list:
                            continue
                        agg = _aggregate_series(series_list)
                        x, y, ystd = _series_to_xy(agg)
                        if x.size == 0:
                            continue
                        cd = CURVE_DEFS[ck]
                        base_offset = {"ours_pride_primary": 0, "ours_pride_th1_2": len(alphas), "ours_pride_online_sqrt": 2 * len(alphas)}
                        color_idx = base_offset.get(ck, 0) + i
                        line_color = cd["color"] if len(alphas) == 1 else ALPHA_COLOR_PALETTE[color_idx % len(ALPHA_COLOR_PALETTE)]
                        base_lab = str(ours_pride_base_label or "Ours").strip()
                        suffix_map = {
                            "ours_pride_primary": "th1/sqrt2",
                            "ours_pride_th1_2": "th1/2",
                            "ours_pride_online_sqrt": "Online Sqrt",
                        }
                        suffix = suffix_map.get(ck)
                        if suffix:
                            base_lab = f"{base_lab} {suffix}"
                        if _show_alpha:
                            base_lab = f"{base_lab} (α={alpha})"
                        label = f"{gname} • {base_lab}" if (len(group_payloads) > 1 and (gname or "").strip()) else base_lab
                        if show_group_std and len(series_list) > 1 and ystd.size == y.size:
                            ylo = np.where(np.isfinite(ystd), y - ystd, y)
                            yhi = np.where(np.isfinite(ystd), y + ystd, y)
                            ax.fill_between(x, ylo, yhi, color=line_color, alpha=0.10, linewidth=0)
                        ax.plot(x, y, color=line_color, linestyle=cd["linestyle"], marker=cd["marker"],
                                linewidth=2.0, markersize=8, alpha=0.90, label=label)
                else:
                    series_list = [_curve_series_from_payload(p, ck, y_key) for p in payloads]
                    m = max_by.get(ck)
                    mn = min_by.get(ck)
                    series_list = [_filter_series_by_max_pct(s, m, ck, min_pct=mn) for s in series_list if s]
                    series_list = [s for s in series_list if s]
                    if not series_list:
                        continue
                    agg = _aggregate_series(series_list)
                    x, y, ystd = _series_to_xy(agg)
                    if x.size == 0:
                        continue
                    cd = CURVE_DEFS[ck]
                    base_lab = str((curve_label_overrides or {}).get(ck) or cd["label"])
                    label = f"{gname} • {base_lab}" if (len(group_payloads) > 1 and (gname or "").strip()) else base_lab
                    if show_group_std and len(series_list) > 1 and ystd.size == y.size:
                        ylo = np.where(np.isfinite(ystd), y - ystd, y)
                        yhi = np.where(np.isfinite(ystd), y + ystd, y)
                        ax.fill_between(x, ylo, yhi, color=cd["color"], alpha=0.10, linewidth=0)
                    ax.plot(x, y, color=cd["color"], linestyle=cd["linestyle"], marker=cd["marker"],
                            linewidth=2.0, markersize=8, alpha=0.90, label=label)

    # overall mean (optional)
    if overall_mode in ("flatten_equal_run_weight",):
        for ck in curve_keys:
            if ck in _pride_keys:
                alphas = ours_pride_alphas or [2]
                _show_alpha = len(alphas) >= 2
                for i, alpha in enumerate(alphas):
                    all_series = []
                    for payloads in group_payloads.values():
                        for p in payloads:
                            s = _curve_series_from_payload(p, ck, y_key, ours_pride_alpha=alpha)
                            m = max_by.get(ck)
                            mn = min_by.get(ck)
                            s = _filter_series_by_max_pct(s, m, ck, min_pct=mn) if s else {}
                            if s:
                                all_series.append(s)
                    agg_all = _aggregate_series(all_series) if all_series else {}
                    x, y, ystd = _series_to_xy(agg_all)
                    if x.size == 0:
                        continue
                    cd = CURVE_DEFS[ck]
                    base_offset = {"ours_pride_primary": 0, "ours_pride_th1_2": len(alphas), "ours_pride_online_sqrt": 2 * len(alphas)}
                    color_idx = base_offset.get(ck, 0) + i
                    line_color = cd["color"] if len(alphas) == 1 else ALPHA_COLOR_PALETTE[color_idx % len(ALPHA_COLOR_PALETTE)]
                    base_lab = str(ours_pride_base_label or "Ours").strip()
                    suffix_map = {
                        "ours_pride_primary": "th1/sqrt2",
                        "ours_pride_th1_2": "th1/2",
                        "ours_pride_online_sqrt": "Online Sqrt",
                    }
                    suffix = suffix_map.get(ck)
                    if suffix:
                        base_lab = f"{base_lab} {suffix}"
                    if _show_alpha:
                        base_lab = f"{base_lab} (α={alpha})"
                    if show_overall_band and ystd.size == y.size:
                        ylo = np.where(np.isfinite(ystd), y - ystd, y)
                        yhi = np.where(np.isfinite(ystd), y + ystd, y)
                        ax.fill_between(x, ylo, yhi, color=line_color, alpha=0.10, linewidth=0)
                    ax.plot(x, y, color=line_color, linestyle=":", marker=cd["marker"],
                            linewidth=2.5, markersize=8, alpha=0.95,
                            label=f"{overall_curve_label} • {base_lab}" if (overall_curve_label or "").strip() else base_lab)
            else:
                all_series = []
                for payloads in group_payloads.values():
                    for p in payloads:
                        s = _curve_series_from_payload(p, ck, y_key)
                        m = max_by.get(ck)
                        mn = min_by.get(ck)
                        s = _filter_series_by_max_pct(s, m, ck, min_pct=mn) if s else {}
                        if s:
                            all_series.append(s)
                agg_all = _aggregate_series(all_series) if all_series else {}
                x, y, ystd = _series_to_xy(agg_all)
                if x.size == 0:
                    continue
                cd = CURVE_DEFS[ck]
                base_lab = str((curve_label_overrides or {}).get(ck) or cd["label"])
                if show_overall_band and ystd.size == y.size:
                    ylo = np.where(np.isfinite(ystd), y - ystd, y)
                    yhi = np.where(np.isfinite(ystd), y + ystd, y)
                    ax.fill_between(x, ylo, yhi, color=cd["color"], alpha=0.10, linewidth=0)
                ax.plot(x, y, color=cd["color"], linestyle=":", marker=cd["marker"],
                        linewidth=2.5, markersize=8, alpha=0.95,
                        label=f"{overall_curve_label} • {base_lab}" if (overall_curve_label or "").strip() else base_lab)

    _y_labels = {"acc": "Accuracy (%)", "delta_acc": "Δ Accuracy (%)", "recall_std": "Recall std", "delta_recall_std": "Δ Recall std"}
    _y_titles = {"acc": "Accuracy", "delta_acc": "Δ Accuracy", "recall_std": "Recall std", "delta_recall_std": "Δ Recall std"}
    subtitle = ""
    ax.set_xlabel("Computational Cost (× of default)", fontsize=20)
    ax.set_ylabel(_y_labels.get(y_key, y_key), fontsize=20)
    ax.set_title(f"{task} — {_y_titles.get(y_key, y_key)}{subtitle}", fontsize=20)
    if y_key in ("delta_acc", "delta_recall_std"):
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=24, ncol=1)
    fig.tight_layout()
    return fig


SIGMA_METRIC_LABELS = {
    "sigma_mean": "Sigma mean",
    "sigma_std": "Sigma std",
    "sigma_single": "Single-view residual sigma",
    "sigma_two_view": "Two-view residual sigma",
    "sigma_ratio": "Two-view / single-view",
    "sigma_ratio_target": "Target ratio",
    "corr_default_gap_sigma": "Corr(default gap, sigma)",
    "corr_flip_sigma": "Corr(flip, sigma)",
    "sigma_low_conf_mean": "Low-conf sigma",
    "sigma_high_conf_mean": "High-conf sigma",
    "flip_low_conf": "Flip rate (low-conf)",
    "flip_high_conf": "Flip rate (high-conf)",
    "flip_low_sigma": "Flip rate (low-sigma)",
    "flip_high_sigma": "Flip rate (high-sigma)",
    "records": "Samples",
}


def _sigma_variant_label(key: str) -> str:
    if key == "baseline":
        return "Baseline"
    if key.startswith("pride_alpha_"):
        alpha = key.replace("pride_alpha_", "")
        return f"PriDe (alpha={alpha}%)"
    return key


def _sigma_summary_rows(selected_run_paths: List[str], records: Dict[str, RunRecord], task: str) -> pd.DataFrame:
    grouped: Dict[str, List[dict]] = {}
    for rp in selected_run_paths:
        rec = records.get(rp)
        if rec is None:
            continue
        payload = rec.sigma_by_task.get(str(task))
        if not isinstance(payload, dict):
            continue
        for variant_key, summary in payload.items():
            if not isinstance(summary, dict):
                continue
            grouped.setdefault(str(variant_key), []).append(summary)

    rows: List[dict] = []
    for variant_key, summaries in grouped.items():
        row = {
            "variant_key": variant_key,
            "variant": _sigma_variant_label(variant_key),
            "n_runs": len(summaries),
        }
        metric_keys = sorted({k for s in summaries for k in s.keys()})
        for metric in metric_keys:
            vals: List[float] = []
            for s in summaries:
                try:
                    v = float(s.get(metric, float("nan")))
                except Exception:
                    v = float("nan")
                if np.isfinite(v):
                    vals.append(v)
            if not vals:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
            else:
                arr = np.asarray(vals, dtype=np.float64)
                row[f"{metric}_mean"] = float(np.nanmean(arr))
                row[f"{metric}_std"] = float(np.nanstd(arr))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    order = ["baseline"] + sorted([vk for vk in df["variant_key"].tolist() if vk != "baseline"])
    df["variant_key"] = pd.Categorical(df["variant_key"], categories=order, ordered=True)
    df = df.sort_values("variant_key").reset_index(drop=True)
    return df


def _format_sigma_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metric_order = [
        "sigma_mean",
        "sigma_std",
        "sigma_single",
        "sigma_two_view",
        "sigma_ratio",
        "sigma_ratio_target",
        "corr_default_gap_sigma",
        "corr_flip_sigma",
        "sigma_low_conf_mean",
        "sigma_high_conf_mean",
        "flip_low_conf",
        "flip_high_conf",
        "flip_low_sigma",
        "flip_high_sigma",
        "records",
    ]
    cols = ["variant", "n_runs"]
    renamed = {"variant": "Variant", "n_runs": "Runs"}
    for metric in metric_order:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col in df.columns:
            cols.append(mean_col)
            renamed[mean_col] = f"{SIGMA_METRIC_LABELS.get(metric, metric)} mean"
        if std_col in df.columns:
            cols.append(std_col)
            renamed[std_col] = f"{SIGMA_METRIC_LABELS.get(metric, metric)} std"
    return df[cols].rename(columns=renamed)


def _plot_sigma_ratio(df: pd.DataFrame):
    if df.empty or "sigma_ratio_mean" not in df.columns:
        return None
    plot_df = df[["variant", "sigma_ratio_mean", "sigma_ratio_std"]].copy()
    plot_df = plot_df[np.isfinite(plot_df["sigma_ratio_mean"])].reset_index(drop=True)
    if plot_df.empty:
        return None

    x = np.arange(len(plot_df))
    y = plot_df["sigma_ratio_mean"].to_numpy(dtype=np.float64)
    yerr = plot_df["sigma_ratio_std"].fillna(0.0).to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    ax.bar(x, y, yerr=yerr, capsize=5, color=_PAL["blue"], alpha=0.9)
    ax.axhline(1 / np.sqrt(2), color=_PAL["vermillion"], linestyle="--", linewidth=2, label="Ideal 1/sqrt(2)")
    ax.axhline(1.0, color=_PAL["gray"], linestyle=":", linewidth=1.5, label="No reduction")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["variant"].tolist(), rotation=20, ha="right")
    ax.set_ylabel("Sigma ratio")
    ax.set_title("Two-view Sigma Reduction")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    return fig


def _plot_sigma_confidence(df: pd.DataFrame):
    required = {"sigma_low_conf_mean_mean", "sigma_high_conf_mean_mean"}
    if df.empty or not required.issubset(set(df.columns)):
        return None
    plot_df = df[["variant", "sigma_low_conf_mean_mean", "sigma_high_conf_mean_mean"]].copy()
    plot_df = plot_df[
        np.isfinite(plot_df["sigma_low_conf_mean_mean"]) | np.isfinite(plot_df["sigma_high_conf_mean_mean"])
    ].reset_index(drop=True)
    if plot_df.empty:
        return None

    x = np.arange(len(plot_df))
    width = 0.36
    low_vals = plot_df["sigma_low_conf_mean_mean"].fillna(0.0).to_numpy(dtype=np.float64)
    high_vals = plot_df["sigma_high_conf_mean_mean"].fillna(0.0).to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
    ax.bar(x - width / 2, low_vals, width=width, color=_PAL["vermillion"], alpha=0.9, label="Low confidence")
    ax.bar(x + width / 2, high_vals, width=width, color=_PAL["bluish_green"], alpha=0.9, label="High confidence")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["variant"].tolist(), rotation=20, ha="right")
    ax.set_ylabel("Mean sigma")
    ax.set_title("Sigma by Confidence Bucket")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    return fig


st.title("LLM-MCQ-Bias • W&B run 평균 그래프")
st.caption("여러 W&B run의 `three_curves_points_v1`(수치 곡선)을 불러와서, 모델(=그룹)별 평균(예: 5-run 평균)과 여러 모델 평균을 그립니다.")

with st.sidebar:
    st.subheader("W&B 연결")
    api_key = st.text_input("WANDB_API_KEY (옵션)", type="password", help="로컬에 wandb login이 되어있으면 비워도 됩니다.")
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key

    st.subheader("Run 입력")
    run_text = st.text_area(
        "run path를 한 줄에 하나씩 입력",
        height=140,
        help='예: `capde/LLM-MCQ-Bias-code/udzbyjxz` (entity/project/run_id)',
    )
    ignore_cache = st.checkbox("W&B 캐시 무시하고 새로고침", value=False, help="방금 끝난 run의 summary가 아직 반영되지 않았을 때 사용하세요.")
    run_paths = _parse_run_paths(run_text)

    load_clicked = st.button("불러오기", use_container_width=True, disabled=(len(run_paths) == 0))

if "run_records" not in st.session_state:
    st.session_state.run_records = {}

load_errors = []
if load_clicked:
    st.session_state.run_records = {}
    refresh_token = int(time.time_ns()) if ignore_cache else 0
    for rp in run_paths:
        rec, err = _fetch_run_record(rp, refresh_token=refresh_token)
        if rec is not None:
            st.session_state.run_records[rp] = rec
        if err:
            load_errors.append(err)

records: Dict[str, RunRecord] = st.session_state.run_records

if load_errors:
    st.warning("일부 run에서 데이터를 못 가져왔어요.\n\n" + "\n".join([f"- {e}" for e in load_errors]))

if not records:
    st.info("왼쪽 사이드바에 run path를 입력하고 `불러오기`를 누르세요.")
    st.stop()

rows = []
all_tasks = set()
for rp, rec in records.items():
    all_tasks.update(rec.tasks)
    rows.append(
        {
            "run_path": rec.run_path,
            "run_id": rec.run_id,
            "name": rec.display_name,
            "model_name": rec.model_name,
            "pretrained_model_path": rec.pretrained_model_path,
            "tasks": ", ".join(rec.tasks),
            "sigma_tasks": ", ".join(sorted(rec.sigma_by_task.keys())),
        }
    )
df = pd.DataFrame(rows)
st.subheader("불러온 runs")
st.dataframe(df, use_container_width=True, hide_index=True)

task_list = sorted(all_tasks)
if len(task_list) == 0:
    st.error("선택 가능한 task가 없어요 (run summary에 `three_curves_points_v1`가 비어있음).")
    st.stop()

task = st.selectbox("Task 선택", options=task_list, index=0)

available_run_paths = list(records.keys())
st.subheader("모델 선택 (각 run = 한 모델 결과)")
st.caption("`--n_runs>1`로 돌린 run은 내부적으로 평균이 반영되어 있으니, 여기서는 모델(run)만 선택하면 됩니다.")

def _run_label(rec: RunRecord) -> str:
    mn = (rec.model_name or "").strip()
    nm = (rec.display_name or "").strip()
    rid = (rec.run_id or "").strip()
    if mn:
        return f"{mn}  ({rid})"
    return f"{nm}  ({rid})" if nm else rec.run_path

run_option_labels = {rp: _run_label(rec) for rp, rec in records.items()}
run_options = sorted(available_run_paths, key=lambda rp: run_option_labels.get(rp, rp))
selected_runs = st.multiselect(
    "그래프에 포함할 run(모델)을 선택",
    options=run_options,
    default=run_options[: min(2, len(run_options))],
    format_func=lambda rp: run_option_labels.get(rp, rp),
)

# Optional per-run label override (legend prefix)
use_custom_run_labels = st.checkbox("선택한 run의 라벨(범례 접두) 직접 수정", value=False)
run_label_overrides: Dict[str, str] = {}
if use_custom_run_labels and selected_runs:
    st.caption("범례에서 각 모델 라인을 구분하기 위한 이름입니다.")
    for rp in selected_runs:
        rec = records.get(rp)
        if rec is None:
            continue
        default_lab = (rec.model_name or rec.display_name or rec.run_id or rp)
        run_label_overrides[rp] = st.text_input(f"라벨: {run_option_labels.get(rp, rp)}", value=default_lab, key=f"runlab_{rp}")

# Build payload map: label -> [payload] (keep list type to reuse plotting code)
group_payloads: Dict[str, List[dict]] = {}
for rp in selected_runs:
    rec = records.get(rp)
    if rec is None:
        continue
    payload = rec.points_by_task.get(str(task))
    if not isinstance(payload, dict):
        continue
    prefix = run_label_overrides.get(rp) if use_custom_run_labels else (rec.model_name or rec.display_name or rec.run_id or rp)
    group_payloads[str(prefix)] = [payload]

st.subheader("그래프 옵션")
st.caption("Δ Accuracy(왼쪽)와 Δ Recall std(오른쪽)를 그립니다. X축은 Cost. 현재 primary는 `ours`와 `ours_pride_primary`의 `th1/sqrt2`입니다.")

curve_keys = st.multiselect(
    "그릴 곡선",
    options=list(CURVE_DEFS.keys()),
    default=["cyclic", "default_pride", "ours", "ours_pride_primary"],
    help="주로 `ours`와 `ours_pride_primary`를 보면 됩니다. `th1/2`는 legacy 비교용, `Online Sqrt`는 별도 adaptive heuristic입니다.",
)

cyclic_options = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
pride_ours_options = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

st.caption("곡선별 퍼센타일 상한 (선택한 % 이하만 표시)")
col_p1, col_p2, col_p3, col_p4, col_p5, col_p6 = st.columns(6)
with col_p1:
    max_pct_cyclic = st.selectbox(
        "Cyclic 상한",
        options=cyclic_options,
        index=len(cyclic_options) - 1,
        format_func=lambda x: f"{x}%" if x < 100 else "100% (전체)",
        key="max_cyclic",
    )
with col_p2:
    max_pct_pride = st.selectbox(
        "PriDe 상한",
        options=pride_ours_options,
        index=len(pride_ours_options) - 1,
        format_func=lambda x: f"{x}%" if x < 100 else "100% (전체)",
        key="max_pride",
    )
with col_p3:
    max_pct_ours = st.selectbox(
        "Ours 상한",
        options=pride_ours_options,
        index=len(pride_ours_options) - 1,
        format_func=lambda x: f"{x}%" if x < 100 else "100% (전체)",
        key="max_ours",
    )
with col_p4:
    max_pct_ours_pride_primary = st.selectbox(
        "Ours+PRIDE sqrt2 상한",
        options=pride_ours_options,
        index=len(pride_ours_options) - 1,
        format_func=lambda x: f"{x}%" if x < 100 else "100% (전체)",
        key="max_ours_pride_primary",
    )
with col_p5:
    max_pct_ours_pride_th1_2 = st.selectbox(
        "Ours+PRIDE th1/2 상한",
        options=pride_ours_options,
        index=len(pride_ours_options) - 1,
        format_func=lambda x: f"{x}%" if x < 100 else "100% (전체)",
        key="max_ours_pride_th12",
    )
with col_p6:
    max_pct_ours_pride_online_sqrt = st.selectbox(
        "Ours+PRIDE Online Sqrt 상한",
        options=pride_ours_options,
        index=len(pride_ours_options) - 1,
        format_func=lambda x: f"{x}%" if x < 100 else "100% (전체)",
        key="max_ours_pride_sqrt",
    )

st.caption("곡선별 퍼센타일 하한 (선택한 % 이상만 표시)")
col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
with col_m1:
    min_pct_cyclic = st.selectbox(
        "Cyclic 하한",
        options=cyclic_options,
        index=0,
        format_func=lambda x: f"{x}%" if x > 0 else "0% (전체)",
        key="min_cyclic",
    )
with col_m2:
    min_pct_pride = st.selectbox(
        "PriDe 하한",
        options=pride_ours_options,
        index=0,
        format_func=lambda x: f"{x}%",
        key="min_pride",
    )
with col_m3:
    min_pct_ours = st.selectbox(
        "Ours 하한",
        options=pride_ours_options,
        index=0,
        format_func=lambda x: f"{x}%",
        key="min_ours",
    )
with col_m4:
    min_pct_ours_pride_primary = st.selectbox(
        "Ours+PRIDE sqrt2 하한",
        options=pride_ours_options,
        index=0,
        format_func=lambda x: f"{x}%",
        key="min_ours_pride_primary",
    )
with col_m5:
    min_pct_ours_pride_th1_2 = st.selectbox(
        "Ours+PRIDE th1/2 하한",
        options=pride_ours_options,
        index=0,
        format_func=lambda x: f"{x}%",
        key="min_ours_pride_th12",
    )
with col_m6:
    min_pct_ours_pride_online_sqrt = st.selectbox(
        "Ours+PRIDE Online Sqrt 하한",
        options=pride_ours_options,
        index=0,
        format_func=lambda x: f"{x}%",
        key="min_ours_pride_sqrt",
    )

pride_alpha_options = [0.5, 1.0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ours_pride_alphas = st.multiselect(
    "Ours+PRIDE PriDe α (prefix)",
    options=pride_alpha_options,
    default=[2, 5, 10, 20],
    format_func=lambda x: f"α={x}%",
    key="ours_pride_alphas",
    help="Ours+PRIDE에서 PriDe prefix 비율. 여러 α 선택 시 각 α별 곡선이 함께 표시됩니다.",
)

max_pct_by_curve = {
    "cyclic": float(max_pct_cyclic),
    "default_pride": float(max_pct_pride),
    "ours": float(max_pct_ours),
    "ours_pride_primary": float(max_pct_ours_pride_primary),
    "ours_pride_th1_2": float(max_pct_ours_pride_th1_2),
    "ours_pride_online_sqrt": float(max_pct_ours_pride_online_sqrt),
}
min_pct_by_curve = {
    "cyclic": float(min_pct_cyclic),
    "default_pride": float(min_pct_pride),
    "ours": float(min_pct_ours),
    "ours_pride_primary": float(min_pct_ours_pride_primary),
    "ours_pride_th1_2": float(min_pct_ours_pride_th1_2),
    "ours_pride_online_sqrt": float(min_pct_ours_pride_online_sqrt),
}

overall_mode = st.radio(
    "표시 방식",
    options=[
        ("각각", "each"),
        ("각각 + 평균", "each_plus_mean"),
        ("평균만", "mean_only"),
    ],
    index=0,
)
display_mode = overall_mode[1]
overall_mode_key = "flatten_equal_run_weight" if display_mode in ("each_plus_mean", "mean_only") else None

st.subheader("라벨(범례) 설정")
st.caption("범례에 표시되는 곡선 이름/Overall 이름을 원하는 대로 바꿀 수 있어요.")

col_a, col_b, col_c, col_d, col_e = st.columns([1, 1, 1, 1, 1])
with col_a:
    overall_label = st.text_input("Overall 라벨", value="Overall mean")
with col_b:
    lab_cyclic = st.text_input("Cyclic 라벨", value=CURVE_DEFS["cyclic"]["label"])
with col_c:
    lab_pride = st.text_input("Default+PRIDE 라벨", value=CURVE_DEFS["default_pride"]["label"])
with col_d:
    lab_ours = st.text_input("OURS 라벨", value=CURVE_DEFS["ours"]["label"])
with col_e:
    lab_ours_pride = st.text_input("Ours (PriDe 붙은 곡선) 라벨", value="Ours")

curve_label_overrides = {
    "cyclic": lab_cyclic,
    "default_pride": lab_pride,
    "ours": lab_ours,
    "ours_pride_primary": lab_ours_pride,
    "ours_pride_th1_2": lab_ours_pride,
    "ours_pride_online_sqrt": lab_ours_pride,
}

show_overall_band = st.checkbox(
    "평균 곡선 밴드 표시",
    value=True,
    help="밴드 = 각 Run 내 시드편차의 평균 (0+0+0+0)/4=0. 끄면 선만 그림.",
)
plot_clicked = st.button("그래프 그리기", type="primary", use_container_width=True)

if plot_clicked:
    nonempty = {k: v for k, v in group_payloads.items() if isinstance(v, list) and len(v) > 0}
    if not nonempty:
        st.error("선택된 run이 없어요. 최소 1개 run(모델)을 선택하세요.")
        st.stop()
    if len(curve_keys) == 0:
        st.error("그릴 곡선을 최소 1개 선택하세요.")
        st.stop()

    c_left, c_right = st.columns(2)
    with c_left:
        n_runs_flattened = sum(len(p) for p in nonempty.values()) if overall_mode_key else 0
        fig_acc = _plot_groups(
            group_payloads=nonempty,
            task=str(task),
            curve_keys=curve_keys,
            y_key="delta_acc",
            show_group_std=False,
            overall_mode=overall_mode_key,
            overall_curve_label=str(overall_label or "Overall mean"),
            curve_label_overrides=curve_label_overrides,
            plot_individual=(display_mode in ("each", "each_plus_mean")),
            max_pct_by_curve=max_pct_by_curve,
            n_runs_flattened=n_runs_flattened,
            show_overall_band=show_overall_band,
            ours_pride_alphas=ours_pride_alphas or [2],
            ours_pride_base_label=str(lab_ours_pride or "Ours+PRIDE"),
            min_pct_by_curve=min_pct_by_curve,
        )
        st.pyplot(fig_acc, use_container_width=True)
    with c_right:
        fig_rstd = _plot_groups(
            group_payloads=nonempty,
            task=str(task),
            curve_keys=curve_keys,
            y_key="delta_recall_std",
            show_group_std=False,
            overall_mode=overall_mode_key,
            overall_curve_label=str(overall_label or "Overall mean"),
            curve_label_overrides=curve_label_overrides,
            plot_individual=(display_mode in ("each", "each_plus_mean")),
            max_pct_by_curve=max_pct_by_curve,
            n_runs_flattened=n_runs_flattened,
            show_overall_band=show_overall_band,
            ours_pride_alphas=ours_pride_alphas or [2],
            ours_pride_base_label=str(lab_ours_pride or "Ours+PRIDE"),
            min_pct_by_curve=min_pct_by_curve,
        )
        st.pyplot(fig_rstd, use_container_width=True)

    sigma_df = _sigma_summary_rows(selected_runs, records, str(task))
    st.subheader("Sigma 분석")
    if sigma_df.empty:
        missing_sigma_runs = []
        for rp in selected_runs:
            rec = records.get(rp)
            if rec is None or str(task) not in rec.sigma_by_task:
                missing_sigma_runs.append(run_option_labels.get(rp, rp))
        st.info("선택한 run들에는 이 task에 대한 `sigma_analysis_v1` 요약이 없어서 sigma 분석을 그릴 수 없어요.")
        if missing_sigma_runs:
            st.caption("다음 run들에는 이 task의 sigma summary가 없습니다: " + ", ".join(missing_sigma_runs))
        st.caption("보통 sigma 로깅을 넣기 전 run이거나, 해당 run이 아직 새 `eval_clm.py`로 다시 실행되지 않은 경우입니다.")
    else:
        st.markdown(
            "\n".join(
                [
                    "- 가설상 이상적인 패턴은 `sigma_ratio`가 `1/sqrt(2) ≈ 0.707`에 가깝고, `corr(default gap, sigma)`는 음수, `corr(flip, sigma)`는 양수인 경우입니다.",
                    "- 또 `sigma_low_conf_mean > sigma_high_conf_mean`, `flip_high_sigma > flip_low_sigma`가 보여야 low-confidence/unstable 샘플에서 variance가 더 크다는 서사와 맞습니다.",
                    "- 아래 표는 선택한 run들 사이 평균과 표준편차를 합친 값이라, run 하나의 우연한 흔들림보다 전체 경향을 보기에 좋습니다.",
                ]
            )
        )
        st.dataframe(_format_sigma_table(sigma_df), use_container_width=True, hide_index=True)

        s_left, s_right = st.columns(2)
        with s_left:
            fig_sigma_ratio = _plot_sigma_ratio(sigma_df)
            if fig_sigma_ratio is not None:
                st.pyplot(fig_sigma_ratio, use_container_width=True)
            else:
                st.caption("`sigma_ratio` 값이 없어 ratio plot은 생략했습니다.")
        with s_right:
            fig_sigma_conf = _plot_sigma_confidence(sigma_df)
            if fig_sigma_conf is not None:
                st.pyplot(fig_sigma_conf, use_container_width=True)
            else:
                st.caption("confidence bucket sigma 값이 없어 confidence plot은 생략했습니다.")
