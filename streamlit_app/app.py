import os
from dataclasses import dataclass
import json
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import wandb


st.set_page_config(page_title="LLM-MCQ-Bias • W&B Curve Averager", layout="wide")


CURVE_DEFS = {
    "cyclic": {
        "x_key": "fraction",
        "label": "Cyclic",
        "color": "#F39C12",
        "linestyle": "-",
        "marker": "o",
    },
    "default_pride": {
        "x_key": "p",
        "label": "PriDe",
        "color": "#27AE60",
        "linestyle": "--",
        "marker": "s",
    },
    "ours": {
        "x_key": "p",
        "label": "Ours",
        "color": "#5DADE2",
        "linestyle": "-.",
        "marker": "^",
    },
    "ours_pride": {
        "x_key": "p",
        "label": "Ours (with PriDe)",
        "color": "#27AE60",
        "linestyle": "--",
        "marker": "D",
    },
}


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
def _fetch_run_record(run_path: str) -> Tuple[Optional[RunRecord], Optional[str]]:
    try:
        api = wandb.Api()
        run = api.run(run_path)
        summary = _safe_dict(run.summary)
        config = _safe_dict(run.config)
        points_by_task = summary.get("three_curves_points_v1", {}) or {}
        if not isinstance(points_by_task, dict):
            points_by_task = {}

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


def _curve_series_from_payload(payload: dict, curve_key: str, y_key: str) -> Dict[float, Dict[str, float]]:
    """
    Returns: x(p or fraction) -> {'cost': float, 'y': float}
    y_key: 'acc'|'recall_std'|'delta_acc'|'delta_recall_std'
    For delta_*: uses stored delta or computes (acc-default_acc) / (default_recall_std-recall_std) for aggregation.
    """
    if not isinstance(payload, dict):
        return {}
    curves = payload.get("curves", {}) or {}
    curve = curves.get(curve_key, {}) or {}
    x_key = CURVE_DEFS[curve_key]["x_key"]

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
    여러 run의 시리즈를 x 키로 맞춰서 평균. (delta 값들 + 형태로 합쳐서 평균)
    Returns: x -> {'cost_mean':..., 'y_mean':..., 'y_std':...}
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
            items.append((x, y, ystd))
    items = sorted(items, key=lambda t: t[0])
    if not items:
        return np.asarray([]), np.asarray([]), np.asarray([])
    x, y, ystd = zip(*items)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(ystd, dtype=np.float64)


def _filter_series_by_max_pct(
    series: Dict[float, Dict[str, float]], max_pct: Optional[float]
) -> Dict[float, Dict[str, float]]:
    """퍼센타일 상한(max_pct)까지만 잘라냄. Cyclic: 0~100, PriDe/Ours: 2~100"""
    if max_pct is None or max_pct >= 100:
        return series
    return {k: v for k, v in (series or {}).items() if k <= max_pct}


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
    max_pct: Optional[float] = None,
):
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=160)

    # plot per-model lines (each selected run is a "group" with a single payload)
    if plot_individual:
        for gname, payloads in group_payloads.items():
            if not payloads:
                continue
            for ck in curve_keys:
                series_list = [_curve_series_from_payload(p, ck, y_key) for p in payloads]
                series_list = [_filter_series_by_max_pct(s, max_pct) for s in series_list if s]
                series_list = [s for s in series_list if s]
                if not series_list:
                    continue
                agg = _aggregate_series(series_list)
                x, y, ystd = _series_to_xy(agg)
                if x.size == 0:
                    continue

                cd = CURVE_DEFS[ck]
                base_lab = str((curve_label_overrides or {}).get(ck) or cd["label"])
                label = f"{gname} • {base_lab}" if len(group_payloads) > 1 else base_lab
                ax.plot(
                    x, y,
                    color=cd["color"],
                    linestyle=cd["linestyle"],
                    marker=cd["marker"],
                    linewidth=2.0,
                    markersize=6,
                    alpha=0.90,
                    label=label,
                )
                # Only meaningful if this group contains multiple payloads.
                if show_group_std and len(series_list) > 1 and ystd.size == y.size:
                    ax.fill_between(
                        x,
                        y - ystd,
                        y + ystd,
                        color=cd["color"],
                        alpha=0.12,
                        linewidth=0,
                    )

    # overall mean (optional)
    if overall_mode in ("flatten_equal_run_weight",):
        for ck in curve_keys:
            all_series = []
            for payloads in group_payloads.values():
                for p in payloads:
                    s = _curve_series_from_payload(p, ck, y_key)
                    s = _filter_series_by_max_pct(s, max_pct) if s else {}
                    if s:
                        all_series.append(s)
            agg_all = _aggregate_series(all_series) if all_series else {}
            x, y, ystd = _series_to_xy(agg_all)

            if x.size == 0:
                continue
            cd = CURVE_DEFS[ck]
            base_lab = str((curve_label_overrides or {}).get(ck) or cd["label"])
            ax.plot(
                x, y,
                color=cd["color"],
                linestyle=":",
                linewidth=3.0,
                alpha=0.95,
                label=f"{overall_curve_label} • {base_lab}",
            )
            ax.fill_between(
                x,
                y - ystd,
                y + ystd,
                color=cd["color"],
                alpha=0.10,
                linewidth=0,
            )

    _y_labels = {"acc": "Accuracy (%)", "delta_acc": "Δ Accuracy (%)", "recall_std": "Recall std", "delta_recall_std": "Δ Recall std"}
    _y_titles = {"acc": "Accuracy", "delta_acc": "Δ Accuracy", "recall_std": "Recall std", "delta_recall_std": "Δ Recall std"}
    ax.set_xlabel("Computational Cost (× of default)", fontsize=11)
    ax.set_ylabel(_y_labels.get(y_key, y_key), fontsize=11)
    ax.set_title(f"{task} — {_y_titles.get(y_key, y_key)}", fontsize=12)
    if y_key in ("delta_acc", "delta_recall_std"):
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8, ncol=1)
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
    run_paths = _parse_run_paths(run_text)

    load_clicked = st.button("불러오기", use_container_width=True, disabled=(len(run_paths) == 0))

if "run_records" not in st.session_state:
    st.session_state.run_records = {}

load_errors = []
if load_clicked:
    st.session_state.run_records = {}
    for rp in run_paths:
        rec, err = _fetch_run_record(rp)
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
st.caption("Δ Accuracy(왼쪽)와 Δ Recall std(오른쪽)를 그립니다. X축은 Cost. 여러 run 선택 시 '각각 + 평균' 또는 '평균만'으로 delta 값들이 합쳐져 평균 곡선이 그려집니다.")

curve_keys = st.multiselect(
    "그릴 곡선",
    options=list(CURVE_DEFS.keys()),
    default=["cyclic", "default_pride", "ours"],
)

max_pct_options = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
max_pct_val = st.selectbox(
    "퍼센타일 상한 (이 퍼센트까지만 표시)",
    options=max_pct_options,
    index=len(max_pct_options) - 1,
    format_func=lambda x: f"{x}%까지" if x < 100 else "100% (전체)",
    help="Cyclic: 0/10/…/100, PriDe/Ours: 2/5/10/…/100. 선택한 숫자 이하만 보임.",
)

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
    lab_ours_pride = st.text_input("OURS+PRIDE 라벨", value=CURVE_DEFS["ours_pride"]["label"])

curve_label_overrides = {
    "cyclic": lab_cyclic,
    "default_pride": lab_pride,
    "ours": lab_ours,
    "ours_pride": lab_ours_pride,
}

plot_clicked = st.button("그래프 그리기", type="primary", use_container_width=True)

if plot_clicked:
    nonempty = {k: v for k, v in group_payloads.items() if isinstance(v, list) and len(v) > 0}
    if not nonempty:
        st.error("선택된 run이 없어요. 최소 1개 run(모델)을 선택하세요.")
        st.stop()
    if len(curve_keys) == 0:
        st.error("그릴 곡선을 최소 1개 선택하세요.")
        st.stop()

    max_pct_float = float(max_pct_val)

    c_left, c_right = st.columns(2)
    with c_left:
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
            max_pct=max_pct_float,
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
            max_pct=max_pct_float,
        )
        st.pyplot(fig_rstd, use_container_width=True)
