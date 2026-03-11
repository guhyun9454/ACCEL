#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import platform
import subprocess
import sys
from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import zlib

# Ensure local imports work whether launched from repo root or code/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
try:
    from debias_utils import simple as debias_simple
except Exception:
    debias_simple = None


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


def _resolve_overlap(ys, min_dist):
    """
    y 좌표 리스트가 주어지면, 최소 간격(min_dist)을 유지하도록 위아래로 밀어냅니다.
    """
    ys = np.array(ys, dtype=float)
    for _ in range(100):
        changed = False
        for i in range(len(ys) - 1):
            if ys[i+1] - ys[i] < min_dist:
                diff = min_dist - (ys[i+1] - ys[i])
                ys[i] -= diff / 2.0
                ys[i+1] += diff / 2.0
                changed = True
        if not changed:
            break
    return ys.tolist()


def draw_state_column(ax, x_pos, title, blocks, width=1.5):
    """
    막대 그래프 컬럼을 그립니다.
    Y축을 60%부터 시작하도록 자르고(Zoom-in), 상단 부분을 확대해서 표시합니다.
    """
    # 0% 블록 제외
    blocks = [b for b in blocks if float(b.get("h", 0)) > 0]
    if not blocks:
        return

    total_val = sum([float(b["h"]) for b in blocks])
    
    # 60% ~ 100% 구간을 전체 높이(6.0)로 매핑
    y_min_pct = 60.0
    y_range_pct = 100.0 - y_min_pct
    col_height = 6.0
    y_base = 2.0

    # 컬럼 제목 (기본 굵기, 검정색)
    ax.text(
        x_pos + width / 2.0,
        8.4,
        title,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="normal",
        color="black"
    )

    labels_info = []

    # 각 상태별 패턴 매핑: 0, 1 만 있는 순수 상태는 패턴 제거("")
    hatch_dict = {
        "(0)": "",
        "(1)": "",
        "(0, 0)": "",
        "(0, 1)": "...",
        "(1, 0)": "...",
        "(1, 1)": "",
        "(0, 0, 0)": "",
        "(0, 0, 1)": "...",
        "(0, 1, 0)": "|||",
        "(0, 1, 1)": "\\\\\\",
        "(1, 0, 0)": "\\\\\\",
        "(1, 0, 1)": "|||",
        "(1, 1, 0)": "...",
        "(1, 1, 1)": ""
    }

    current_bottom_pct = 0.0

    # 1) 블록 그리기 및 라벨 정보 수집 (아래에서 위로)
    for block in reversed(blocks):
        h_raw = float(block["h"])
        pct = (h_raw / total_val) * 100.0
        block_top_pct = current_bottom_pct + pct

        text_str = str(block["text"])
        
        # 파스텔 톤 색상 적용
        if text_str.endswith("1)"):
            facecolor = "#bae1ff"  # 파스텔 블루 (Correct)
        elif text_str.endswith("0)"):
            facecolor = "#ffb3ba"  # 파스텔 핑크 (Incorrect)
        else:
            facecolor = "#e2e3e5"

        hatch_pattern = hatch_dict.get(text_str, "")

        # 60% 이상의 영역만 그리기 위한 범위 계산
        visible_bottom = max(current_bottom_pct, y_min_pct)
        visible_top = min(block_top_pct, 100.0)
        visible_h = visible_top - visible_bottom

        # 화면에 표시될 부분이 있는 경우에만 렌더링
        if visible_h > 0:
            h_norm = (visible_h / y_range_pct) * col_height
            rect_y = y_base + ((visible_bottom - y_min_pct) / y_range_pct) * col_height
            
            rect = patches.Rectangle(
                (x_pos, rect_y),
                width,
                h_norm,
                linewidth=1.2,
                edgecolor="#555555",
                facecolor=facecolor,
                hatch=hatch_pattern
            )
            ax.add_patch(rect)

            text_y = rect_y + (h_norm / 2.0)
            display_text = f"{text_str} ({pct:.1f}%)"
            labels_info.append((text_y, display_text))

        current_bottom_pct = block_top_pct

    # 2) 라벨들이 겹치지 않게 Y 좌표 조정 및 라벨/가이드선 그리기
    if labels_info:
        label_height = 0.35  # 글자 간 최소 Y 간격
        labels_info = sorted(labels_info, key=lambda t: t[0])
        target_ys = [info[0] for info in labels_info]
        actual_ys = _resolve_overlap(target_ys, label_height)

        for (target_y, text), actual_y in zip(labels_info, actual_ys):
            label_x = x_pos + width + 0.15
            
            # 블록 오른쪽에서 라벨로 이어지는 가이드선 (약간 회색)
            ax.plot(
                [x_pos + width, label_x - 0.05],
                [target_y, actual_y],
                color="#555555",
                linewidth=0.8,
                zorder=0,
            )
            
            # 라벨 텍스트 (기본 굵기, 검정색)
            ax.text(
                label_x, actual_y, text,
                ha="left", va="center", fontsize=10, color="black", fontweight="normal"
            )


def _rotations(k: int):
    return [tuple((i + s) % k for i in range(k)) for s in range(k)]


def _aggregate_probs_over_permutations(probs_seq, permuted_indices, k: int):
    agg = np.zeros(k, dtype=np.float64)
    for perm_idx, p in enumerate(permuted_indices):
        letter_probs = np.asarray(probs_seq[perm_idx], dtype=np.float64)
        for j in range(k):
            agg[p[j]] += letter_probs[j]
    if len(permuted_indices) > 0:
        agg /= float(len(permuted_indices))
    return agg


def _probe_shift_put_top2_into_top1_slot(base_probs, k: int):
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


def _stable_u32_seed(s: str, base_seed: int = 0) -> int:
    return (int(zlib.crc32(str(s).encode("utf-8"))) + int(base_seed)) & 0xFFFFFFFF


def _pride_correct_row(row: np.ndarray, prior: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    r = np.asarray(row, dtype=np.float64)
    pr = np.asarray(prior, dtype=np.float64)
    adj = r / (pr + eps)
    adj = adj / (adj.sum() + eps)
    return adj


def _estimate_pride_prior_random_prefix_mean(per_sample_probs, cyclic_indices, k: int, prefix_ratio: float, seed: int, eps: float = 1e-12):
    N = len(per_sample_probs)
    if N <= 0 or debias_simple is None:
        prior = np.ones((k,), dtype=np.float64) / float(k)
        return prior, {"N": int(N), "m": 0, "used": 0, "ratio": float(prefix_ratio), "seed": int(seed), "prefix_ids": []}
    ratio = float(max(0.0, min(1.0, float(prefix_ratio))))
    m = int(max(1, int(round(N * ratio))))
    rng = np.random.default_rng(int(seed))
    prefix_ids = rng.choice(np.arange(N, dtype=np.int64), size=m, replace=False)
    prefix_ids = [int(x) for x in prefix_ids.tolist()]
    priors = []
    used = 0
    for i in prefix_ids:
        ps = np.asarray(per_sample_probs[i], dtype=np.float64)
        observed = np.asarray([ps[j] for j in cyclic_indices], dtype=np.float64)
        try:
            _, _, prior_i = debias_simple(observed)
        except Exception:
            continue
        prior_i = np.asarray(prior_i, dtype=np.float64)
        prior_i = prior_i / (prior_i.sum() + eps)
        priors.append(prior_i)
        used += 1
    if len(priors) == 0:
        prior = np.ones((k,), dtype=np.float64) / float(k)
    else:
        prior = np.mean(np.asarray(priors, dtype=np.float64), axis=0)
        prior = np.asarray(prior, dtype=np.float64)
        prior = prior / (prior.sum() + eps)
    return prior, {"N": int(N), "m": int(m), "used": int(used), "ratio": float(ratio), "seed": int(seed), "prefix_ids": prefix_ids}


def _run_online_sqrt_policy_with_preds(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_pred_idx: list,
    cyclic_pred_idx: list,
    probe2_pred_idx: list,
    labels_idx: list,
    k: int,
    th1_percent: float,
    forced_cyclic_ids: Optional[set] = None,
):
    N = len(labels_idx)
    if N == 0:
        return float("nan"), float("nan"), []
    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)
    total_cost = 0.0
    corrects = 0
    preds = []
    running_gap_sum = 0.0
    running_cnt = 0
    past_gaps = []
    for i in range(N):
        gap_i = float(dc[i])
        th1_val = float(np.quantile(np.asarray(past_gaps, dtype=np.float64), float(th1_percent) / 100.0)) if len(past_gaps) > 0 else 0.0
        current_avg_gap = (running_gap_sum / running_cnt) if running_cnt > 0 else 0.0
        safe_avg = min(1.0, max(0.0, float(current_avg_gap)))
        th2_val = float(th1_val) * float(np.sqrt(1.0 - safe_avg))
        if gap_i >= th1_val:
            c_step = 1.0
            pred_i = int(base_pred_idx[i])
        else:
            if float(mc[i]) < float(th2_val):
                c_step = float(k)
                pred_i = int(cyclic_pred_idx[i])
            else:
                c_step = 2.0
                pred_i = int(probe2_pred_idx[i])
        if forced_cyclic_ids is not None and int(i) in forced_cyclic_ids:
            c_step = max(float(c_step), float(k))
            pred_i = int(cyclic_pred_idx[i])
        preds.append(pred_i)
        total_cost += float(c_step)
        corrects += 1 if (pred_i == int(labels_idx[i])) else 0
        running_gap_sum += gap_i
        running_cnt += 1
        past_gaps.append(gap_i)
    return total_cost / float(N), corrects / float(N), preds


def _compute_transition_counts(
    results_dir: str,
    n_runs: int,
    option_id_set: Optional[str],
    pride_prefix_ratio: float,
    pride_seed: int,
    th1_percent: float,
):
    run_suffix = "_run0" if int(n_runs) > 1 else ""
    pattern = os.path.join(results_dir, f"*{run_suffix}.jsonl")
    files = [p for p in sorted(glob.glob(pattern)) if p.endswith(".jsonl") and not p.endswith("_curve.jsonl") and not p.endswith("_pride_curve.jsonl")]
    if not files:
        files = [p for p in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))) if p.endswith(".jsonl") and not p.endswith("_curve.jsonl") and not p.endswith("_pride_curve.jsonl")]
    if not files:
        raise FileNotFoundError(f"No result jsonl files found under: {results_dir}")

    first = _read_results_jsonl(files[0])
    if not first:
        raise ValueError(f"Empty results: {files[0]}")
    k = int(len(first[0].get("options", []) or []))
    if k <= 0:
        raise ValueError("Cannot infer k from results.")
    option_ids = list(option_id_set) if option_id_set else (list("ABCDE"[:k]) if k in (4, 5) else [str(i) for i in range(k)])

    counts_init = {(0,): 0, (1,): 0}
    counts_flip = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    counts_third = {t: 0 for t in [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]}
    policy_cost_sum = 0.0
    policy_n = 0

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

        cyc_perms = _rotations(k)
        cyc_idxs = [perm_list.index(p) for p in cyc_perms if p in perm_list]
        cyc_perm_list = [perm_list[i] for i in cyc_idxs]

        per_sample_probs = []
        ideals = []
        for d in rows:
            probs_seq = d.get("probs", None)
            ideal = str(d.get("ideal"))
            if not isinstance(probs_seq, list) or len(probs_seq) != len(perm_list):
                continue
            if ideal not in option_ids:
                continue
            per_sample_probs.append(np.asarray(probs_seq, dtype=np.float64))
            ideals.append(ideal)

        if not per_sample_probs:
            continue

        subj = os.path.basename(fp)
        if subj.endswith(".jsonl"):
            subj = subj[:-5]
        seed = _stable_u32_seed(str(subj), int(pride_seed))
        prior, meta = _estimate_pride_prior_random_prefix_mean(
            per_sample_probs=per_sample_probs,
            cyclic_indices=cyc_idxs if cyc_idxs else list(range(min(k, len(perm_list)))),
            k=k,
            prefix_ratio=float(pride_prefix_ratio),
            seed=seed,
        )
        prefix_ids_set = set(int(x) for x in (meta.get("prefix_ids") or []))

        base_pred_idx_list = []
        cyclic_pred_idx_list = []
        probe2_pred_idx_list = []
        labels_idx = []
        default_conf = []
        mean_conf = []
        base_corrs = []
        probe2_corrs = []

        for i, (ps, ideal) in enumerate(zip(per_sample_probs, ideals)):
            ps = np.asarray(ps, dtype=np.float64)
            ps_corr = np.asarray([_pride_correct_row(ps[j], prior) for j in range(ps.shape[0])], dtype=np.float64)

            base_row_corr = np.asarray(ps_corr[identity_idx], dtype=np.float64)
            base_pred_idx = int(np.argmax(base_row_corr))
            base_pred_idx_list.append(base_pred_idx)
            base_corr = 1 if option_ids[base_pred_idx] == ideal else 0
            base_corrs.append(base_corr)

            if cyc_idxs:
                probs_c = [ps_corr[j].tolist() for j in cyc_idxs]
                perms_c = [perm_list[j] for j in cyc_idxs]
                agg_cyc = _aggregate_probs_over_permutations(probs_c, perms_c, k)
                cyc_pred_idx = int(np.argmax(agg_cyc))
            else:
                cyc_pred_idx = base_pred_idx
            cyclic_pred_idx_list.append(cyc_pred_idx)

            vals = np.sort(base_row_corr)[::-1]
            default_conf.append((float(vals[0]) if len(vals) > 0 else 0.0) - (float(vals[1]) if len(vals) > 1 else 0.0))
            shift = _probe_shift_put_top2_into_top1_slot(base_row_corr, k)
            probe_perm = tuple((ii + shift) % k for ii in range(k))
            probe_idx = perm_list.index(probe_perm) if probe_perm in perm_list else identity_idx
            agg_base = _aggregate_probs_over_permutations([base_row_corr.tolist()], [tuple(range(k))], k)
            probe_row_corr = np.asarray(ps_corr[probe_idx], dtype=np.float64)
            agg_probe = _aggregate_probs_over_permutations([probe_row_corr.tolist()], [probe_perm], k)
            mean_probs = (np.asarray(agg_base, dtype=np.float64) + np.asarray(agg_probe, dtype=np.float64)) / 2.0
            vals_mean = np.sort(mean_probs)[::-1]
            mean_conf.append(float(vals_mean[0] - vals_mean[1]) if len(vals_mean) > 1 else 0.0)
            probe2_pred_idx = int(np.argmax(mean_probs))
            probe2_pred_idx_list.append(probe2_pred_idx)
            probe2_corr = 1 if option_ids[probe2_pred_idx] == ideal else 0
            probe2_corrs.append(probe2_corr)

            labels_idx.append(option_ids.index(ideal))

        avg_cost, _, preds_ours = _run_online_sqrt_policy_with_preds(
            default_conf=np.asarray(default_conf, dtype=np.float64),
            mean_conf=np.asarray(mean_conf, dtype=np.float64),
            base_pred_idx=base_pred_idx_list,
            cyclic_pred_idx=cyclic_pred_idx_list,
            probe2_pred_idx=probe2_pred_idx_list,
            labels_idx=labels_idx,
            k=k,
            th1_percent=float(th1_percent),
            forced_cyclic_ids=prefix_ids_set if prefix_ids_set else None,
        )
        policy_cost_sum += float(avg_cost) * float(len(labels_idx))
        policy_n += int(len(labels_idx))

        for bc, pc, pred_idx, y in zip(base_corrs, probe2_corrs, preds_ours, labels_idx):
            oc = 1 if int(pred_idx) == int(y) else 0
            counts_init[(int(bc),)] += 1
            counts_flip[(int(bc), int(pc))] += 1
            counts_third[(int(bc), int(pc), int(oc))] += 1

    avg_policy_cost = (policy_cost_sum / float(policy_n)) if policy_n > 0 else float("nan")
    return k, counts_init, counts_flip, counts_third, avg_policy_cost


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--out", type=str, default="state_transition_diagram.png", help="Output PNG path")
    ap.add_argument("--dpi", type=int, default=300)
    # 가로축 공간을 더 확보하여 겹침을 방지합니다.
    ap.add_argument("--width", type=float, default=12.0) 
    ap.add_argument("--height", type=float, default=10.0)
    ap.add_argument("--wandb", action="store_true", help="Upload PNG to W&B using wandb.Image.")
    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default="capde")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    ap.add_argument("--eval_clm_path", type=str, default="", help="Path to eval_clm.py (to run if cache missing).")
    ap.add_argument("--skip_eval", action="store_true", help="Do not run eval_clm even if cache missing.")
    ap.add_argument("--pretrained_model_path", type=str, default="", help="HF model id/path (required for eval+plot).")
    ap.add_argument("--eval_names", type=str, nargs="+", default=[], help="e.g., arc,0,full")
    ap.add_argument("--option_id_set", type=str, default=None)
    ap.add_argument("--n_runs", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--th1_percent", type=float, default=100.0, help="th1 percentile for Online Sqrt All")
    ap.add_argument("--pride_prefix_ratio", type=float, default=0.02, help="PRIDE prefix ratio")
    ap.add_argument("--pride_seed", type=int, default=0, help="PRIDE seed offset")

    args, unknown = ap.parse_known_args()
    _set_font()

    results_dir = None
    if str(args.pretrained_model_path).strip() and args.eval_names:
        eval_clm_path = str(args.eval_clm_path).strip() or os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_clm.py")
        if not os.path.exists(eval_clm_path):
            raise SystemExit(f"eval_clm.py not found: {eval_clm_path}")
        code_dir = os.path.dirname(os.path.abspath(eval_clm_path))
        eval_name = str(args.eval_names[0])
        results_dir = _compute_results_dir(code_dir, eval_name, str(args.pretrained_model_path), args.option_id_set)

        try:
            _ = _compute_transition_counts(
                results_dir, int(args.n_runs), args.option_id_set,
                float(args.pride_prefix_ratio), int(args.pride_seed), float(args.th1_percent),
            )
            cache_ok = True
        except Exception:
            cache_ok = False

        if (not cache_ok) and (not bool(args.skip_eval)):
            cmd = [sys.executable, os.path.abspath(eval_clm_path)]
            cmd += ["--pretrained_model_path", str(args.pretrained_model_path)]
            cmd += ["--eval_names", eval_name]
            if args.option_id_set: cmd += ["--option_id_set", str(args.option_id_set)]
            if int(args.n_runs) != 1: cmd += ["--n_runs", str(int(args.n_runs))]
            if bool(args.force): cmd += ["--force"]
            if bool(args.wandb): cmd += ["--wandb"]
            if args.wandb_project: cmd += ["--wandb_project", str(args.wandb_project)]
            if args.wandb_entity: cmd += ["--wandb_entity", str(args.wandb_entity)]
            if args.wandb_run_name: cmd += ["--wandb_run_name", str(args.wandb_run_name)]
            cmd += list(unknown)
            subprocess.run(cmd, cwd=code_dir, check=True)

        k, counts_init, counts_flip, counts_cyc, avg_cost = _compute_transition_counts(
            results_dir, int(args.n_runs), args.option_id_set,
            float(args.pride_prefix_ratio), int(args.pride_seed), float(args.th1_percent),
        )
    else:
        k = 4
        counts_init = {(0,): 1, (1,): 1}
        counts_flip = {(0, 0): 2, (0, 1): 1, (1, 0): 1, (1, 1): 2}
        counts_cyc = {t: 1 for t in [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]}
        avg_cost = float("nan")

    fig, ax = plt.subplots(figsize=(float(args.width), float(args.height)))
    
    # 텍스트가 잘리지 않도록 전체 x 영역을 13.5까지 넓게 설정
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 10)
    ax.axis("off")

    col_width = 1.5
    
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
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
    ]]

    # 사이 간격: 0.5 -> 3.5 -> 6.5
    draw_state_column(ax, 0.5, "Initial\n(Default)", col1_data, width=col_width)
    draw_state_column(ax, 3.5, "Only Flip", col2_data, width=col_width)
    
    # 세 번째 컬럼 타이틀: Ours + Cost로만 깔끔하게 출력
    if np.isfinite(avg_cost):
        draw_state_column(ax, 6.5, f"Ours\n(Cost={avg_cost:.2f})", col3_data, width=col_width)
    else:
        draw_state_column(ax, 6.5, "Ours", col3_data, width=col_width)

    # =========================================================
    # 범례 박스 설정 (우측으로 멀리 이동 & 내부 가운데 정렬 완벽 적용)
    # =========================================================
    leg_x = 10.5  # 텍스트와 겹치지 않게 우측으로 확 뺐습니다.
    leg_y = 4.0
    leg_w = 2.4   # 박스 너비 여유롭게 확장
    leg_h = 1.6   # 박스 높이 여유롭게 확장
    
    # 겉 테두리 박스
    leg_box = patches.Rectangle(
        (leg_x, leg_y), leg_w, leg_h,
        linewidth=1.2, edgecolor="#555555", facecolor="white"
    )
    ax.add_patch(leg_box)
    
    # 박스 내 수직 가운데 정렬을 위해 영역을 상/하로 나누어 중앙점 계산
    item0_center_y = leg_y + (leg_h * 0.70)  # 위쪽 아이템 중앙
    item1_center_y = leg_y + (leg_h * 0.30)  # 아래쪽 아이템 중앙

    # 0 = Incorrect (Pastel Pink)
    ax.add_patch(patches.Rectangle((leg_x + 0.35, item0_center_y - 0.15), 0.3, 0.3, facecolor="#ffb3ba", edgecolor="#555555", linewidth=1.2))
    ax.text(leg_x + 0.8, item0_center_y, "0: Incorrect", ha="left", va="center", fontsize=11, fontweight="normal", color="black")
    
    # 1 = Correct (Pastel Blue)
    ax.add_patch(patches.Rectangle((leg_x + 0.35, item1_center_y - 0.15), 0.3, 0.3, facecolor="#bae1ff", edgecolor="#555555", linewidth=1.2))
    ax.text(leg_x + 0.8, item1_center_y, "1: Correct", ha="left", va="center", fontsize=11, fontweight="normal", color="black")

    plt.tight_layout()

    out_path = str(args.out)
    if results_dir and (not os.path.isabs(out_path)):
        out_path = os.path.join(results_dir, out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    if results_dir:
        print(f"results_dir: {results_dir}")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()