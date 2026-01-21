# eval_clm.py
# -*- coding: utf-8 -*-

import os
import sys
import gc
import json
import copy
import logging
import random
import math
import zlib
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_clm_utils import (
    parse_arguments,
    prepare_eval,
)

from debias_utils import simple as debias_simple

from utils import (
    _orange, _blue, _purple,
    eval_all_samples,
    get_accuracy,
    get_bootstrap_accuracy_std,
    save_results,
    patch_open,
)

# -------------------------
# [FIX] Safe NVML init
# -------------------------
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False

logger = logging.getLogger(__name__)


def logging_cuda_memory_usage():
    if not _NVML_OK:
        logger.info("******** Memory usage ********")
        logger.info("NVML unavailable; skipping GPU memory usage logging.")
        return
    logger.info("******** Memory usage ********")
    n_gpus = pynvml.nvmlDeviceGetCount()
    for i in range(n_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.info(
            "GPU {}: {:.2f} GB / {:.2f} GB".format(
                i,
                meminfo.used / 1024 ** 3,
                meminfo.total / 1024 ** 3
            )
        )


def _rotations(k: int):
    """cyclic rotations: ABCD, BCDA, CDAB, DABC"""
    return [tuple((i + s) % k for i in range(k)) for s in range(k)]


def _aggregate_probs_over_permutations(probs_seq, permuted_indices, k: int):
    """
    probs_seq: list/array of length = (#permutations used)
      each element: length k (letter-space probs)
    permuted_indices: list of permutations p where p[j] is content-index at letter position j.
    Returns: agg (k,) content-space aggregated probs (mean over permutations)
    """
    agg = np.zeros(k, dtype=np.float64)
    for perm_idx, p in enumerate(permuted_indices):
        letter_probs = np.asarray(probs_seq[perm_idx], dtype=np.float64)
        for j in range(k):
            agg[p[j]] += letter_probs[j]
    if len(permuted_indices) > 0:
        agg /= float(len(permuted_indices))
    return agg


def _probe_shift_cyclic_put_top2_into_top1_slot(base_probs: np.ndarray, k: int) -> Tuple[int, int, int]:
    """
    규칙:
      - base(letter-space)에서 top1=t1, top2=t2를 찾고,
      - cyclic rotations 중 "원래 top1 슬롯(=t1 위치)에 top2(t2)가 오도록" shift s 선택
    shift s = (t2 - t1) mod k
    """
    bp = np.asarray(base_probs, dtype=np.float64)
    sidx = np.argsort(bp)[::-1]
    t1 = int(sidx[0])
    t2 = int(sidx[1]) if len(sidx) > 1 else int(sidx[0])
    s = int((t2 - t1) % int(k))
    if s == 0:
        s = 1 if k > 1 else 0
    return s, t1, t2


def _read_results_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        lines = [e for e in lines if e.get('type') == 'result']
        lines = sorted(lines, key=lambda x: int(x['data']['idx']))
        return lines
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _stable_u32_seed(s: str, base_seed: int = 0) -> int:
    """Deterministic per-string seed (independent of Python hash randomization)."""
    return (int(zlib.crc32(s.encode("utf-8"))) + int(base_seed)) & 0xFFFFFFFF


def _apply_pride_global_prior_to_probs_seq(probs_seq: np.ndarray,
                                          prior: np.ndarray,
                                          eps: float = 1e-12) -> np.ndarray:
    """
    probs_seq: (num_perms, k) letter-space probs
    prior: (k,) global prior over option IDs (letter tokens)
    returns: corrected probs_seq with row-wise renormalization
    """
    p = np.asarray(probs_seq, dtype=np.float64)
    pr = np.asarray(prior, dtype=np.float64).reshape(1, -1)
    adj = p / (pr + eps)
    adj = adj / (adj.sum(axis=1, keepdims=True) + eps)
    return adj


def _estimate_pride_global_prior(per_sample_probs: List[np.ndarray],
                                 cyclic_indices: List[int],
                                 ratio_prefix_samples: float,
                                 prefix_selector: str,
                                 base_seed: int,
                                 subject_key: str,
                                 base_conf: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
    """
    Estimate global prior (PriDe) from subset of samples using cyclic permutations only.
    prefix_selector:
      - "random": deterministic random by subject_key+seed
      - "low_conf": base_conf 작은 샘플들로 prefix 구성
    """
    N = len(per_sample_probs)
    ratio = float(max(0.0, min(1.0, ratio_prefix_samples)))
    num_prefix = max(1, int(round(N * ratio))) if N > 0 else 0
    num_prefix = min(num_prefix, N)

    if num_prefix <= 0:
        prior = np.ones((len(cyclic_indices),), dtype=np.float64)
        prior = prior / prior.sum()
        return prior, {"num_prefix_samples": 0, "selector": prefix_selector}

    if prefix_selector == "low_conf" and base_conf is not None and len(base_conf) == N:
        prefix_ids = np.argsort(np.asarray(base_conf, dtype=np.float64))[:num_prefix].tolist()
        selector_used = "low_conf"
    else:
        seed = _stable_u32_seed(subject_key, base_seed=base_seed)
        rng = random.Random(seed)
        ids = list(range(N))
        rng.shuffle(ids)
        prefix_ids = ids[:num_prefix]
        selector_used = "random"

    priors = []
    for i in prefix_ids:
        observed = np.asarray([per_sample_probs[i][j] for j in cyclic_indices], dtype=np.float64)  # (k,k)
        _, _, prior_i = debias_simple(observed)
        priors.append(prior_i)

    prior_global = np.mean(np.asarray(priors, dtype=np.float64), axis=0)
    prior_global = prior_global / (prior_global.sum() + 1e-12)

    meta = {
        "ratio_prefix_samples": float(ratio),
        "num_prefix_samples": int(num_prefix),
        "selector": selector_used,
        "seed_key": str(subject_key),
        "base_seed": int(base_seed),
    }
    return prior_global, meta


# -------------------------
# Plot helpers (NO FULL-PERM lines)
# -------------------------
def _plot_pride_methods_no_full_png(curve_obj: dict,
                                    out_path: str,
                                    title: str) -> None:
    """
    PRIDE 보정된 애들끼리만: default/cyclic/switch_cyclic/ours_top2flip/ours_avggap
    (full, switch_full 등은 그리지 않음)
    """
    keys = ["cyclic", "switch_cyclic", "ours_top2flip", "ours_avggap"]

    plt.figure(figsize=(8.0, 5.6), dpi=180)

    # curves
    for k in keys:
        if k in curve_obj and "costs" in curve_obj[k] and "accuracies" in curve_obj[k]:
            c = curve_obj[k]["costs"]
            a = curve_obj[k]["accuracies"]
            if len(c) > 0:
                plt.plot(c, a, marker='o', label=f'{k}')

    # default point
    default_acc = float(curve_obj.get("default_accuracy", float("nan")))
    plt.scatter([1.0], [default_acc], marker='*', s=170, c='black', label='default')

    # always-cyclic point (sanity)
    always = curve_obj.get("always", {})
    if "cyclic" in always:
        plt.scatter([float(always["cyclic"]["cost"])],
                    [float(always["cyclic"]["acc"])],
                    marker='D', s=70, label='cyclic_ensemble')

    plt.xlabel("Computational Cost (× of default)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.30)
    plt.legend(ncol=2, fontsize=9)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_avggap_curve_png(curve_obj: dict,
                           out_path: str,
                           title: str,
                           label: str) -> None:
    """
    ours_avggap 단독 plot
    """
    plt.figure(figsize=(7.4, 5.0), dpi=180)
    c = curve_obj["ours_avggap"]["costs"]
    a = curve_obj["ours_avggap"]["accuracies"]
    plt.plot(c, a, marker='o', label=label)

    default_acc = float(curve_obj.get("default_accuracy", float("nan")))
    plt.scatter([1.0], [default_acc], marker='*', s=170, c='black', label='default')

    plt.xlabel("Computational Cost (× of default)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.30)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_avggap_compare_png(baseline_curve_obj: dict,
                             pride_curve_obj: dict,
                             out_path: str,
                             title: str) -> None:
    """
    baseline vs pride ours_avggap 비교 plot (full permu 없음)
    """
    plt.figure(figsize=(7.6, 5.2), dpi=180)

    bc = baseline_curve_obj["ours_avggap"]["costs"]
    ba = baseline_curve_obj["ours_avggap"]["accuracies"]
    pc = pride_curve_obj["ours_avggap"]["costs"]
    pa = pride_curve_obj["ours_avggap"]["accuracies"]

    plt.plot(bc, ba, marker='o', label='baseline ours_avggap')
    plt.plot(pc, pa, marker='o', linestyle='--', label='pride ours_avggap')

    b_def = float(baseline_curve_obj.get("default_accuracy", float("nan")))
    p_def = float(pride_curve_obj.get("default_accuracy", float("nan")))
    plt.scatter([1.0], [b_def], marker='*', s=170, c='black', label='baseline default')
    plt.scatter([1.0], [p_def], marker='*', s=170, c='gray', label='pride default')

    plt.xlabel("Computational Cost (× of default)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.30)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_heatmap_with_text(acc_grid: np.ndarray,
                            cost_grid: np.ndarray,
                            x_ticks: List[float],
                            y_ticks: List[float],
                            out_path: str,
                            title: str,
                            xlabel: str,
                            ylabel: str,
                            mode: str = "base"):
    """
    mode:
      - "base" or "pride": show "acc\\nc=cost"
      - "delta": show "Δacc\\nΔc"
    """
    plt.figure(figsize=(8.6, 7.2), dpi=200)
    im = plt.imshow(acc_grid, aspect='auto', origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    xt = list(range(len(x_ticks)))
    yt = list(range(len(y_ticks)))
    plt.xticks(xt, [f"{x:.0f}" for x in x_ticks], rotation=45, ha='right')
    plt.yticks(yt, [f"{y:.0f}" for y in y_ticks])

    ny, nx = acc_grid.shape
    for iy in range(ny):
        for ix in range(nx):
            a = acc_grid[iy, ix]
            c = cost_grid[iy, ix]
            if np.isnan(a) or np.isnan(c):
                txt = "nan"
            else:
                if mode == "delta":
                    txt = f"{a:+.3f}\n{c:+.2f}"
                else:
                    txt = f"{a:.3f}\nc={c:.2f}"
            plt.text(ix, iy, txt, ha='center', va='center', fontsize=6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _parse_percent_value_list(v) -> List[float]:
    """
    args.ours_low_conf_percent를:
      - float/int -> [float]
      - "10,20,30" -> [10.0,20.0,30.0]
      - ["10","20"] 같은 타입도 최대한 처리
    """
    if v is None:
        return [10.0]
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            try:
                out.append(float(x))
            except Exception:
                pass
        return out if len(out) > 0 else [10.0]
    if isinstance(v, str):
        s = v.strip()
        if "," in s:
            out = []
            for t in s.split(","):
                t = t.strip()
                if t == "":
                    continue
                try:
                    out.append(float(t))
                except Exception:
                    pass
            return out if len(out) > 0 else [10.0]
        try:
            return [float(s)]
        except Exception:
            return [10.0]
    return [10.0]


def _quantile(arr: np.ndarray, p01: float) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    p01 = float(max(0.0, min(1.0, p01)))
    return float(np.quantile(arr, p01))


def _policy_metrics_avggap_beta0(default_conf: np.ndarray,
                                 mean_conf: np.ndarray,
                                 base_correct: List[bool],
                                 cyclic_correct: List[bool],
                                 probe2_correct: np.ndarray,
                                 k: int,
                                 th1_percent: float,
                                 th2_percent: float) -> Tuple[float, float]:
    """
    ours_avggap 정책을 beta=0에서 (th1, th2 percentile)로 평가.
    return: (avg_cost, acc)
    """
    N = len(base_correct)
    if N == 0:
        return float("nan"), float("nan")
    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)
    th1 = _quantile(dc, th1_percent / 100.0)
    th2 = _quantile(mc, th2_percent / 100.0)

    total_cost = 0.0
    corrects = 0
    for i in range(N):
        if float(dc[i]) >= th1:
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
        else:
            if float(mc[i]) < th2:
                total_cost += float(k)
                corrects += 1 if cyclic_correct[i] else 0
            else:
                total_cost += 2.0
                corrects += 1 if bool(probe2_correct[i]) else 0

    return total_cost / float(N), corrects / float(N)


def _compute_curves_for_one_percentile(subject: str,
                                      tag: str,
                                      k: int,
                                      perm_list: List[Tuple[int, ...]],
                                      base_correct_list: List[bool],
                                      cyclic_correct_list: List[bool],
                                      full_correct_list: List[bool],
                                      default_conf: np.ndarray,
                                      mean_conf: np.ndarray,
                                      flip_trigger: np.ndarray,
                                      probe2_correct: np.ndarray,
                                      perc_value: float,
                                      betas: Optional[List[float]] = None) -> dict:
    """
    curve key:
      cyclic, full, switch_full, switch_cyclic, ours_top2flip, ours_avggap (+ default_accuracy)
    + always(default/cyclic/full ensemble)도 같이 저장해서 리포트 cost 혼동 방지
    """
    if betas is None:
        betas = [i / 10.0 for i in range(11)]

    N = len(base_correct_list)
    if N == 0:
        return {}

    perc01 = float(max(0.0, min(100.0, perc_value))) / 100.0

    C_cyc = float(k)
    C_full = float(len(perm_list))

    # always ensemble acc (sanity)
    default_acc = float(np.mean(np.asarray(base_correct_list, dtype=np.float64)))
    cyclic_acc_always = float(np.mean(np.asarray(cyclic_correct_list, dtype=np.float64)))
    full_acc_always = float(np.mean(np.asarray(full_correct_list, dtype=np.float64)))

    # 1) beta mix (cyclic/full)  ※ 첫 점(beta=0)은 base가 맞음 (그래서 리포트에선 always로 찍는다)
    curve_cyc = []
    curve_full = []
    for beta in betas:
        n = int(N * beta + 1e-9)
        if n > 0:
            acc_cyc_mix = (sum(cyclic_correct_list[:n]) + sum(base_correct_list[n:])) / float(N)
            acc_full_mix = (sum(full_correct_list[:n]) + sum(base_correct_list[n:])) / float(N)
        else:
            acc_cyc_mix = sum(base_correct_list) / float(N)
            acc_full_mix = sum(base_correct_list) / float(N)
        cost_cyc = beta * C_cyc + (1.0 - beta) * 1.0
        cost_full = beta * C_full + (1.0 - beta) * 1.0
        curve_cyc.append((cost_cyc, acc_cyc_mix))
        curve_full.append((cost_full, acc_full_mix))

    # 2) switch curves
    curve_switch_full = []
    curve_switch_cyc = []
    for beta in betas:
        n = int(N * beta + 1e-9)
        thresh = _quantile(default_conf[:n], perc01) if n > 0 else _quantile(default_conf, perc01)

        total_cost_sf = 0.0
        corrects_sf = 0
        total_cost_sc = 0.0
        corrects_sc = 0

        # online prefix: base만
        for i in range(0, n):
            total_cost_sf += 1.0
            total_cost_sc += 1.0
            if base_correct_list[i]:
                corrects_sf += 1
                corrects_sc += 1

        for i in range(n, N):
            amb = (float(default_conf[i]) < thresh)

            if amb:
                total_cost_sf += C_full
                corrects_sf += 1 if full_correct_list[i] else 0
            else:
                total_cost_sf += 1.0
                corrects_sf += 1 if base_correct_list[i] else 0

            if amb:
                total_cost_sc += C_cyc
                corrects_sc += 1 if cyclic_correct_list[i] else 0
            else:
                total_cost_sc += 1.0
                corrects_sc += 1 if base_correct_list[i] else 0

        curve_switch_full.append((total_cost_sf / float(N), corrects_sf / float(N)))
        curve_switch_cyc.append((total_cost_sc / float(N), corrects_sc / float(N)))

    # 3) ours_top2flip
    curve_top2flip = []
    for beta in betas:
        n = int(N * beta + 1e-9)
        th1 = _quantile(default_conf[:n], perc01) if n > 0 else _quantile(default_conf, perc01)

        total_cost = 0.0
        corrects = 0

        for i in range(0, n):
            total_cost += 1.0
            corrects += 1 if base_correct_list[i] else 0

        for i in range(n, N):
            if float(default_conf[i]) >= th1:
                total_cost += 1.0
                corrects += 1 if base_correct_list[i] else 0
            else:
                if bool(flip_trigger[i]):
                    total_cost += C_cyc
                    corrects += 1 if cyclic_correct_list[i] else 0
                else:
                    total_cost += 2.0
                    corrects += 1 if bool(probe2_correct[i]) else 0

        curve_top2flip.append((total_cost / float(N), corrects / float(N)))

    # 4) ours_avggap
    curve_avggap = []
    for beta in betas:
        n = int(N * beta + 1e-9)
        th1 = _quantile(default_conf[:n], perc01) if n > 0 else _quantile(default_conf, perc01)
        th2 = _quantile(mean_conf[:n], perc01) if n > 0 else _quantile(mean_conf, perc01)

        total_cost = 0.0
        corrects = 0

        for i in range(0, n):
            total_cost += 1.0
            corrects += 1 if base_correct_list[i] else 0

        for i in range(n, N):
            if float(default_conf[i]) >= th1:
                total_cost += 1.0
                corrects += 1 if base_correct_list[i] else 0
            else:
                if float(mean_conf[i]) < th2:
                    total_cost += C_cyc
                    corrects += 1 if cyclic_correct_list[i] else 0
                else:
                    total_cost += 2.0
                    corrects += 1 if bool(probe2_correct[i]) else 0

        curve_avggap.append((total_cost / float(N), corrects / float(N)))

    curve_obj = {
        "subject": subject,
        "tag": str(tag),
        "k": int(k),
        "percentile": float(perc_value),
        "betas": [float(b) for b in betas],
        "default_accuracy": float(default_acc),

        # always ensemble (리포트/검증용)
        "always": {
            "default": {"cost": 1.0, "acc": float(default_acc)},
            "cyclic": {"cost": float(C_cyc), "acc": float(cyclic_acc_always)},
            "full": {"cost": float(C_full), "acc": float(full_acc_always)},
        },

        # curves
        "cyclic": {"costs": [float(c) for c, _ in curve_cyc], "accuracies": [float(a) for _, a in curve_cyc]},
        "full": {"costs": [float(c) for c, _ in curve_full], "accuracies": [float(a) for _, a in curve_full]},
        "switch_full": {"costs": [float(c) for c, _ in curve_switch_full], "accuracies": [float(a) for _, a in curve_switch_full]},
        "switch_cyclic": {"costs": [float(c) for c, _ in curve_switch_cyc], "accuracies": [float(a) for _, a in curve_switch_cyc]},
        "ours_top2flip": {"costs": [float(c) for c, _ in curve_top2flip], "accuracies": [float(a) for _, a in curve_top2flip]},
        "ours_avggap": {"costs": [float(c) for c, _ in curve_avggap], "accuracies": [float(a) for _, a in curve_avggap]},
    }
    return curve_obj


def _log_beta0_report(curve_obj: dict, prefix: str):
    """
    beta=0 기준 derived policies(cost/acc) 출력
    NOTE:
      - cyclic/full은 mix-curve의 beta=0이 base로 떨어지는 게 정상이라,
        여기서는 always(ensemble) 기준으로 cost=k / cost=factorial(k)로 출력.
    """
    logger.info(_purple(f"==== {prefix} Derived policy report (beta=0, p={curve_obj.get('percentile')}) ===="))

    always = curve_obj.get("always", {})
    if "default" in always:
        logger.info(f"{prefix} default(ensemble) : cost={always['default']['cost']:.3f}, acc={always['default']['acc']:.4f}")
    if "cyclic" in always:
        logger.info(f"{prefix} cyclic(ensemble)  : cost={always['cyclic']['cost']:.3f}, acc={always['cyclic']['acc']:.4f}")
    if "full" in always:
        logger.info(f"{prefix} full(ensemble)    : cost={always['full']['cost']:.3f}, acc={always['full']['acc']:.4f}")

    # derived policies at beta=0 (curve 첫 점)
    def _p(key: str):
        c = curve_obj[key]["costs"][0]
        a = curve_obj[key]["accuracies"][0]
        logger.info(f"{prefix} {key:<12} : cost={c:.3f}, acc={a:.4f}")

    for k in ["switch_full", "switch_cyclic", "ours_top2flip", "ours_avggap"]:
        if k in curve_obj:
            _p(k)


def main():
    patch_open()

    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    hf_logging.set_verbosity_error()

    args = parse_arguments()
    if len(getattr(args, "eval_names", [])) == 0:
        return

    # -------- W&B init (optional) --------
    wandb_run = None
    wandb_ok = False
    if bool(getattr(args, "wandb", False)):
        try:
            import wandb
            wandb_ok = True
            project = getattr(args, "wandb_project", None) or "eval_clm"
            run_name = getattr(args, "wandb_run_name", None) or f"{getattr(args,'model_name','model')}-{args.eval_names[0]}"
            entity = getattr(args, "wandb_entity", None)
            cfg = {
                "pretrained_model_path": getattr(args, "pretrained_model_path", None),
                "model_name": getattr(args, "model_name", None),
                "eval_names": getattr(args, "eval_names", None),
                "option_id_set": getattr(args, "option_id_set", None),
                "ours_low_conf_percent": getattr(args, "ours_low_conf_percent", None),
                "disable_pride": getattr(args, "disable_pride", False),
                "pride_ratio_prefix_samples": getattr(args, "pride_ratio_prefix_samples", 0.05),
                "pride_prefix_selector": getattr(args, "pride_prefix_selector", "random"),
                "pride_seed": getattr(args, "pride_seed", 0),
            }
            wandb_run = wandb.init(project=project, entity=entity, name=run_name, config=cfg)
            logger.info(_blue(f"W&B init ok: project={project}, entity={entity}, name={run_name}"))
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")
            wandb_run = None
            wandb_ok = False

    # -------- Tokenizer / Model --------
    toker = AutoTokenizer.from_pretrained(
        args.pretrained_model_path,
        use_fast=False,
        add_bos_token=False,
        add_eos_token=False,
        cache_dir=getattr(args, "cache_dir", None),
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_path,
        device_map='auto',
        use_safetensors=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        cache_dir=getattr(args, "cache_dir", None),
    )

    logging_cuda_memory_usage()

    for eval_name in args.eval_names[::1]:
        (subjects, prepare_few_shot_samples,
         prepare_eval_samples, prepare_eval_fn) = prepare_eval(args, eval_name)

        for subject in subjects[::1]:
            cached_path = f'{args.save_path}/{subject}.jsonl'
            use_cached = (not bool(getattr(args, 'force', False))) and os.path.exists(cached_path)

            logger.info(_blue(f"Preparing: {subject}"))
            few_shot_samples = prepare_few_shot_samples(subject)
            eval_samples = prepare_eval_samples(subject)
            eval_fn = prepare_eval_fn(model, toker, few_shot_samples)

            if use_cached:
                logger.info(_blue(f"Using cached results: {cached_path}"))
                results = _read_results_file(cached_path) or []
            else:
                logger.info(_blue(f"Run started: {subject}"))
                max_samples = 100 if bool(getattr(args, 'test', False)) else None
                results = eval_all_samples(
                    eval_fn, eval_samples,
                    name=f'{args.task},{args.num_few_shot},{args.setting},{subject}',
                    threads=torch.cuda.device_count() if 'falcon' not in args.pretrained_model_path else 1,
                    max_num_samples=max_samples,
                )
                gc.collect()
                torch.cuda.empty_cache()

            # -------- metrics for existing settings --------
            metrics = None
            if len(results) > 0:
                if args.setting in ['perm', 'full', 'cyclic']:
                    if getattr(args, 'option_id_set', None):
                        option_ids = list(args.option_id_set)
                    else:
                        k_guess = len(results[0]['data']['options'])
                        option_ids = list('ABCDE' if k_guess == 5 else 'ABCD')
                    k = len(option_ids)

                    if args.setting in ['perm', 'full']:
                        from itertools import permutations
                        perm_list = list(sorted(permutations(range(k))))
                    else:
                        perm_list = _rotations(k)

                    total = 0
                    corrects = 0
                    for r in results:
                        if r.get('type') != 'result':
                            continue
                        data = r['data']
                        probs_seq = data.get('probs', None)
                        if not isinstance(probs_seq, list) or len(probs_seq) != len(perm_list):
                            continue
                        agg = _aggregate_probs_over_permutations(probs_seq, perm_list, k)
                        pred_letter = option_ids[int(np.argmax(agg))]
                        if pred_letter == data['ideal']:
                            corrects += 1
                        total += 1
                    acc = (corrects / total) if total > 0 else float('nan')
                    metrics = {'type': 'metric', 'data': {'accuracy': acc}}
                    logger.info(_purple(f"==== Ensemble report ({args.setting}) ===="))
                    logger.info(f"accuracy: {acc:.4f}")
                else:
                    metrics = {'type': 'metric', 'data': {}}
                    metrics['data']['accuracy'] = get_accuracy(results)
                    metrics['data']['boostrap_std'] = get_bootstrap_accuracy_std(results)

            logger.info(_orange(f"Run completed: {subject}"))

            if not use_cached:
                save_results(cached_path, results, metrics)
                logger.info(f"Results saved: {subject}")

            # =========================================================
            # Derived policies & beta curves (ONLY when args.setting == 'full')
            # =========================================================
            if args.setting == 'full' and len(results) > 0:
                try:
                    if getattr(args, 'option_id_set', None):
                        option_ids = list(args.option_id_set)
                    else:
                        k_guess = len(results[0]['data']['options'])
                        option_ids = list('ABCDE' if k_guess == 5 else 'ABCD')
                    k = len(option_ids)

                    from itertools import permutations
                    perm_list = list(sorted(permutations(range(k))))
                    identity_idx = perm_list.index(tuple(range(k)))

                    cyclic_indices = [
                        perm_list.index(tuple((i + s) % k for i in range(k)))
                        for s in range(k)
                    ]
                    cyc_perms = [tuple((i + s) % k for i in range(k)) for s in range(k)]

                    # ---------- collect per-sample raw probs ----------
                    per_sample_probs = []
                    ideals = []
                    sample_idxs = []
                    sample_prompts = []
                    sample_options_list = []

                    # ---------- derived correctness lists (baseline) ----------
                    base_correct_list = []
                    cyclic_correct_list = []
                    full_correct_list = []

                    base_probs_list = []  # identity row (letter-space)

                    cyclic_results = []
                    base_results = []

                    full_total = 0
                    full_corrects = 0
                    cyclic_total = 0
                    cyclic_corrects = 0

                    for r in results:
                        if r.get('type') != 'result':
                            continue
                        data = r['data']
                        probs_seq = data.get('probs')
                        if not isinstance(probs_seq, list) or len(probs_seq) != len(perm_list):
                            continue

                        probs_seq_np = np.asarray(probs_seq, dtype=np.float64)
                        per_sample_probs.append(probs_seq_np)

                        ideals.append(data['ideal'])
                        sample_idxs.append(data.get('idx'))
                        sample_prompts.append(data.get('prompt'))
                        sample_options_list.append(data.get('options'))

                        # cyclic (k rotations)
                        cyc_probs = [probs_seq_np[idx] for idx in cyclic_indices]
                        cyclic_results.append({
                            'type': 'result',
                            'data': {
                                'idx': data['idx'],
                                'prompt': data.get('prompt'),
                                'options': data['options'],
                                'probs': [cp.tolist() for cp in cyc_probs],
                                'ideal': data['ideal'],
                            },
                        })
                        agg_cyc = _aggregate_probs_over_permutations([cp.tolist() for cp in cyc_probs], cyc_perms, k)
                        pred_cyc = option_ids[int(np.argmax(agg_cyc))]
                        corr_cyc = (pred_cyc == data['ideal'])
                        cyclic_correct_list.append(corr_cyc)
                        cyclic_corrects += 1 if corr_cyc else 0
                        cyclic_total += 1

                        # base (identity only)
                        base_probs = np.asarray(probs_seq_np[identity_idx], dtype=np.float64)
                        base_probs_list.append(base_probs)
                        pred_base = option_ids[int(np.argmax(base_probs))]
                        corr_base = (pred_base == data['ideal'])
                        base_correct_list.append(corr_base)
                        base_results.append({
                            'type': 'result',
                            'data': {
                                'idx': data['idx'],
                                'prompt': data.get('prompt'),
                                'options': data['options'],
                                'probs': base_probs.tolist(),
                                'sampled': pred_base,
                                'ideal': data['ideal'],
                                'correct': corr_base,
                            },
                        })

                        # full (all perms)
                        agg_full = _aggregate_probs_over_permutations(probs_seq_np, perm_list, k)
                        pred_full = option_ids[int(np.argmax(agg_full))]
                        corr_full = (pred_full == data['ideal'])
                        full_correct_list.append(corr_full)
                        full_corrects += 1 if corr_full else 0
                        full_total += 1

                    # ---------- confidence stats & probe triggers (baseline) ----------
                    default_conf = []          # base gap (letter-space)
                    mean_gap_list = []         # gap(mean(base,probe)) (content-space)
                    flip_trigger_mask = []     # pred_base != pred_probe (content-space)
                    probe2_correct_list = []   # correctness of argmax(mean_probs)

                    for i, bp in enumerate(base_probs_list):
                        bp = np.asarray(bp, dtype=np.float64)
                        vals = np.sort(bp)[::-1]
                        top1 = float(vals[0]) if vals.shape[0] > 0 else 0.0
                        top2 = float(vals[1]) if vals.shape[0] > 1 else 0.0
                        default_conf.append(top1 - top2)

                        shift, _, _ = _probe_shift_cyclic_put_top2_into_top1_slot(bp, k)
                        probe_perm_idx = cyclic_indices[shift]

                        probs_base_raw = per_sample_probs[i][identity_idx]
                        agg_base = _aggregate_probs_over_permutations([probs_base_raw.tolist()], [tuple(range(k))], k)

                        probs_probe_raw = per_sample_probs[i][probe_perm_idx]
                        agg_probe = _aggregate_probs_over_permutations([probs_probe_raw.tolist()], [cyc_perms[shift]], k)

                        mean_probs = (agg_base + agg_probe) / 2.0
                        vals_mean = np.sort(mean_probs)[::-1]
                        mean_gap = float(vals_mean[0] - vals_mean[1]) if len(vals_mean) > 1 else 0.0
                        mean_gap_list.append(mean_gap)

                        pred_base_cs = option_ids[int(np.argmax(agg_base))]
                        pred_probe_cs = option_ids[int(np.argmax(agg_probe))]
                        flip_trigger_mask.append(pred_base_cs != pred_probe_cs)

                        pred2 = option_ids[int(np.argmax(mean_probs))]
                        probe2_correct_list.append(pred2 == ideals[i])

                    default_conf = np.asarray(default_conf, dtype=np.float64)
                    mean_conf = np.asarray(mean_gap_list, dtype=np.float64)
                    arr_flip_trigger = np.asarray(flip_trigger_mask, dtype=bool)
                    arr_probe2_correct = np.asarray(probe2_correct_list, dtype=bool)

                    # ---------- save cyclic/base derived results ----------
                    cyclic_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_cyclic'
                    if getattr(args, 'option_id_set', None):
                        cyclic_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(cyclic_save_path, exist_ok=True)

                    cyclic_acc = (cyclic_corrects / cyclic_total) if cyclic_total > 0 else float('nan')
                    save_results(f'{cyclic_save_path}/{subject}.jsonl', cyclic_results,
                                 metrics={'type': 'metric', 'data': {'accuracy': cyclic_acc}})

                    base_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}'
                    if getattr(args, 'option_id_set', None):
                        base_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(base_save_path, exist_ok=True)

                    base_acc = float(np.mean(np.asarray(base_correct_list, dtype=np.float64))) if len(base_correct_list) else float('nan')
                    save_results(f'{base_save_path}/{subject}.jsonl', base_results,
                                 metrics={'type': 'metric', 'data': {'accuracy': base_acc}})

                    full_acc = (full_corrects / full_total) if full_total > 0 else float('nan')

                    logger.info(_orange(f"Derived and saved cyclic results: {subject}"))
                    logger.info(_orange(f"Derived and saved base results: {subject}"))
                    logger.info(_purple(f"[{subject}] Accuracies — Full: {full_acc:.4f}, Cyclic: {cyclic_acc:.4f}, Default: {base_acc:.4f}"))

                    # ---------- compute & save beta curves (baseline) ----------
                    curve_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full'
                    if getattr(args, 'option_id_set', None):
                        curve_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(curve_save_path, exist_ok=True)

                    perc_list = _parse_percent_value_list(getattr(args, "ours_low_conf_percent", 10.0))
                    curve_objs_baseline = []
                    baseline_by_p = {}

                    for perc in perc_list:
                        cobj = _compute_curves_for_one_percentile(
                            subject=subject,
                            tag="baseline",
                            k=k,
                            perm_list=perm_list,
                            base_correct_list=base_correct_list,
                            cyclic_correct_list=cyclic_correct_list,
                            full_correct_list=full_correct_list,
                            default_conf=default_conf,
                            mean_conf=mean_conf,
                            flip_trigger=arr_flip_trigger,
                            probe2_correct=arr_probe2_correct,
                            perc_value=float(perc),
                        )
                        if cobj:
                            curve_objs_baseline.append(cobj)
                            baseline_by_p[float(perc)] = cobj
                            _log_beta0_report(cobj, prefix="BASELINE")

                            # baseline avggap 단독 plot (full permu 없음)
                            ptag = f"p{int(round(float(perc)))}"
                            base_avggap_png = os.path.join(curve_save_path, f"{subject}_{ptag}_avggap_baseline.png")
                            _plot_avggap_curve_png(
                                curve_obj=cobj,
                                out_path=base_avggap_png,
                                title=f"{args.task} {subject} — ours_avggap (Baseline, {ptag})",
                                label="baseline ours_avggap",
                            )
                            if wandb_ok and wandb_run is not None:
                                import wandb
                                wandb_run.log({f"plots/{subject}/{ptag}/avggap_baseline": wandb.Image(base_avggap_png)})

                    save_results(f'{curve_save_path}/{subject}_beta_curve.jsonl', curve_objs_baseline, metrics=None)

                    # =========================================================
                    # PRIDE
                    # =========================================================
                    curve_objs_pride = []
                    pride_by_p = {}

                    if not bool(getattr(args, "disable_pride", False)) and len(per_sample_probs) > 0:
                        subject_key = f"{args.task}|{args.num_few_shot}|{args.model_name}|{subject}|{getattr(args,'option_id_set','')}"
                        prior_global, prior_meta = _estimate_pride_global_prior(
                            per_sample_probs=per_sample_probs,
                            cyclic_indices=cyclic_indices,
                            ratio_prefix_samples=float(getattr(args, "pride_ratio_prefix_samples", 0.05)),
                            prefix_selector=str(getattr(args, "pride_prefix_selector", "random")),
                            base_seed=int(getattr(args, "pride_seed", 0)),
                            subject_key=subject_key,
                            base_conf=default_conf,
                        )

                        per_sample_probs_pride = [
                            _apply_pride_global_prior_to_probs_seq(ps, prior_global)
                            for ps in per_sample_probs
                        ]

                        base_correct_pride = []
                        cyclic_correct_pride = []
                        full_correct_pride = []

                        default_conf_pride = []
                        mean_conf_pride_list = []
                        flip_trigger_pride = []
                        probe2_correct_pride_list = []

                        base_results_pride = []
                        cyclic_results_pride = []
                        cyclic_corrects_pride = 0

                        for i in range(len(per_sample_probs_pride)):
                            probs_seq_np = per_sample_probs_pride[i]
                            s_idx = sample_idxs[i] if i < len(sample_idxs) else i
                            s_prompt = sample_prompts[i] if i < len(sample_prompts) else None
                            s_options = sample_options_list[i] if i < len(sample_options_list) else None

                            # cyclic
                            cyc_probs = [probs_seq_np[idx] for idx in cyclic_indices]
                            cyclic_results_pride.append({
                                'type': 'result',
                                'data': {
                                    'idx': s_idx,
                                    'prompt': s_prompt,
                                    'options': s_options,
                                    'probs': [cp.tolist() for cp in cyc_probs],
                                    'ideal': ideals[i],
                                },
                            })
                            agg_cyc = _aggregate_probs_over_permutations([cp.tolist() for cp in cyc_probs], cyc_perms, k)
                            pred_cyc = option_ids[int(np.argmax(agg_cyc))]
                            corr_cyc = (pred_cyc == ideals[i])
                            cyclic_correct_pride.append(corr_cyc)
                            cyclic_corrects_pride += 1 if corr_cyc else 0

                            # base
                            base_probs = np.asarray(probs_seq_np[identity_idx], dtype=np.float64)
                            pred_base = option_ids[int(np.argmax(base_probs))]
                            corr_base = (pred_base == ideals[i])
                            base_correct_pride.append(corr_base)
                            base_results_pride.append({
                                'type': 'result',
                                'data': {
                                    'idx': s_idx,
                                    'prompt': s_prompt,
                                    'options': s_options,
                                    'probs': base_probs.tolist(),
                                    'sampled': pred_base,
                                    'ideal': ideals[i],
                                    'correct': corr_base,
                                },
                            })

                            # full
                            agg_full = _aggregate_probs_over_permutations(probs_seq_np, perm_list, k)
                            pred_full = option_ids[int(np.argmax(agg_full))]
                            full_correct_pride.append(pred_full == ideals[i])

                            # confs
                            vals = np.sort(base_probs)[::-1]
                            top1 = float(vals[0]) if vals.shape[0] > 0 else 0.0
                            top2 = float(vals[1]) if vals.shape[0] > 1 else 0.0
                            default_conf_pride.append(top1 - top2)

                            shift, _, _ = _probe_shift_cyclic_put_top2_into_top1_slot(base_probs, k)
                            probe_perm_idx = cyclic_indices[shift]

                            probs_base_raw = probs_seq_np[identity_idx]
                            agg_base = _aggregate_probs_over_permutations([probs_base_raw.tolist()], [tuple(range(k))], k)

                            probs_probe_raw = probs_seq_np[probe_perm_idx]
                            agg_probe = _aggregate_probs_over_permutations([probs_probe_raw.tolist()], [cyc_perms[shift]], k)

                            mean_probs = (agg_base + agg_probe) / 2.0
                            vals_mean = np.sort(mean_probs)[::-1]
                            mean_gap = float(vals_mean[0] - vals_mean[1]) if len(vals_mean) > 1 else 0.0
                            mean_conf_pride_list.append(mean_gap)

                            pred_base2 = option_ids[int(np.argmax(agg_base))]
                            pred_probe2 = option_ids[int(np.argmax(agg_probe))]
                            flip_trigger_pride.append(pred_base2 != pred_probe2)

                            pred2 = option_ids[int(np.argmax(mean_probs))]
                            probe2_correct_pride_list.append(pred2 == ideals[i])

                        default_conf_pride = np.asarray(default_conf_pride, dtype=np.float64)
                        mean_conf_pride = np.asarray(mean_conf_pride_list, dtype=np.float64)
                        arr_flip_pride = np.asarray(flip_trigger_pride, dtype=bool)
                        arr_probe2_correct_pride = np.asarray(probe2_correct_pride_list, dtype=bool)

                        # save pride base/cyclic
                        cyclic_save_path_pride = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_cyclic_pride'
                        if getattr(args, 'option_id_set', None):
                            cyclic_save_path_pride += f'_id-{args.option_id_set}'
                        os.makedirs(cyclic_save_path_pride, exist_ok=True)
                        cyclic_acc_pride = float(cyclic_corrects_pride) / float(len(cyclic_correct_pride)) if len(cyclic_correct_pride) else float('nan')
                        save_results(f'{cyclic_save_path_pride}/{subject}.jsonl', cyclic_results_pride,
                                     metrics={'type': 'metric', 'data': {'accuracy': cyclic_acc_pride}})

                        base_save_path_pride = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_pride'
                        if getattr(args, 'option_id_set', None):
                            base_save_path_pride += f'_id-{args.option_id_set}'
                        os.makedirs(base_save_path_pride, exist_ok=True)
                        base_acc_pride = float(np.mean(np.asarray(base_correct_pride, dtype=np.float64))) if len(base_correct_pride) else float('nan')
                        save_results(f'{base_save_path_pride}/{subject}.jsonl', base_results_pride,
                                     metrics={'type': 'metric', 'data': {'accuracy': base_acc_pride}})

                        # curves pride
                        curve_save_path_pride = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full_pride'
                        if getattr(args, 'option_id_set', None):
                            curve_save_path_pride += f'_id-{args.option_id_set}'
                        os.makedirs(curve_save_path_pride, exist_ok=True)

                        for perc in perc_list:
                            cobjp = _compute_curves_for_one_percentile(
                                subject=subject,
                                tag="pride",
                                k=k,
                                perm_list=perm_list,
                                base_correct_list=base_correct_pride,
                                cyclic_correct_list=cyclic_correct_pride,
                                full_correct_list=full_correct_pride,
                                default_conf=default_conf_pride,
                                mean_conf=mean_conf_pride,
                                flip_trigger=arr_flip_pride,
                                probe2_correct=arr_probe2_correct_pride,
                                perc_value=float(perc),
                            )
                            if cobjp:
                                cobjp["pride"] = {
                                    "prior": [float(x) for x in prior_global.tolist()],
                                    "prior_map": {str(k_): float(v) for k_, v in zip(option_ids, prior_global.tolist())},
                                    "meta": prior_meta,
                                }
                                curve_objs_pride.append(cobjp)
                                pride_by_p[float(perc)] = cobjp
                                _log_beta0_report(cobjp, prefix="PRIDE")

                                # 1) PRIDE 보정된 애들끼리만 plot (full permu 없음)
                                ptag = f"p{int(round(float(perc)))}"
                                pride_methods_png = os.path.join(curve_save_path_pride, f"{subject}_{ptag}_methods_pride_no_full.png")
                                _plot_pride_methods_no_full_png(
                                    curve_obj=cobjp,
                                    out_path=pride_methods_png,
                                    title=f"{args.task} {subject} — PRIDE-only methods (no full, {ptag})",
                                )

                                # 2) PRIDE avggap 단독
                                pride_avggap_png = os.path.join(curve_save_path_pride, f"{subject}_{ptag}_avggap_pride.png")
                                _plot_avggap_curve_png(
                                    curve_obj=cobjp,
                                    out_path=pride_avggap_png,
                                    title=f"{args.task} {subject} — ours_avggap (PRIDE, {ptag})",
                                    label="pride ours_avggap",
                                )

                                # 3) baseline vs pride avggap 비교
                                if float(perc) in baseline_by_p:
                                    compare_avggap_png = os.path.join(curve_save_path_pride, f"{subject}_{ptag}_avggap_compare.png")
                                    _plot_avggap_compare_png(
                                        baseline_curve_obj=baseline_by_p[float(perc)],
                                        pride_curve_obj=cobjp,
                                        out_path=compare_avggap_png,
                                        title=f"{args.task} {subject} — ours_avggap (Baseline vs PRIDE, {ptag})",
                                    )
                                else:
                                    compare_avggap_png = None

                                if wandb_ok and wandb_run is not None:
                                    import wandb
                                    payload = {
                                        f"plots/{subject}/{ptag}/methods_pride_no_full": wandb.Image(pride_methods_png),
                                        f"plots/{subject}/{ptag}/avggap_pride": wandb.Image(pride_avggap_png),
                                    }
                                    if compare_avggap_png is not None:
                                        payload[f"plots/{subject}/{ptag}/avggap_compare"] = wandb.Image(compare_avggap_png)
                                    wandb_run.log(payload)

                        save_results(f'{curve_save_path_pride}/{subject}_beta_curve.jsonl', curve_objs_pride, metrics=None)

                        # =========================================================
                        # Percentile grid (ours_avggap) — baseline/pride/delta
                        # with TEXT (acc + cost)
                        # =========================================================
                        grid_perc = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
                        x_ticks = [float(p) for p in grid_perc]  # th1
                        y_ticks = [float(p) for p in grid_perc]  # th2

                        base_acc_grid = np.zeros((len(y_ticks), len(x_ticks)), dtype=np.float64)
                        base_cost_grid = np.zeros((len(y_ticks), len(x_ticks)), dtype=np.float64)
                        pride_acc_grid = np.zeros((len(y_ticks), len(x_ticks)), dtype=np.float64)
                        pride_cost_grid = np.zeros((len(y_ticks), len(x_ticks)), dtype=np.float64)

                        for iy, th2p in enumerate(y_ticks):
                            for ix, th1p in enumerate(x_ticks):
                                c_b, a_b = _policy_metrics_avggap_beta0(
                                    default_conf=default_conf,
                                    mean_conf=mean_conf,
                                    base_correct=base_correct_list,
                                    cyclic_correct=cyclic_correct_list,
                                    probe2_correct=arr_probe2_correct,
                                    k=k,
                                    th1_percent=th1p,
                                    th2_percent=th2p,
                                )
                                base_acc_grid[iy, ix] = a_b
                                base_cost_grid[iy, ix] = c_b

                                c_p, a_p = _policy_metrics_avggap_beta0(
                                    default_conf=default_conf_pride,
                                    mean_conf=mean_conf_pride,
                                    base_correct=base_correct_pride,
                                    cyclic_correct=cyclic_correct_pride,
                                    probe2_correct=arr_probe2_correct_pride,
                                    k=k,
                                    th1_percent=th1p,
                                    th2_percent=th2p,
                                )
                                pride_acc_grid[iy, ix] = a_p
                                pride_cost_grid[iy, ix] = c_p

                        delta_acc_grid = pride_acc_grid - base_acc_grid
                        delta_cost_grid = pride_cost_grid - base_cost_grid

                        grid_base_png = os.path.join(curve_save_path_pride, f"{subject}_avggap_grid_baseline.png")
                        grid_pride_png = os.path.join(curve_save_path_pride, f"{subject}_avggap_grid_pride.png")
                        grid_delta_png = os.path.join(curve_save_path_pride, f"{subject}_avggap_grid_delta.png")

                        _plot_heatmap_with_text(
                            acc_grid=base_acc_grid, cost_grid=base_cost_grid,
                            x_ticks=x_ticks, y_ticks=y_ticks,
                            out_path=grid_base_png,
                            title=f"{args.task} {subject} — ours_avggap grid (Baseline)  [cell: acc / cost]",
                            xlabel="th1 percentile (base gap)", ylabel="th2 percentile (avg gap)",
                            mode="base",
                        )
                        _plot_heatmap_with_text(
                            acc_grid=pride_acc_grid, cost_grid=pride_cost_grid,
                            x_ticks=x_ticks, y_ticks=y_ticks,
                            out_path=grid_pride_png,
                            title=f"{args.task} {subject} — ours_avggap grid (PRIDE)  [cell: acc / cost]",
                            xlabel="th1 percentile (base gap)", ylabel="th2 percentile (avg gap)",
                            mode="pride",
                        )
                        _plot_heatmap_with_text(
                            acc_grid=delta_acc_grid, cost_grid=delta_cost_grid,
                            x_ticks=x_ticks, y_ticks=y_ticks,
                            out_path=grid_delta_png,
                            title=f"{args.task} {subject} — ours_avggap grid (Δ PRIDE - Baseline)  [cell: Δacc / Δcost]",
                            xlabel="th1 percentile (base gap)", ylabel="th2 percentile (avg gap)",
                            mode="delta",
                        )

                        if wandb_ok and wandb_run is not None:
                            import wandb
                            wandb_run.log({
                                f"grids/{subject}/baseline": wandb.Image(grid_base_png),
                                f"grids/{subject}/pride": wandb.Image(grid_pride_png),
                                f"grids/{subject}/delta": wandb.Image(grid_delta_png),
                            })

                except Exception as e:
                    logger.warning(f"Failed to derive beta curves for subject '{subject}': {e}")
                    import traceback
                    traceback.print_exc()

            logging_cuda_memory_usage()

    # -------- finalize W&B --------
    try:
        if wandb_ok and wandb_run is not None:
            import wandb
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
