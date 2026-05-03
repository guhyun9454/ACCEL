import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt

from utils import _purple


logger = logging.getLogger(__name__)

PRIMARY_OURS_LABEL = "th1/sqrt2"
LEGACY_OURS_LABEL = "th1/2"
EMPIRICAL_PRIDE_LABEL = "empirical_pride_primary"


def _plot_three_curves_acc_recall_std(
    derived_records_by_p: Dict[float, List[dict]],
    derived_records_pride_by_p: Dict[float, List[dict]],
    derived_records_pride_by_alpha: Dict[float, List[dict]],
    derived_records_empirical_by_alpha: Dict[float, List[dict]],
    out_dir: str,
    task: str,
    cyclic_fractions: List[int],
    pride_ours_fractions: List[float],
    pride_prefix_list: List[float],
    empirical_prefix_list: List[float],
    wandb_ok: bool = False,
    wandb_run: Any = None,
):
    """
    Three curves: (1) Cyclic (no PRIDE), (2) Default+PRIDE, (3) OURS (th1/sqrt2, no PRIDE).
    X=Cost, Y=Accuracy or Recall_std. Fractions configurable via args.
    """
    if not derived_records_by_p:
        logger.debug("three-curves plot skipped: derived_records_by_p empty")
        return
    color_cyclic = "#F39C12"
    color_pride = "#27AE60"
    color_ours = "#5DADE2"
    color_empirical = "#8E44AD"
    n_subjects = len(next(iter(derived_records_by_p.values()), []))
    macro_note = f" (macro over {n_subjects} subjects)" if n_subjects > 1 else ""

    def _pick_preferred_alpha(available_alphas, preferred=2.0):
        vals = [float(x) for x in (available_alphas or [])]
        if not vals:
            return None
        for alpha in vals:
            if abs(float(alpha) - float(preferred)) <= 1e-12:
                return float(alpha)
        return float(vals[0])

    def _agg_cyclic(by_p, fracs):
        costs, accs, rstds, acc_stds, rstd_stds = [], [], [], [], []
        p_any = next((float(p) for p in fracs if float(p) in by_p), None) or next(iter(by_p.keys()), None)
        cobjs = by_p.get(float(p_any), []) if p_any is not None else []
        for fp in fracs:
            key = f"cyclic_random_{fp}"
            cbs = [float(c[key]["costs"][0]) for c in cobjs if key in c]
            abs_ = [float(c[key]["accuracies"][0]) * 100.0 for c in cobjs if key in c]
            rbs = [float(c.get(f"{key}_recall_std", float("nan"))) for c in cobjs if key in c]
            costs.append(np.mean(cbs) if cbs else float("nan"))
            accs.append(np.mean(abs_) if abs_ else float("nan"))
            rstds.append(np.nanmean(rbs) if rbs else float("nan"))
            acc_stds.append(float(np.nanstd(abs_)) if len(abs_) > 1 else 0.0)
            rstd_stds.append(float(np.nanstd(rbs)) if len(rbs) > 1 else 0.0)
        return costs, accs, rstds, acc_stds, rstd_stds

    def _agg_heur(by_p, label, fracs):
        costs, accs, rstds, acc_stds, rstd_stds = [], [], [], [], []
        for p in fracs:
            pts = by_p.get(float(p), [])
            cl, al, rl = [], [], []
            for c in pts:
                hp_map = {str(h.get("label")): h for h in (c.get("heuristic_points", []) or []) if isinstance(h, dict)}
                h = hp_map.get(label, {})
                if h and "cost" in h:
                    cl.append(float(h["cost"]))
                if h and "acc" in h:
                    al.append(float(h["acc"]) * 100.0)
                if h and "recall_std" in h:
                    rl.append(float(h["recall_std"]))
            costs.append(np.mean(cl) if cl else float("nan"))
            accs.append(np.mean(al) if al else float("nan"))
            rstds.append(np.nanmean(rl) if rl else float("nan"))
            acc_stds.append(float(np.nanstd(al)) if len(al) > 1 else 0.0)
            rstd_stds.append(float(np.nanstd(rl)) if len(rl) > 1 else 0.0)
        return costs, accs, rstds, acc_stds, rstd_stds

    def _agg_pride_default(by_p, fracs):
        costs, accs, rstds, acc_stds, rstd_stds = [], [], [], [], []
        for p in fracs:
            pts = by_p.get(float(p), [])
            cl, al, rl = [], [], []
            for c in pts:
                pf = float(p)
                key_candidates = [
                    f"cyclic_random_{pf}",
                    f"cyclic_random_{pf:g}",
                ]
                key = next((kk for kk in key_candidates if kk in c), None)
                if key is not None:
                    cl.append(float(c[key]["costs"][0]))
                    al.append(float(c[key]["accuracies"][0]) * 100.0)
                    rkey = f"{key}_recall_std"
                    if rkey in c and isinstance(c.get(rkey), (int, float)):
                        rl.append(float(c[rkey]))
            costs.append(np.mean(cl) if cl else float("nan"))
            accs.append(np.mean(al) if al else float("nan"))
            rstds.append(np.nanmean(rl) if rl else float("nan"))
            acc_stds.append(float(np.nanstd(al)) if len(al) > 1 else 0.0)
            rstd_stds.append(float(np.nanstd(rl)) if len(rl) > 1 else 0.0)
        return costs, accs, rstds, acc_stds, rstd_stds

    def _agg_heur_by_sweep(cobjs_list, sweep_values, sweep_key, label_filter="online_sqrt_all"):
        costs, accs, rstds, acc_stds, rstd_stds = [], [], [], [], []
        for sweep_value in sweep_values:
            cl, al, rl = [], [], []
            for c in cobjs_list:
                for h in (c.get("heuristic_points", []) or []):
                    if isinstance(h, dict) and h.get(sweep_key) == sweep_value and h.get("label") == label_filter:
                        if "cost" in h:
                            cl.append(float(h["cost"]))
                        if "acc" in h:
                            al.append(float(h["acc"]) * 100.0)
                        if "recall_std" in h:
                            rl.append(float(h["recall_std"]))
                        break
            costs.append(np.mean(cl) if cl else float("nan"))
            accs.append(np.mean(al) if al else float("nan"))
            rstds.append(np.nanmean(rl) if rl else float("nan"))
            acc_stds.append(float(np.nanstd(al)) if len(al) > 1 else 0.0)
            rstd_stds.append(float(np.nanstd(rl)) if len(rl) > 1 else 0.0)
        return costs, accs, rstds, acc_stds, rstd_stds

    def _agg_heur_by_th1_p(cobjs_list, th1_list, label_filter="online_sqrt_all"):
        return _agg_heur_by_sweep(cobjs_list, th1_list, "th1_p", label_filter=label_filter)

    def _infer_empirical_sweep(by_alpha, prefix_list, fallback_percentiles):
        for alpha in prefix_list:
            for c in (by_alpha.get(alpha, []) or []):
                if not isinstance(c, dict):
                    continue
                mode = str(c.get("sweep_mode", "percentile")).strip().lower()
                if mode not in {"percentile", "confidence"}:
                    mode = "percentile"
                residual_model = str(c.get("residual_model", "empirical")).strip().lower()
                percentile_mode = str(c.get("percentile_mode", "online")).strip().lower()
                if percentile_mode not in {"online", "fixed_prefix"}:
                    percentile_mode = "online"
                schedule = str(c.get("threshold_schedule", "flat")).strip().lower()
                if schedule not in {"flat", "sqrt"}:
                    schedule = "flat"
                gamma = float(c.get("threshold_gamma", 0.5)) if isinstance(c.get("threshold_gamma"), (int, float)) else 0.5
                transition_mode = str(c.get("transition_mode", "latin")).strip().lower()
                if transition_mode not in {"latin", "probe_cyclic", "cyclic_random", "cyclic_targeted", "cyclic_learned"}:
                    transition_mode = "latin"
                skip_residual = bool(c.get("skip_residual_on_cyclic", False))
                for h in (c.get("heuristic_points") or []):
                    if not isinstance(h, dict) or h.get("label") != EMPIRICAL_PRIDE_LABEL:
                        continue
                    if mode == "confidence" and h.get("conf_th") is not None:
                        vals = sorted({
                            float(hp.get("conf_th"))
                            for cc in (by_alpha.get(alpha, []) or [])
                            for hp in (cc.get("heuristic_points") or [])
                            if isinstance(hp, dict) and hp.get("label") == EMPIRICAL_PRIDE_LABEL and hp.get("conf_th") is not None
                        })
                        return mode, "conf_th", vals, schedule, gamma, transition_mode, skip_residual, residual_model, percentile_mode
                    if h.get("th1_p") is not None:
                        vals = sorted({
                            float(hp.get("th1_p"))
                            for cc in (by_alpha.get(alpha, []) or [])
                            for hp in (cc.get("heuristic_points") or [])
                            if isinstance(hp, dict) and hp.get("label") == EMPIRICAL_PRIDE_LABEL and hp.get("th1_p") is not None
                        })
                        return "percentile", "th1_p", vals, schedule, gamma, transition_mode, skip_residual, residual_model, percentile_mode
        return "percentile", "th1_p", [float(x) for x in fallback_percentiles], "flat", 0.5, "latin", False, "empirical", "online"

    cost_cyc, acc_cyc, rstd_cyc, acc_std_cyc, rstd_std_cyc = _agg_cyclic(derived_records_by_p, cyclic_fractions)
    _n = len(pride_prefix_list) if pride_prefix_list else len(pride_ours_fractions)
    _def5 = ([float("nan")] * _n, [float("nan")] * _n, [float("nan")] * _n, [0.0] * _n, [0.0] * _n)

    pride_fracs_for_plot = pride_ours_fractions
    if derived_records_pride_by_alpha:
        fracs_pride = [p for p in pride_prefix_list if p in derived_records_pride_by_alpha]
        by_p_def = {float(alpha): derived_records_pride_by_alpha[alpha] for alpha in fracs_pride}
        cost_pride, acc_pride, rstd_pride, acc_std_pride, rstd_std_pride = _agg_pride_default(by_p_def, fracs_pride) if by_p_def else _def5
        pride_fracs_for_plot = fracs_pride
    else:
        cost_pride, acc_pride, rstd_pride, acc_std_pride, rstd_std_pride = _agg_pride_default(derived_records_pride_by_p, pride_ours_fractions) if derived_records_pride_by_p else _def5

    cost_ours, acc_ours, rstd_ours, acc_std_ours, rstd_std_ours = _agg_heur(derived_records_by_p, PRIMARY_OURS_LABEL, pride_ours_fractions)

    if derived_records_pride_by_alpha:
        alpha_ours = _pick_preferred_alpha(pride_prefix_list, preferred=2.0)
        cobjs_op = derived_records_pride_by_alpha.get(alpha_ours, [])
        cost_ours_pride, acc_ours_pride, rstd_ours_pride, acc_std_ours_pride, rstd_std_ours_pride = _agg_heur_by_th1_p(cobjs_op, pride_ours_fractions, PRIMARY_OURS_LABEL) if cobjs_op else _def5
        cost_ours_pride_th12, acc_ours_pride_th12, rstd_ours_pride_th12, acc_std_ours_pride_th12, rstd_std_ours_pride_th12 = _agg_heur_by_th1_p(cobjs_op, pride_ours_fractions, LEGACY_OURS_LABEL) if cobjs_op else _def5
    else:
        cost_ours_pride, acc_ours_pride, rstd_ours_pride, acc_std_ours_pride, rstd_std_ours_pride = _agg_heur(derived_records_pride_by_p, PRIMARY_OURS_LABEL, pride_ours_fractions) if derived_records_pride_by_p else _def5
        cost_ours_pride_th12, acc_ours_pride_th12, rstd_ours_pride_th12, acc_std_ours_pride_th12, rstd_std_ours_pride_th12 = _agg_heur(derived_records_pride_by_p, LEGACY_OURS_LABEL, pride_ours_fractions) if derived_records_pride_by_p else _def5

    if derived_records_empirical_by_alpha:
        empirical_alpha = _pick_preferred_alpha(empirical_prefix_list or list(derived_records_empirical_by_alpha.keys()), preferred=2.0)
        empirical_cobjs = derived_records_empirical_by_alpha.get(empirical_alpha, []) if empirical_alpha is not None else []
        empirical_mode, empirical_sweep_key, empirical_sweep_values, empirical_schedule, empirical_gamma, empirical_transition_mode, empirical_skip_residual, empirical_residual_model, empirical_percentile_mode = _infer_empirical_sweep(
            derived_records_empirical_by_alpha, empirical_prefix_list, pride_ours_fractions
        )
        cost_empirical, acc_empirical, rstd_empirical, acc_std_empirical, rstd_std_empirical = _agg_heur_by_sweep(
            empirical_cobjs, empirical_sweep_values, empirical_sweep_key, EMPIRICAL_PRIDE_LABEL
        ) if empirical_cobjs else _def5
    else:
        empirical_mode, empirical_sweep_key, empirical_sweep_values, empirical_schedule, empirical_gamma, empirical_transition_mode, empirical_skip_residual, empirical_residual_model, empirical_percentile_mode = "percentile", "th1_p", [float(x) for x in pride_ours_fractions], "flat", 0.5, "latin", False, "empirical", "online"
        cost_empirical, acc_empirical, rstd_empirical, acc_std_empirical, rstd_std_empirical = _def5

    default_acc = float(acc_cyc[0]) if acc_cyc and np.isfinite(acc_cyc[0]) else float("nan")
    default_recall_std = float(rstd_cyc[0]) if rstd_cyc and np.isfinite(rstd_cyc[0]) else float("nan")

    delta_acc_cyc = [float(a - default_acc) if np.isfinite(a) and np.isfinite(default_acc) else float("nan") for a in acc_cyc]
    delta_acc_pride = [float(a - default_acc) if np.isfinite(a) and np.isfinite(default_acc) else float("nan") for a in acc_pride]
    delta_acc_ours = [float(a - default_acc) if np.isfinite(a) and np.isfinite(default_acc) else float("nan") for a in acc_ours]
    delta_acc_ours_pride = [float(a - default_acc) if np.isfinite(a) and np.isfinite(default_acc) else float("nan") for a in acc_ours_pride]
    delta_acc_ours_pride_th12 = [float(a - default_acc) if np.isfinite(a) and np.isfinite(default_acc) else float("nan") for a in acc_ours_pride_th12]
    delta_acc_empirical = [float(a - default_acc) if np.isfinite(a) and np.isfinite(default_acc) else float("nan") for a in acc_empirical]

    delta_rstd_cyc = [float(default_recall_std - r) if np.isfinite(r) and np.isfinite(default_recall_std) else float("nan") for r in rstd_cyc]
    delta_rstd_pride = [float(default_recall_std - r) if np.isfinite(r) and np.isfinite(default_recall_std) else float("nan") for r in rstd_pride]
    delta_rstd_ours = [float(default_recall_std - r) if np.isfinite(r) and np.isfinite(default_recall_std) else float("nan") for r in rstd_ours]
    delta_rstd_ours_pride = [float(default_recall_std - r) if np.isfinite(r) and np.isfinite(default_recall_std) else float("nan") for r in rstd_ours_pride]
    delta_rstd_ours_pride_th12 = [float(default_recall_std - r) if np.isfinite(r) and np.isfinite(default_recall_std) else float("nan") for r in rstd_ours_pride_th12]
    delta_rstd_empirical = [float(default_recall_std - r) if np.isfinite(r) and np.isfinite(default_recall_std) else float("nan") for r in rstd_empirical]

    def _plot_curve(ax, costs, yvals, marker, color, linestyle, label):
        valid = [(c, y) for c, y in zip(costs, yvals) if np.isfinite(c) and np.isfinite(y)]
        if not valid:
            return
        xs, ys = zip(*sorted(valid, key=lambda t: t[0]))
        ax.plot(xs, ys, marker=marker, color=color, linestyle=linestyle, linewidth=2, markersize=8, label=label)

    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=160)
    _plot_curve(ax, cost_cyc, delta_acc_cyc, "o", color_cyclic, "-", "Cyclic")
    _plot_curve(ax, cost_pride, delta_acc_pride, "s", color_pride, "--", "PriDe")
    _plot_curve(ax, cost_ours_pride_th12, delta_acc_ours_pride_th12, "*", color_ours, "-.", "Ours+PriDe (th1/2)")
    _plot_curve(ax, cost_empirical, delta_acc_empirical, "X", color_empirical, "-", "Empirical PriDe")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel("Computational Cost (× of default forward pass)", fontsize=11)
    ax.set_ylabel("Δ Accuracy (%)", fontsize=11)
    ax.set_title(f"{task} — Δ Accuracy{macro_note}", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_acc = os.path.join(out_dir, f"{task}_three_curves_acc.png")
    fig.savefig(out_acc, bbox_inches="tight")
    plt.close(fig)
    logger.info(_purple(f"Saved three-curves delta acc: {out_acc}"))
    if wandb_ok and wandb_run is not None:
        try:
            import wandb
            wandb_run.log({f"plots/{task}/three_curves_acc": wandb.Image(out_acc)})
        except Exception:
            pass

    fig2, ax2 = plt.subplots(figsize=(10, 6.5), dpi=160)
    _plot_curve(ax2, cost_cyc, delta_rstd_cyc, "o", color_cyclic, "-", "Cyclic")
    _plot_curve(ax2, cost_pride, delta_rstd_pride, "s", color_pride, "--", "PriDe")
    _plot_curve(ax2, cost_ours_pride_th12, delta_rstd_ours_pride_th12, "*", color_ours, "-.", "Ours+PriDe (th1/2)")
    _plot_curve(ax2, cost_empirical, delta_rstd_empirical, "X", color_empirical, "-", "Empirical PriDe")
    ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
    ax2.set_xlabel("Computational Cost (× of default forward pass)", fontsize=11)
    ax2.set_ylabel("Δ Recall std", fontsize=11)
    ax2.set_title(f"{task} — Δ Recall std{macro_note}", fontsize=12)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)
    fig2.tight_layout()
    out_rstd = os.path.join(out_dir, f"{task}_three_curves_recall_std.png")
    fig2.savefig(out_rstd, bbox_inches="tight")
    plt.close(fig2)
    logger.info(_purple(f"Saved three-curves delta recall_std: {out_rstd}"))
    if wandb_ok and wandb_run is not None:
        try:
            import wandb
            wandb_run.log({f"plots/{task}/three_curves_recall_std": wandb.Image(out_rstd)})
        except Exception:
            pass

    fig3, ax3 = plt.subplots(figsize=(10, 6.5), dpi=160)
    _plot_curve(ax3, cost_ours, delta_acc_ours, "^", color_ours, "-.", "Ours")
    _plot_curve(ax3, cost_ours_pride, delta_acc_ours_pride, "D", color_pride, "--", "Ours (with PriDe)")
    ax3.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
    ax3.set_xlabel("Computational Cost (× of default forward pass)", fontsize=11)
    ax3.set_ylabel("Δ Accuracy (%)", fontsize=11)
    ax3.set_title(f"{task} — Ours vs Ours+PRIDE Δ Accuracy{macro_note}", fontsize=12)
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.4)
    fig3.tight_layout()
    out_ours_acc = os.path.join(out_dir, f"{task}_ours_vs_ours_pride_acc.png")
    fig3.savefig(out_ours_acc, bbox_inches="tight")
    plt.close(fig3)
    logger.info(_purple(f"Saved ours vs ours_pride delta acc: {out_ours_acc}"))
    if wandb_ok and wandb_run is not None:
        try:
            import wandb
            wandb_run.log({f"plots/{task}/ours_vs_ours_pride_acc": wandb.Image(out_ours_acc)})
        except Exception:
            pass

    fig4, ax4 = plt.subplots(figsize=(10, 6.5), dpi=160)
    _plot_curve(ax4, cost_ours, delta_rstd_ours, "^", color_ours, "-.", "Ours")
    _plot_curve(ax4, cost_ours_pride, delta_rstd_ours_pride, "D", color_pride, "--", "Ours (with PriDe)")
    ax4.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
    ax4.set_xlabel("Computational Cost (× of default forward pass)", fontsize=11)
    ax4.set_ylabel("Δ Recall std", fontsize=11)
    ax4.set_title(f"{task} — Ours vs Ours+PRIDE Δ Recall std{macro_note}", fontsize=12)
    ax4.legend(loc="best", fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.4)
    fig4.tight_layout()
    out_ours_rstd = os.path.join(out_dir, f"{task}_ours_vs_ours_pride_recall_std.png")
    fig4.savefig(out_ours_rstd, bbox_inches="tight")
    plt.close(fig4)
    logger.info(_purple(f"Saved ours vs ours_pride delta recall_std: {out_ours_rstd}"))
    if wandb_ok and wandb_run is not None:
        try:
            import wandb
            wandb_run.log({f"plots/{task}/ours_vs_ours_pride_recall_std": wandb.Image(out_ours_rstd)})
        except Exception:
            pass

    def _build_ours_pride_payload(
        by_alpha, prefix_list, th1_fracs, def_acc, def_rstd, agg_fn,
        cost_legacy, acc_legacy, rstd_legacy, dacc_legacy, drstd_legacy, acc_std_legacy, rstd_std_legacy,
    ):
        if by_alpha:
            by_alpha_out = {}
            for alpha in prefix_list:
                cobjs = by_alpha.get(alpha, [])
                if not cobjs:
                    continue
                entry = {}
                for variant in ("th1/2", PRIMARY_OURS_LABEL, "online_sqrt_all"):
                    co, ac, rs, asd, rsd = agg_fn(cobjs, th1_fracs, variant)
                    entry[variant] = {
                        "p": [float(x) for x in th1_fracs],
                        "cost": [float(x) if np.isfinite(x) else float("nan") for x in co],
                        "acc": [float(x) if np.isfinite(x) else float("nan") for x in ac],
                        "recall_std": [float(x) if np.isfinite(x) else float("nan") for x in rs],
                        "delta_acc": [float(a - def_acc) if np.isfinite(a) and np.isfinite(def_acc) else float("nan") for a in ac],
                        "delta_recall_std": [float(def_rstd - r) if np.isfinite(r) and np.isfinite(def_rstd) else float("nan") for r in rs],
                        "delta_acc_std": [float(x) if np.isfinite(x) else 0.0 for x in asd],
                        "delta_recall_std_std": [float(x) if np.isfinite(x) else 0.0 for x in rsd],
                    }
                alpha_key = f"{float(alpha):g}"
                by_alpha_out[alpha_key] = entry
            return {
                "pride_prefix_fractions": [float(a) for a in prefix_list],
                "p": [float(x) for x in th1_fracs],
                "by_alpha": by_alpha_out,
            }
        return {
            "pride_prefix_fractions": [],
            "p": [int(x) for x in th1_fracs],
            "by_alpha": {},
            "cost": [float(x) if np.isfinite(x) else float("nan") for x in cost_legacy],
            "acc": [float(x) if np.isfinite(x) else float("nan") for x in acc_legacy],
            "recall_std": [float(x) if np.isfinite(x) else float("nan") for x in rstd_legacy],
            "delta_acc": [float(x) if np.isfinite(x) else float("nan") for x in dacc_legacy],
            "delta_recall_std": [float(x) if np.isfinite(x) else float("nan") for x in drstd_legacy],
            "delta_acc_std": [float(x) if np.isfinite(x) else 0.0 for x in acc_std_legacy],
            "delta_recall_std_std": [float(x) if np.isfinite(x) else 0.0 for x in rstd_std_legacy],
        }

    def _build_empirical_payload(by_alpha, prefix_list, percentile_fracs, def_acc, def_rstd):
        empirical_mode, empirical_sweep_key, empirical_sweep_values, empirical_schedule, empirical_gamma, empirical_transition_mode, empirical_skip_residual, empirical_residual_model, empirical_percentile_mode = _infer_empirical_sweep(by_alpha, prefix_list, percentile_fracs)
        payload_sweep_key = "confidence" if empirical_sweep_key == "conf_th" else "p"
        by_alpha_out = {}
        selection_policy = None
        selected_sequence_name = None
        selected_action_sequence = None
        for alpha in prefix_list:
            cobjs = by_alpha.get(alpha, [])
            if not cobjs:
                continue
            if selection_policy is None:
                selection_policy = cobjs[0].get("selection_policy")
                selected_sequence_name = cobjs[0].get("selected_sequence_name")
                selected_action_sequence = cobjs[0].get("selected_action_sequence")
            co, ac, rs, asd, rsd = _agg_heur_by_sweep(cobjs, empirical_sweep_values, empirical_sweep_key, EMPIRICAL_PRIDE_LABEL)
            seq_counts = {}
            for c in cobjs:
                seq_name = str(c.get("selected_sequence_name", "")).strip()
                if not seq_name:
                    continue
                seq_counts[seq_name] = int(seq_counts.get(seq_name, 0)) + 1
            by_alpha_out[f"{float(alpha):g}"] = {
                "primary": {
                    payload_sweep_key: [float(x) for x in empirical_sweep_values],
                    "cost": [float(x) if np.isfinite(x) else float("nan") for x in co],
                    "acc": [float(x) if np.isfinite(x) else float("nan") for x in ac],
                    "recall_std": [float(x) if np.isfinite(x) else float("nan") for x in rs],
                    "delta_acc": [float(a - def_acc) if np.isfinite(a) and np.isfinite(def_acc) else float("nan") for a in ac],
                    "delta_recall_std": [float(def_rstd - r) if np.isfinite(r) and np.isfinite(def_rstd) else float("nan") for r in rs],
                    "delta_acc_std": [float(x) if np.isfinite(x) else 0.0 for x in asd],
                    "delta_recall_std_std": [float(x) if np.isfinite(x) else 0.0 for x in rsd],
                },
                "selection": {
                    "policy": cobjs[0].get("selection_policy"),
                    "selected_sequence_name": cobjs[0].get("selected_sequence_name"),
                    "selected_action_sequence": cobjs[0].get("selected_action_sequence"),
                    "sequence_counts": seq_counts,
                },
            }
        return {
            "pride_prefix_fractions": [float(a) for a in prefix_list],
            "empirical_prefix_fractions": [float(a) for a in prefix_list],
            "sweep_mode": empirical_mode,
            "percentile_mode": empirical_percentile_mode,
            "residual_model": empirical_residual_model,
            "transition_mode": empirical_transition_mode,
            "selection_policy": selection_policy,
            "selected_sequence_name": selected_sequence_name,
            "selected_action_sequence": selected_action_sequence,
            "skip_residual_on_cyclic": bool(empirical_skip_residual),
            "threshold_schedule": empirical_schedule,
            "threshold_gamma": float(empirical_gamma),
            payload_sweep_key: [float(x) for x in empirical_sweep_values],
            "by_alpha": by_alpha_out,
        }

    try:
        points_path = os.path.join(out_dir, f"{task}_three_curves_points.json")
        payload = {
            "version": 2,
            "task": str(task),
            "default_acc": float(default_acc),
            "default_recall_std": float(default_recall_std),
            "cyclic_fractions": [int(x) for x in cyclic_fractions],
            "pride_ours_fractions": [float(x) for x in pride_ours_fractions],
            "curves": {
                "cyclic": {
                    "fraction": [int(x) for x in cyclic_fractions],
                    "cost": [float(x) if np.isfinite(x) else float("nan") for x in cost_cyc],
                    "acc": [float(x) if np.isfinite(x) else float("nan") for x in acc_cyc],
                    "recall_std": [float(x) if np.isfinite(x) else float("nan") for x in rstd_cyc],
                    "delta_acc": [float(x) if np.isfinite(x) else float("nan") for x in delta_acc_cyc],
                    "delta_recall_std": [float(x) if np.isfinite(x) else float("nan") for x in delta_rstd_cyc],
                    "delta_acc_std": [float(x) if np.isfinite(x) else 0.0 for x in acc_std_cyc],
                    "delta_recall_std_std": [float(x) if np.isfinite(x) else 0.0 for x in rstd_std_cyc],
                },
                "default_pride": {
                    "p": [float(x) for x in pride_fracs_for_plot],
                    "cost": [float(x) if np.isfinite(x) else float("nan") for x in cost_pride],
                    "acc": [float(x) if np.isfinite(x) else float("nan") for x in acc_pride],
                    "recall_std": [float(x) if np.isfinite(x) else float("nan") for x in rstd_pride],
                    "delta_acc": [float(x) if np.isfinite(x) else float("nan") for x in delta_acc_pride],
                    "delta_recall_std": [float(x) if np.isfinite(x) else float("nan") for x in delta_rstd_pride],
                    "delta_acc_std": [float(x) if np.isfinite(x) else 0.0 for x in acc_std_pride],
                    "delta_recall_std_std": [float(x) if np.isfinite(x) else 0.0 for x in rstd_std_pride],
                },
                "ours": {
                    "label": PRIMARY_OURS_LABEL,
                    "p": [float(x) for x in pride_ours_fractions],
                    "cost": [float(x) if np.isfinite(x) else float("nan") for x in cost_ours],
                    "acc": [float(x) if np.isfinite(x) else float("nan") for x in acc_ours],
                    "recall_std": [float(x) if np.isfinite(x) else float("nan") for x in rstd_ours],
                    "delta_acc": [float(x) if np.isfinite(x) else float("nan") for x in delta_acc_ours],
                    "delta_recall_std": [float(x) if np.isfinite(x) else float("nan") for x in delta_rstd_ours],
                    "delta_acc_std": [float(x) if np.isfinite(x) else 0.0 for x in acc_std_ours],
                    "delta_recall_std_std": [float(x) if np.isfinite(x) else 0.0 for x in rstd_std_ours],
                },
                "ours_pride": _build_ours_pride_payload(
                    derived_records_pride_by_alpha,
                    pride_prefix_list,
                    pride_ours_fractions,
                    default_acc,
                    default_recall_std,
                    _agg_heur_by_th1_p,
                    cost_ours_pride,
                    acc_ours_pride,
                    rstd_ours_pride,
                    delta_acc_ours_pride,
                    delta_rstd_ours_pride,
                    acc_std_ours_pride,
                    rstd_std_ours_pride,
                ),
                "empirical_pride": _build_empirical_payload(
                    derived_records_empirical_by_alpha,
                    empirical_prefix_list,
                    pride_ours_fractions,
                    default_acc,
                    default_recall_std,
                ),
            },
        }
        with open(points_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(_purple(f"Saved three-curves points: {points_path}"))

        if wandb_ok and wandb_run is not None:
            try:
                import wandb
                existing = wandb_run.summary.get("three_curves_points_v1", {})
                if not isinstance(existing, dict):
                    existing = {}
                existing = dict(existing)
                existing[str(task)] = payload
                wandb_run.summary["three_curves_points_v1"] = existing

                art_name = f"three-curves-points-{str(task)}-{wandb_run.id}"
                art = wandb.Artifact(name=art_name, type="three_curves_points")
                art.add_file(points_path)
                wandb_run.log_artifact(art)
            except Exception as e:
                logger.warning(f"W&B three-curves points logging failed: {e}")
    except Exception as e:
        logger.warning(f"Failed to save three-curves points json: {e}")
