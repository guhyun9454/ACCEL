import logging

from utils import _purple


logger = logging.getLogger(__name__)


def _log_baseline_report(curve_obj: dict):
    """
    BASELINE은 풀로 찍고,
    PRIDE_FREE는 (아래 main에서) 한 줄만 찍는다.
    """
    p = curve_obj.get("percentile")
    logger.info(_purple(f"==== BASELINE Derived policy report (REAL-WORLD online, p={p}) ===="))

    always = curve_obj.get("always", {})

    def _recall_str(obj):
        return f", recall_std={obj:.4f}" if isinstance(obj, (int, float)) else ""

    logger.info(f"BASELINE default(ensemble) : cost={always['default']['cost']:.3f}, acc={always['default']['acc']:.4f}{_recall_str(curve_obj.get('default_recall_std'))}")
    logger.info(f"BASELINE cyclic(ensemble)  : cost={always['cyclic']['cost']:.3f}, acc={always['cyclic']['acc']:.4f}{_recall_str(curve_obj.get('cyclic_recall_std'))}")
    if "full" in always:
        logger.info(f"BASELINE full(ensemble)    : cost={always['full']['cost']:.3f}, acc={always['full']['acc']:.4f}{_recall_str(curve_obj.get('full_recall_std'))}")
    else:
        logger.info("BASELINE full(ensemble)    : (disabled)")

    for key in ["switch_full", "switch_cyclic", "ours_top2flip", "ours_avggap"]:
        if key in curve_obj:
            c0 = float(curve_obj[key]["costs"][0])
            a0 = float(curve_obj[key]["accuracies"][0])
            st = curve_obj.get("ours_avggap_stats", {}) if key == "ours_avggap" else curve_obj[key].get("stats", {})
            if not isinstance(st, dict):
                st = {}
            nb = int(st.get("n_base", 0))
            np2 = int(st.get("n_probe2", 0))
            nc = int(st.get("n_cyclic", 0))
            extra = f", n_base={nb}, n_probe2={np2}, n_cyclic={nc}"
            recall_key = f"{key}_recall_std"
            rstd = curve_obj.get(recall_key)
            if isinstance(rstd, (int, float)):
                extra += f", recall_std={rstd:.4f}"
            logger.info(f"BASELINE {key:<12} : cost={c0:.3f}, acc={a0:.4f}{extra}")

    def _log_cyclic_random(obj: dict, prefix: str):
        keys = [
            k for k in (obj or {}).keys()
            if isinstance(k, str) and k.startswith("cyclic_random_") and not k.endswith("_recall_std")
        ]

        def _k_to_float(s: str) -> float:
            try:
                return float(s.replace("cyclic_random_", ""))
            except Exception:
                return float("inf")

        keys = sorted(keys, key=_k_to_float)
        if keys:
            logger.info(_purple(f"---- Cyclic random fractions (plot과 동일) [{prefix}] ----"))
            for k in keys:
                if k in obj and isinstance(obj[k], dict) and "costs" in obj[k] and "accuracies" in obj[k]:
                    c = float(obj[k]["costs"][0])
                    a = float(obj[k]["accuracies"][0])
                    rstd = obj.get(f"{k}_recall_std")
                    extra = f", recall_std={rstd:.4f}" if isinstance(rstd, (int, float)) else ""
                    logger.info(f"BASELINE {k:<16}: cost={c:.3f}, acc={a:.4f}{extra}")

    _log_cyclic_random(curve_obj, "BASELINE")


def _log_named_report(name: str, curve_obj: dict):
    """Same format as baseline report, but with custom header prefix."""
    p = curve_obj.get("percentile")
    logger.info(_purple(f"==== {name} Derived policy report (REAL-WORLD online, p={p}) ===="))

    always = curve_obj.get("always", {})

    def _recall_str(obj):
        return f", recall_std={obj:.4f}" if isinstance(obj, (int, float)) else ""

    if "default" in always:
        logger.info(f"{name} default(ensemble) : cost={always['default']['cost']:.3f}, acc={always['default']['acc']:.4f}{_recall_str(curve_obj.get('default_recall_std'))}")
    if "cyclic" in always:
        logger.info(f"{name} cyclic(ensemble)  : cost={always['cyclic']['cost']:.3f}, acc={always['cyclic']['acc']:.4f}{_recall_str(curve_obj.get('cyclic_recall_std'))}")
    if "full" in always:
        logger.info(f"{name} full(ensemble)    : cost={always['full']['cost']:.3f}, acc={always['full']['acc']:.4f}{_recall_str(curve_obj.get('full_recall_std'))}")

    for key in ["switch_full", "switch_cyclic", "ours_top2flip", "ours_avggap"]:
        if key in curve_obj:
            c0 = float(curve_obj[key]["costs"][0])
            a0 = float(curve_obj[key]["accuracies"][0])
            st = curve_obj.get("ours_avggap_stats", {}) if key == "ours_avggap" else curve_obj[key].get("stats", {})
            if not isinstance(st, dict):
                st = {}
            nb, np2, nc = int(st.get("n_base", 0)), int(st.get("n_probe2", 0)), int(st.get("n_cyclic", 0))
            extra = f", n_base={nb}, n_probe2={np2}, n_cyclic={nc}"
            rstd = curve_obj.get(f"{key}_recall_std")
            if isinstance(rstd, (int, float)):
                extra += f", recall_std={rstd:.4f}"
            logger.info(f"{name} {key:<12} : cost={c0:.3f}, acc={a0:.4f}{extra}")

    fracs = sorted(
        [
            int(k.replace("cyclic_random_", ""))
            for k in (curve_obj or {}).keys()
            if isinstance(k, str) and k.startswith("cyclic_random_") and not k.endswith("_recall_std")
        ],
        key=lambda x: x,
    )
    if fracs:
        logger.info(_purple(f"---- Cyclic random fractions (plot Default+PRIDE와 동일) [{name}] ----"))
        for fp in fracs:
            k = f"cyclic_random_{fp}"
            if k in curve_obj and isinstance(curve_obj[k], dict) and "costs" in curve_obj[k] and "accuracies" in curve_obj[k]:
                c = float(curve_obj[k]["costs"][0])
                a = float(curve_obj[k]["accuracies"][0])
                rstd = curve_obj.get(f"{k}_recall_std")
                extra = f", recall_std={rstd:.4f}" if isinstance(rstd, (int, float)) else ""
                logger.info(f"{name} cyclic_{fp}%      : cost={c:.3f}, acc={a:.4f}{extra}")
