"""Pure parsing helpers for legacy and commercial-API W&B summaries."""

from typing import Any, Dict, List, Mapping


def extract_api_by_task(summary: Mapping[str, Any]) -> Dict[str, dict]:
    raw = (summary or {}).get("api_evaluation_v1", {}) or {}
    if not isinstance(raw, Mapping):
        return {}
    return {str(task): dict(payload) for task, payload in raw.items() if isinstance(payload, Mapping)}


def summary_task_names(points_by_task: Any, api_by_task: Any) -> List[str]:
    point_keys = points_by_task.keys() if isinstance(points_by_task, Mapping) else []
    api_keys = api_by_task.keys() if isinstance(api_by_task, Mapping) else []
    return sorted({str(key) for key in point_keys} | {str(key) for key in api_keys})
