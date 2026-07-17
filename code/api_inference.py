"""Commercial API inference helpers for option-label log-probability evaluation.

The public surface intentionally stays small:

* :class:`CommercialAPIClient` sends one chat request and returns a normalized
  distribution over the requested option labels.
* Provider responses are normalized into :class:`LabelProbabilityResponse`.
* Every successful response is cached without credentials and is accompanied by
  token usage and price-snapshot metadata.

Provider SDK imports are lazy so local Hugging Face evaluation does not require
commercial API dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PRICE_SNAPSHOT_DATE = "2026-07-14"


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "openai:gpt-4.1-2025-04-14": {
        "provider": "openai",
        "model": "gpt-4.1-2025-04-14",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
        "input_usd_per_mtok": 2.00,
        "cached_input_usd_per_mtok": 0.50,
        "output_usd_per_mtok": 8.00,
    },
    "gemini:gemini-2.5-flash-lite": {
        "provider": "gemini",
        "model": "gemini-2.5-flash-lite",
        "api_key_env": "GEMINI_API_KEY",
        "base_url": None,
        "input_usd_per_mtok": 0.18,
        "cached_input_usd_per_mtok": None,
        "output_usd_per_mtok": 0.72,
    },
    "deepseek:deepseek-v4-flash": {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "input_usd_per_mtok": 0.14,
        "cached_input_usd_per_mtok": None,
        "output_usd_per_mtok": 0.28,
        "extra_body": {"thinking": {"type": "disabled"}},
    },
    "together:Qwen/Qwen3.5-397B-A17B": {
        "provider": "together",
        "model": "Qwen/Qwen3.5-397B-A17B",
        "api_key_env": "TOGETHER_API_KEY",
        "base_url": "https://api.together.xyz/v1",
        "input_usd_per_mtok": 0.60,
        "cached_input_usd_per_mtok": None,
        "output_usd_per_mtok": 3.60,
        "extra_body": {"reasoning": {"enabled": False}},
    },
}


class APIInferenceError(Exception):
    """Base error for commercial inference failures."""


class LabelCoverageError(APIInferenceError):
    """Raised when top log-probabilities do not contain every option label."""

    def __init__(self, missing_labels: Sequence[str], top_tokens: Sequence[str]):
        self.missing_labels = [str(x) for x in missing_labels]
        self.top_tokens = [str(x) for x in top_tokens]
        super().__init__(
            "top logprobs missing required labels "
            f"{self.missing_labels}; returned tokens={self.top_tokens}"
        )


class APIBudgetExceeded(APIInferenceError):
    """Raised before a request that would exceed an explicit runtime guard."""


class OnlinePercentileRouter:
    """Online stage-confidence thresholds shared by real adaptive API runs."""

    def __init__(self, k: int, percentile: float, schedule: str = "flat", gamma: float = 0.5):
        self.k = int(k)
        self.percentile = float(percentile)
        self.schedule = str(schedule).strip().lower()
        self.gamma = float(gamma)
        if self.k <= 0 or not 0.0 <= self.percentile <= 100.0:
            raise ValueError("invalid adaptive router configuration")
        if self.schedule not in {"flat", "sqrt"}:
            raise ValueError("schedule must be flat or sqrt")
        self.histories: List[List[float]] = [[] for _ in range(self.k)]

    @staticmethod
    def _quantile(values: Sequence[float], q: float) -> float:
        ordered = sorted(float(x) for x in values if math.isfinite(float(x)))
        if not ordered:
            return 0.0
        position = (len(ordered) - 1) * min(max(float(q), 0.0), 1.0)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    def threshold(self, stage_id: int) -> float:
        stage_id = int(stage_id)
        if not 1 <= stage_id <= self.k:
            raise ValueError(f"stage_id must be in [1, {self.k}]")
        percentile = self.percentile
        if self.schedule == "sqrt":
            percentile /= float(stage_id) ** self.gamma
        return self._quantile(self.histories[stage_id - 1], percentile / 100.0)

    def should_stop(self, stage_id: int, confidence: float) -> bool:
        return float(confidence) >= self.threshold(stage_id)

    def observe(self, confidences: Sequence[float]) -> None:
        for idx, confidence in enumerate(confidences[: self.k]):
            self.histories[idx].append(float(confidence))


@dataclass
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.input_tokens += int(other.input_tokens)
        self.cached_input_tokens += int(other.cached_input_tokens)
        self.output_tokens += int(other.output_tokens)
        self.reasoning_tokens += int(other.reasoning_tokens)
        self.total_tokens += int(other.total_tokens)


@dataclass
class LabelProbabilityResponse:
    label_probs: List[float]
    label_logprobs: Dict[str, float]
    top_tokens: List[Dict[str, Any]]
    usage: TokenUsage
    requested_model: str
    returned_model: str
    provider: str
    response_id: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cache_hit: bool = False
    retry_count: int = 0
    request_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["usage"] = asdict(self.usage)
        return out

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "LabelProbabilityResponse":
        data = dict(obj)
        data["usage"] = TokenUsage(**dict(data.get("usage") or {}))
        return cls(**data)


@dataclass
class _Meter:
    requests: int = 0
    cache_hits: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0

    def add(self, response: LabelProbabilityResponse, cache_hit: bool) -> None:
        self.requests += 1
        self.cache_hits += 1 if cache_hit else 0
        self.usage.add(response.usage)
        self.cost_usd += float(response.cost_usd)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requests": int(self.requests),
            "cache_hits": int(self.cache_hits),
            "usage": asdict(self.usage),
            "cost_usd": float(self.cost_usd),
        }


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _first(value: Any) -> Any:
    try:
        return value[0]
    except Exception:
        return None


def _canonical_label(token: Any, labels: Sequence[str]) -> Optional[str]:
    text = str(token or "").strip()
    # Providers sometimes return punctuation around the single option token.
    while text and text[0] in "`'\"([{":
        text = text[1:].lstrip()
    while text and text[-1] in "`'\")]}.,:":
        text = text[:-1].rstrip()
    for label in labels:
        if text == str(label) or text.upper() == str(label).upper():
            return str(label)
    return None


def normalize_label_logprobs(
    top_entries: Iterable[Any], labels: Sequence[str]
) -> tuple[List[float], Dict[str, float], List[Dict[str, Any]]]:
    """Convert first-token top candidates into a strict label distribution.

    Whitespace/case variants of the same option ID are summed in probability
    space, mirroring the local evaluator's aggregation of ``" A"`` and ``"A"``
    token variants.  The selected labels are then renormalized among themselves.
    """

    labels = [str(x) for x in labels]
    masses = {label: 0.0 for label in labels}
    sanitized: List[Dict[str, Any]] = []
    for entry in list(top_entries or []):
        token = _get(entry, "token", "")
        logprob = _get(
            entry,
            "logprob",
            _get(entry, "log_probability", _get(entry, "logProbability", None)),
        )
        try:
            lp = float(logprob)
        except (TypeError, ValueError):
            continue
        sanitized.append({"token": str(token), "logprob": lp})
        label = _canonical_label(token, labels)
        if label is not None and math.isfinite(lp):
            masses[label] += math.exp(lp)

    missing = [label for label in labels if not (masses[label] > 0.0)]
    if missing:
        raise LabelCoverageError(missing, [row["token"] for row in sanitized])

    total = sum(masses.values())
    if not math.isfinite(total) or total <= 0.0:
        raise APIInferenceError("invalid option-label probability mass")
    probs = [masses[label] / total for label in labels]
    label_logprobs = {label: math.log(max(masses[label], 1e-300)) for label in labels}
    return probs, label_logprobs, sanitized


def _openai_usage(response: Any) -> TokenUsage:
    usage = _get(response, "usage", {}) or {}
    prompt_details = _get(usage, "prompt_tokens_details", {}) or {}
    completion_details = _get(usage, "completion_tokens_details", {}) or {}
    prompt_tokens = int(_get(usage, "prompt_tokens", 0) or 0)
    cached_tokens = int(
        _get(prompt_details, "cached_tokens", _get(usage, "prompt_cache_hit_tokens", 0)) or 0
    )
    completion_tokens = int(_get(usage, "completion_tokens", 0) or 0)
    reasoning_tokens = int(_get(completion_details, "reasoning_tokens", 0) or 0)
    total_tokens = int(_get(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    return TokenUsage(
        input_tokens=prompt_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
    )


def normalize_openai_compatible_response(
    response: Any,
    *,
    provider: str,
    requested_model: str,
    labels: Sequence[str],
) -> LabelProbabilityResponse:
    choice = _first(_get(response, "choices", []))
    logprobs = _get(choice, "logprobs", None)
    content = _get(logprobs, "content", None)
    first_token = _first(content or [])
    top_entries = _get(first_token, "top_logprobs", None)
    if not top_entries:
        # Together can return the legacy completion-style shape where the first
        # top-logprob set is a token -> logprob mapping.
        legacy_top = _first(_get(logprobs, "top_logprobs", []) or [])
        if isinstance(legacy_top, Mapping):
            top_entries = [
                {"token": token, "logprob": logprob}
                for token, logprob in legacy_top.items()
            ]
        elif isinstance(legacy_top, list):
            top_entries = legacy_top
    if not top_entries:
        raise APIInferenceError("response does not contain first-token top logprobs")
    probs, label_lps, sanitized = normalize_label_logprobs(top_entries, labels)
    return LabelProbabilityResponse(
        label_probs=probs,
        label_logprobs=label_lps,
        top_tokens=sanitized,
        usage=_openai_usage(response),
        requested_model=requested_model,
        returned_model=str(_get(response, "model", requested_model) or requested_model),
        provider=provider,
        response_id=str(_get(response, "id", "") or ""),
    )


def _gemini_usage(response: Any) -> TokenUsage:
    usage = _get(response, "usage_metadata", _get(response, "usageMetadata", {})) or {}
    prompt = int(_get(usage, "prompt_token_count", _get(usage, "promptTokenCount", 0)) or 0)
    cached = int(
        _get(usage, "cached_content_token_count", _get(usage, "cachedContentTokenCount", 0)) or 0
    )
    output = int(_get(usage, "candidates_token_count", _get(usage, "candidatesTokenCount", 0)) or 0)
    thoughts = int(_get(usage, "thoughts_token_count", _get(usage, "thoughtsTokenCount", 0)) or 0)
    total = int(_get(usage, "total_token_count", _get(usage, "totalTokenCount", prompt + output + thoughts)) or 0)
    return TokenUsage(prompt, cached, output, thoughts, total)


def normalize_gemini_response(
    response: Any, *, requested_model: str, labels: Sequence[str]
) -> LabelProbabilityResponse:
    candidate = _first(_get(response, "candidates", []))
    logprobs = _get(candidate, "logprobs_result", _get(candidate, "logprobsResult", None))
    top_candidates = _get(logprobs, "top_candidates", _get(logprobs, "topCandidates", [])) or []
    first_step = _first(top_candidates)
    entries = _get(first_step, "candidates", []) or []
    if not entries:
        block_reason = _get(_get(response, "prompt_feedback", {}), "block_reason", "")
        raise APIInferenceError(f"Gemini response missing top logprobs (block_reason={block_reason})")
    probs, label_lps, sanitized = normalize_label_logprobs(entries, labels)
    return LabelProbabilityResponse(
        label_probs=probs,
        label_logprobs=label_lps,
        top_tokens=sanitized,
        usage=_gemini_usage(response),
        requested_model=requested_model,
        returned_model=str(
            _get(response, "model_version", _get(response, "modelVersion", requested_model)) or requested_model
        ),
        provider="gemini",
        response_id=str(_get(response, "response_id", _get(response, "responseId", "")) or ""),
    )


class CommercialAPIClient:
    """Strict option-label probability client with resume-safe caching."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        cache_dir: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 6,
        max_cost_usd: Optional[float] = None,
        max_requests: Optional[int] = None,
        force_requests: bool = False,
        base_url: Optional[str] = None,
        input_usd_per_mtok: Optional[float] = None,
        cached_input_usd_per_mtok: Optional[float] = None,
        output_usd_per_mtok: Optional[float] = None,
        transport: Any = None,
    ) -> None:
        self.provider = str(provider).strip().lower()
        self.model = str(model).strip()
        key = f"{self.provider}:{self.model}"
        if key not in MODEL_REGISTRY:
            raise ValueError(
                f"unsupported API model '{key}'. Supported: {', '.join(sorted(MODEL_REGISTRY))}"
            )
        self.registry = dict(MODEL_REGISTRY[key])
        self.base_url = base_url if base_url is not None else self.registry.get("base_url")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(0, int(max_retries))
        self.max_cost_usd = None if max_cost_usd is None else float(max_cost_usd)
        self.max_requests = None if max_requests is None else int(max_requests)
        self.force_requests = bool(force_requests)
        self.input_rate = float(
            self.registry["input_usd_per_mtok"] if input_usd_per_mtok is None else input_usd_per_mtok
        )
        cached_default = self.registry.get("cached_input_usd_per_mtok")
        self.cached_input_rate = (
            None
            if cached_input_usd_per_mtok is None and cached_default is None
            else float(cached_default if cached_input_usd_per_mtok is None else cached_input_usd_per_mtok)
        )
        self.output_rate = float(
            self.registry["output_usd_per_mtok"] if output_usd_per_mtok is None else output_usd_per_mtok
        )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dirs_seen = {str(self.cache_dir)}
        self._transport = transport
        self._sdk_client = None
        self._lock = threading.RLock()
        self.logical_meter = _Meter()
        self.physical_meter = _Meter()
        self._network_attempts = 0
        self._returned_models: Dict[str, int] = {}
        self._context: Dict[str, Any] = {}

    @property
    def pricing(self) -> Dict[str, Any]:
        return {
            "snapshot_date": PRICE_SNAPSHOT_DATE,
            "input_usd_per_mtok": self.input_rate,
            "cached_input_usd_per_mtok": self.cached_input_rate,
            "output_usd_per_mtok": self.output_rate,
        }

    def set_context(self, **kwargs: Any) -> None:
        self._context = {str(k): v for k, v in kwargs.items()}

    def set_cache_dir(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dirs_seen.add(str(self.cache_dir))

    def _cost(self, usage: TokenUsage) -> float:
        cached = max(0, min(int(usage.cached_input_tokens), int(usage.input_tokens)))
        uncached = max(0, int(usage.input_tokens) - cached)
        cached_rate = self.input_rate if self.cached_input_rate is None else self.cached_input_rate
        billable_output = int(usage.output_tokens)
        if self.provider == "gemini":
            # Gemini reports generated candidate and internal thought tokens as
            # separate counters; both are output-priced.
            billable_output += int(usage.reasoning_tokens)
        return (
            uncached * self.input_rate
            + cached * cached_rate
            + billable_output * self.output_rate
        ) / 1_000_000.0

    def _request_payload(
        self, messages: Sequence[Mapping[str, str]], labels: Sequence[str]
    ) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "labels": [str(x) for x in labels],
            "temperature": 0.0,
            "max_output_tokens": 1,
            "top_logprobs": 20,
            "context": dict(self._context),
        }

    def _request_hash(self, payload: Mapping[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, request_hash: str) -> Path:
        return self.cache_dir / "responses" / f"{request_hash}.json"

    def _write_json_atomic(self, path: Path, obj: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp-{os.getpid()}-{threading.get_ident()}")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _append_jsonl(self, filename: str, row: Mapping[str, Any]) -> None:
        path = self.cache_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    def _load_cached(self, path: Path) -> Optional[LabelProbabilityResponse]:
        if not path.exists() or self.force_requests:
            return None
        try:
            response = LabelProbabilityResponse.from_dict(json.loads(path.read_text(encoding="utf-8")))
            response.cache_hit = True
            return response
        except Exception:
            return None

    def _reserve_network_attempt(self) -> None:
        with self._lock:
            if self.max_requests is not None and self._network_attempts >= self.max_requests:
                raise APIBudgetExceeded(
                    f"API request limit reached: {self._network_attempts}/{self.max_requests}"
                )
            if self.max_cost_usd is not None and self.physical_meter.cost_usd >= self.max_cost_usd:
                raise APIBudgetExceeded(
                    f"API cost limit reached: ${self.physical_meter.cost_usd:.6f}/${self.max_cost_usd:.6f}"
                )
            # Count every provider attempt, including retries and failed responses. This
            # reservation happens under the lock so concurrent workers cannot overrun
            # --api_max_requests.
            self._network_attempts += 1

    def _ensure_sdk_client(self) -> Any:
        if self._sdk_client is not None:
            return self._sdk_client
        env_name = str(self.registry["api_key_env"])
        api_key = os.environ.get(env_name)
        if not api_key:
            raise APIInferenceError(f"missing required environment variable {env_name}")
        if self.provider == "gemini":
            try:
                from google import genai
            except ImportError as exc:
                raise APIInferenceError("Gemini backend requires `google-genai`") from exc
            self._sdk_client = genai.Client(api_key=api_key)
        else:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise APIInferenceError("OpenAI-compatible backends require `openai`") from exc
            kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": self.timeout_seconds}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._sdk_client = OpenAI(**kwargs)
        return self._sdk_client

    def _call_provider(self, messages: Sequence[Mapping[str, str]]) -> Any:
        if self._transport is not None:
            return self._transport(
                provider=self.provider,
                model=self.model,
                messages=[dict(m) for m in messages],
            )
        client = self._ensure_sdk_client()
        if self.provider == "gemini":
            try:
                from google.genai import types
            except ImportError as exc:
                raise APIInferenceError("Gemini backend requires `google-genai`") from exc
            system_text = "\n\n".join(
                str(m.get("content", "")) for m in messages if m.get("role") == "system"
            )
            contents = [
                types.Content(
                    role="model" if m.get("role") == "assistant" else "user",
                    parts=[types.Part.from_text(text=str(m.get("content", "")))],
                )
                for m in messages
                if m.get("role") != "system"
            ]
            config_kwargs: Dict[str, Any] = {
                "temperature": 0.0,
                "max_output_tokens": 1,
                "response_logprobs": True,
                "logprobs": 20,
            }
            if system_text:
                config_kwargs["system_instruction"] = system_text
            return client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "temperature": 0.0,
            "max_tokens": 1,
        }
        if self.provider == "together":
            # Together uses an integer count instead of OpenAI's bool + count pair.
            kwargs["logprobs"] = 20
        else:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 20
        extra_body = self.registry.get("extra_body")
        if extra_body:
            kwargs["extra_body"] = dict(extra_body)
        return client.chat.completions.create(**kwargs)

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status in {408, 409, 429} or (isinstance(status, int) and status >= 500):
            return True
        name = exc.__class__.__name__.lower()
        return any(token in name for token in ("timeout", "ratelimit", "connection", "serviceunavailable"))

    def complete_labels(
        self,
        messages: Sequence[Mapping[str, str]],
        labels: Sequence[str],
    ) -> LabelProbabilityResponse:
        payload = self._request_payload(messages, labels)
        request_hash = self._request_hash(payload)
        cache_path = self._cache_path(request_hash)
        with self._lock:
            cached = self._load_cached(cache_path)
            if cached is not None:
                cached.request_hash = request_hash
                self.logical_meter.add(cached, cache_hit=True)
                self._log_call(cached)
                return cached

        started = time.perf_counter()
        retry_count = 0
        while True:
            try:
                self._reserve_network_attempt()
                raw = self._call_provider(messages)
                if self.provider == "gemini":
                    normalized = normalize_gemini_response(
                        raw, requested_model=self.model, labels=labels
                    )
                else:
                    normalized = normalize_openai_compatible_response(
                        raw,
                        provider=self.provider,
                        requested_model=self.model,
                        labels=labels,
                    )
                break
            except APIBudgetExceeded:
                raise
            except LabelCoverageError as exc:
                self._append_jsonl(
                    "diagnostics.jsonl",
                    {
                        "type": "label_coverage_error",
                        "request_hash": request_hash,
                        "provider": self.provider,
                        "model": self.model,
                        "context": dict(self._context),
                        "missing_labels": exc.missing_labels,
                        "top_tokens": exc.top_tokens,
                    },
                )
                raise
            except Exception as exc:
                if retry_count >= self.max_retries or not self._retryable(exc):
                    self._append_jsonl(
                        "diagnostics.jsonl",
                        {
                            "type": "request_error",
                            "request_hash": request_hash,
                            "provider": self.provider,
                            "model": self.model,
                            "context": dict(self._context),
                            "error_type": exc.__class__.__name__,
                            "error": str(exc),
                            "retry_count": retry_count,
                        },
                    )
                    raise APIInferenceError(f"API request failed after {retry_count} retries: {exc}") from exc
                delay = min(30.0, 0.75 * (2 ** retry_count)) + random.random() * 0.25
                retry_count += 1
                time.sleep(delay)

        normalized.latency_ms = (time.perf_counter() - started) * 1000.0
        normalized.cost_usd = self._cost(normalized.usage)
        normalized.cache_hit = False
        normalized.retry_count = retry_count
        normalized.request_hash = request_hash
        with self._lock:
            self._write_json_atomic(cache_path, normalized.to_dict())
            self.logical_meter.add(normalized, cache_hit=False)
            self.physical_meter.add(normalized, cache_hit=False)
            self._log_call(normalized)
        return normalized

    def _log_call(self, response: LabelProbabilityResponse) -> None:
        returned_model = str(response.returned_model or response.requested_model)
        self._returned_models[returned_model] = self._returned_models.get(returned_model, 0) + 1
        self._append_jsonl(
            "calls.jsonl",
            {
                "type": "api_call",
                "provider": response.provider,
                "requested_model": response.requested_model,
                "returned_model": response.returned_model,
                "response_id": response.response_id,
                "request_hash": response.request_hash,
                "context": dict(self._context),
                "usage": asdict(response.usage),
                "cost_usd": response.cost_usd,
                "cache_hit": response.cache_hit,
                "latency_ms": response.latency_ms,
                "retry_count": response.retry_count,
                "label_probs": response.label_probs,
                "top_tokens": response.top_tokens,
            },
        )

    def summary(self) -> Dict[str, Any]:
        returned_models = dict(sorted(self._returned_models.items()))
        return {
            "version": 1,
            "provider": self.provider,
            "requested_model": self.model,
            "returned_model": next(iter(returned_models)) if len(returned_models) == 1 else None,
            "returned_models": returned_models,
            "pricing": self.pricing,
            "logical": self.logical_meter.to_dict(),
            "physical": {
                **self.physical_meter.to_dict(),
                "attempted_requests": int(self._network_attempts),
            },
            "cache_dir": str(self.cache_dir),
            "cache_dirs": sorted(self._cache_dirs_seen),
        }
