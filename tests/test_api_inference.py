import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from api_inference import (  # noqa: E402
    APIBudgetExceeded,
    APIInferenceError,
    CommercialAPIClient,
    LabelCoverageError,
    OnlinePercentileRouter,
    normalize_gemini_response,
    normalize_label_logprobs,
    normalize_openai_compatible_response,
)


def _top_entries(labels=("A", "B", "C", "D")):
    return [
        {"token": f" {label}", "logprob": -0.1 - idx}
        for idx, label in enumerate(labels)
    ]


def _openai_response(model="gpt-4.1-2025-04-14", labels=("A", "B", "C", "D")):
    return {
        "id": "resp-1",
        "model": model,
        "choices": [{"logprobs": {"content": [{"top_logprobs": _top_entries(labels)}]}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 1,
            "total_tokens": 101,
            "prompt_tokens_details": {"cached_tokens": 20},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }


class LabelNormalizationTest(unittest.TestCase):
    def test_sums_whitespace_and_case_variants(self):
        entries = _top_entries() + [{"token": "a", "logprob": -2.0}]
        probs, label_lps, sanitized = normalize_label_logprobs(entries, list("ABCD"))
        self.assertEqual(len(probs), 4)
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertGreater(probs[0], math.exp(-0.1) / sum(math.exp(-0.1 - i) for i in range(4)))
        self.assertEqual(set(label_lps), set("ABCD"))
        self.assertEqual(len(sanitized), 5)

    def test_missing_label_is_strict_error(self):
        with self.assertRaises(LabelCoverageError) as ctx:
            normalize_label_logprobs(_top_entries(("A", "B", "C")), list("ABCD"))
        self.assertEqual(ctx.exception.missing_labels, ["D"])

    def test_openai_usage_and_returned_model(self):
        out = normalize_openai_compatible_response(
            _openai_response(model="returned-snapshot"),
            provider="openai",
            requested_model="gpt-4.1-2025-04-14",
            labels=list("ABCD"),
        )
        self.assertEqual(out.returned_model, "returned-snapshot")
        self.assertEqual(out.usage.input_tokens, 100)
        self.assertEqual(out.usage.cached_input_tokens, 20)
        self.assertEqual(out.usage.output_tokens, 1)

    def test_all_openai_compatible_provider_shapes(self):
        for provider in ("openai", "deepseek", "together"):
            with self.subTest(provider=provider):
                out = normalize_openai_compatible_response(
                    _openai_response(model=f"{provider}-returned"),
                    provider=provider,
                    requested_model="requested",
                    labels=list("ABCD"),
                )
                self.assertEqual(out.provider, provider)
                self.assertAlmostEqual(sum(out.label_probs), 1.0)

    def test_together_legacy_top_logprobs_shape(self):
        response = _openai_response(model="together-returned")
        response["choices"][0]["logprobs"] = {
            "tokens": [" A"],
            "token_logprobs": [-0.1],
            "top_logprobs": [{row["token"]: row["logprob"] for row in _top_entries()}],
        }
        out = normalize_openai_compatible_response(
            response, provider="together", requested_model="requested", labels=list("ABCD")
        )
        self.assertEqual(out.returned_model, "together-returned")
        self.assertAlmostEqual(sum(out.label_probs), 1.0)

    def test_gemini_response_shape(self):
        response = {
            "responseId": "gem-1",
            "modelVersion": "gemini-version",
            "candidates": [{
                "logprobsResult": {
                    "topCandidates": [{
                        "candidates": [
                            {"token": row["token"], "logProbability": row["logprob"]}
                            for row in _top_entries()
                        ]
                    }]
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 90,
                "cachedContentTokenCount": 10,
                "candidatesTokenCount": 1,
                "thoughtsTokenCount": 0,
                "totalTokenCount": 91,
            },
        }
        out = normalize_gemini_response(
            response, requested_model="gemini-2.5-flash-lite", labels=list("ABCD")
        )
        self.assertEqual(out.returned_model, "gemini-version")
        self.assertEqual(out.usage.input_tokens, 90)
        self.assertEqual(out.usage.cached_input_tokens, 10)


class ClientBehaviorTest(unittest.TestCase):
    def test_cache_separates_logical_and_physical_usage(self):
        calls = []

        def transport(**kwargs):
            calls.append(kwargs)
            return _openai_response()

        with tempfile.TemporaryDirectory() as td:
            client = CommercialAPIClient(
                provider="openai",
                model="gpt-4.1-2025-04-14",
                cache_dir=td,
                max_requests=5,
                transport=transport,
            )
            messages = [{"role": "user", "content": "Answer: A/B/C/D"}]
            first = client.complete_labels(messages, list("ABCD"))
            second = client.complete_labels(messages, list("ABCD"))
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(len(calls), 1)
            summary = client.summary()
            self.assertEqual(summary["physical"]["requests"], 1)
            self.assertEqual(summary["logical"]["requests"], 2)
            self.assertEqual(summary["logical"]["cache_hits"], 1)
            self.assertGreater(summary["physical"]["cost_usd"], 0.0)
            self.assertEqual(summary["physical"]["attempted_requests"], 1)
            self.assertAlmostEqual(first.cost_usd, 178.0 / 1_000_000.0)

    def test_missing_label_writes_diagnostic_and_fails_without_runtime_retry(self):
        with tempfile.TemporaryDirectory() as td:
            client = CommercialAPIClient(
                provider="openai",
                model="gpt-4.1-2025-04-14",
                cache_dir=td,
                max_requests=5,
                transport=lambda **_: _openai_response(labels=("A", "B", "C")),
            )
            with self.assertRaises(LabelCoverageError):
                client.complete_labels([{"role": "user", "content": "q"}], list("ABCD"))
            diagnostic = Path(td, "diagnostics.jsonl").read_text(encoding="utf-8")
            self.assertIn('"type": "label_coverage_error"', diagnostic)
            self.assertNotIn(RuntimeError, LabelCoverageError.__mro__)

    def test_empty_logprobs_is_request_error(self):
        response = _openai_response()
        response["choices"][0]["logprobs"]["content"] = []
        with self.assertRaises(APIInferenceError):
            normalize_openai_compatible_response(
                response, provider="openai", requested_model="requested", labels=list("ABCD")
            )

    def test_request_budget_stops_before_second_network_call(self):
        with tempfile.TemporaryDirectory() as td:
            client = CommercialAPIClient(
                provider="openai",
                model="gpt-4.1-2025-04-14",
                cache_dir=td,
                max_requests=1,
                transport=lambda **_: _openai_response(),
            )
            client.complete_labels([{"role": "user", "content": "q1"}], list("ABCD"))
            with self.assertRaises(APIBudgetExceeded):
                client.complete_labels([{"role": "user", "content": "q2"}], list("ABCD"))

    def test_retries_transient_error(self):
        class RateLimited(Exception):
            status_code = 429

        attempts = {"n": 0}

        def transport(**_):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RateLimited("slow down")
            return _openai_response()

        with tempfile.TemporaryDirectory() as td, mock.patch("api_inference.time.sleep"):
            client = CommercialAPIClient(
                provider="openai",
                model="gpt-4.1-2025-04-14",
                cache_dir=td,
                max_requests=5,
                max_retries=2,
                transport=transport,
            )
            out = client.complete_labels([{"role": "user", "content": "q"}], list("ABCD"))
            self.assertEqual(attempts["n"], 2)
            self.assertEqual(out.retry_count, 1)
            self.assertEqual(client.summary()["physical"]["attempted_requests"], 2)


class AdaptiveRouterTest(unittest.TestCase):
    def test_forced_prefix_and_confidence_gated_request_count(self):
        router = OnlinePercentileRouter(k=4, percentile=80, schedule="flat")

        # A calibration-prefix example is forced through all four stages.
        forced_confidences = [0.90, 0.80, 0.75, 0.72]
        router.observe(forced_confidences)
        stages = [4]

        # A high-confidence example stops after its identity request.
        high = [0.95]
        self.assertTrue(router.should_stop(1, high[-1]))
        router.observe(high)
        stages.append(1)

        # A low-confidence identity needs one more request, then clears stage 2.
        low = [0.20, 0.90]
        self.assertFalse(router.should_stop(1, low[0]))
        self.assertTrue(router.should_stop(2, low[1]))
        router.observe(low)
        stages.append(2)

        self.assertEqual(sum(stages), 7)  # one API request per acquired stage
        self.assertAlmostEqual(sum(stages) / len(stages), 7.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
