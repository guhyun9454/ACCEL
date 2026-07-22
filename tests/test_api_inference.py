import math
import inspect
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from api_inference import (  # noqa: E402
    APIBudgetExceeded,
    APIInferenceError,
    CommercialAPIClient,
    LabelCoverageError,
    OnlinePercentileRouter,
    TokenUsage,
    normalize_gemini_response,
    normalize_label_logprobs,
    normalize_openai_compatible_response,
)
from eval_clm_utils import (  # noqa: E402
    _api_label_only_instruction,
    _api_messages,
    prepare_eval,
    prepare_eval_fn_api_base,
    prepare_eval_fn_api_perm,
    prepare_eval_fn_base,
    select_api_probe_subjects,
)
from probe_equal_label_bias import _parse_biases, _total_variation  # noqa: E402


def _top_entries(labels=("A", "B", "C", "D")):
    return [
        {"token": f" {label}", "logprob": -0.1 - idx}
        for idx, label in enumerate(labels)
    ]


def _openai_response(model="gpt-4.1-2025-04-14", labels=("A", "B", "C", "D")):
    return {
        "id": "resp-1",
        "model": model,
        "choices": [{"logprobs": {"content": [{
            "token": f" {labels[0]}",
            "top_logprobs": _top_entries(labels),
        }]}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 1,
            "total_tokens": 101,
            "prompt_tokens_details": {"cached_tokens": 20},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }


class _RecordingClient:
    def __init__(self):
        self.messages = []

    def complete_labels(self, messages, labels):
        self.messages.append([dict(row) for row in messages])
        probs = [1.0 / len(labels)] * len(labels)
        return SimpleNamespace(label_probs=probs, to_dict=lambda: {"label_probs": probs})


class APIPromptModeTest(unittest.TestCase):
    def test_baseline_api_prompt_preserves_original_system_text(self):
        system = "The following are multiple choice questions. You should directly answer the question by choosing the correct option."
        messages = _api_messages(system, "Question: q\nAnswer:", [], 0, list("ABCD"))
        self.assertEqual(messages[0], {"role": "system", "content": system})
        self.assertNotIn("Respond with exactly one option label", messages[0]["content"])

    def test_label_only_uses_exact_abcd_instruction(self):
        instruction = _api_label_only_instruction(list("ABCD"))
        self.assertEqual(
            instruction,
            'Respond with exactly one option label from: A, B, C, D. '
            'Do not output the word "Answer", an explanation, punctuation, or any other text.',
        )

    def test_label_only_uses_csqa_abcde(self):
        self.assertIn(
            "from: A, B, C, D, E.",
            _api_label_only_instruction(list("ABCDE")),
        )

    def test_label_only_supports_custom_option_ids(self):
        self.assertIn(
            "from: 1, 2, 3, 4.",
            _api_label_only_instruction(list("1234")),
        )

    def test_api_base_result_records_prompt_mode(self):
        client = _RecordingClient()
        fn = prepare_eval_fn_api_base(
            client, None, [], 0, list("ABCD"), api_prompt_mode="label_only"
        )
        result = fn(
            (0, (["Original system", "Question: q\nAnswer:"], ["a", "b", "c", "d"], "A")),
            random.Random(0),
        )
        self.assertEqual(result["data"]["api_prompt_mode"], "label_only")
        self.assertIn("Respond with exactly one option label", client.messages[0][0]["content"])

    def test_offline_and_adaptive_prepare_paths_forward_prompt_mode(self):
        for execution_mode in ("offline_sweep", "adaptive"):
            with self.subTest(execution_mode=execution_mode), mock.patch(
                "eval_clm_utils.os.makedirs"
            ), mock.patch(
                "eval_clm_utils.os.listdir", return_value=["sample_test.csv"]
            ):
                args = SimpleNamespace(
                    inference_backend="api",
                    api_execution_mode=execution_mode,
                    api_prompt_mode="label_only",
                    model_name="model",
                    option_id_set=None,
                    result_tag=None,
                    skip_full=True,
                )
                _, _, _, prep_fn = prepare_eval(args, "arc,0,full")
                self.assertIs(prep_fn.func, prepare_eval_fn_api_perm)
                self.assertEqual(prep_fn.keywords["api_prompt_mode"], "label_only")

    def test_local_prompt_and_label_restricted_scoring_are_unchanged(self):
        source = inspect.getsource(prepare_eval_fn_base)
        self.assertIn("input_text = sys_msg + '\\n\\n'", source)
        self.assertIn('toker(f": {e}", add_special_tokens=False)', source)
        self.assertIn('toker(f":{e}", add_special_tokens=False)', source)
        self.assertNotIn("api_prompt_mode", inspect.signature(prepare_eval_fn_base).parameters)
        self.assertNotIn("label_only", source)


class APIProbeSubjectSelectionTest(unittest.TestCase):
    def test_probe_defaults_to_first_subject(self):
        args = SimpleNamespace(api_probe_only=True, api_probe_all_subjects=False)
        self.assertEqual(select_api_probe_subjects(args, ["a", "b", "c"]), ["a"])

    def test_stratified_probe_keeps_all_subjects(self):
        args = SimpleNamespace(api_probe_only=True, api_probe_all_subjects=True)
        self.assertEqual(select_api_probe_subjects(args, ["a", "b", "c"]), ["a", "b", "c"])

    def test_non_probe_keeps_all_subjects(self):
        args = SimpleNamespace(api_probe_only=False, api_probe_all_subjects=False)
        self.assertEqual(select_api_probe_subjects(args, ["a", "b", "c"]), ["a", "b", "c"])


class EqualLabelBiasProbeTest(unittest.TestCase):
    def test_bias_parser_and_tv(self):
        self.assertEqual(_parse_biases("0,20,100"), [0.0, 20.0, 100.0])
        self.assertAlmostEqual(_total_variation([0.7, 0.3], [0.6, 0.4]), 0.1)

    def test_bias_parser_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            _parse_biases("101")


class StructuredLabelProbeTest(unittest.TestCase):
    def test_extracts_exact_structured_label_candidates(self):
        from probe_structured_labels import _extract_choice

        row = SimpleNamespace(
            token="C",
            logprob=-0.2,
            top_logprobs=[
                SimpleNamespace(token="A", logprob=-2.0),
                SimpleNamespace(token="B", logprob=-1.0),
                SimpleNamespace(token="C", logprob=-0.2),
                SimpleNamespace(token="D", logprob=-3.0),
            ],
        )
        choice = SimpleNamespace(
            message=SimpleNamespace(content='{"answer":"C"}'),
            logprobs=SimpleNamespace(content=[row]),
            finish_reason="stop",
        )
        result = _extract_choice(choice, list("ABCD"))
        self.assertTrue(result["coverage"])
        self.assertEqual(result["decision_token"], "C")
        self.assertAlmostEqual(sum(result["label_probs"].values()), 1.0)

    def test_structured_probe_keeps_missing_label_strict(self):
        from probe_structured_labels import _extract_choice

        row = SimpleNamespace(
            token="C",
            logprob=0.0,
            top_logprobs=[SimpleNamespace(token="C", logprob=0.0)],
        )
        choice = SimpleNamespace(
            message=SimpleNamespace(content='{"answer":"C"}'),
            logprobs=SimpleNamespace(content=[row]),
            finish_reason="stop",
        )
        result = _extract_choice(choice, list("ABCD"))
        self.assertFalse(result["coverage"])
        self.assertEqual(result["missing_labels"], ["A", "B", "D"])

    def test_pairwise_reconstruction_recovers_consistent_scores(self):
        from probe_structured_labels import _reconstruct_pairwise

        labels = list("ABC")
        scores = {"A": 1.0, "B": 0.0, "C": -1.0}
        rows = []
        for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
            left_mass = math.exp(scores[left])
            right_mass = math.exp(scores[right])
            denominator = left_mass + right_mass
            rows.append({
                "allowed_labels": [left, right],
                "choices": [{
                    "coverage": True,
                    "missing_labels": [],
                    "label_probs": {
                        left: left_mass / denominator,
                        right: right_mass / denominator,
                    },
                }],
            })
        result = _reconstruct_pairwise(rows, labels)
        self.assertTrue(result["coverage"])
        self.assertEqual(result["argmax"], "A")
        self.assertAlmostEqual(result["log_odds_rmse"], 0.0, places=12)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)

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

    def test_exact_label_mode_rejects_whitespace_variant(self):
        with self.assertRaises(LabelCoverageError) as ctx:
            normalize_label_logprobs(
                _top_entries(), list("ABCD"), exact_label_tokens=True
            )
        self.assertEqual(ctx.exception.missing_labels, list("ABCD"))

    def test_common_bias_cancels_after_exact_label_normalization(self):
        raw = [
            {"token": label, "logprob": logprob}
            for label, logprob in zip("ABCD", (-0.2, -1.7, -3.4, -8.0))
        ]
        shifted = [
            {"token": row["token"], "logprob": row["logprob"] + 40.0}
            for row in raw
        ]
        raw_probs, _, _ = normalize_label_logprobs(
            raw, list("ABCD"), exact_label_tokens=True
        )
        shifted_probs, _, _ = normalize_label_logprobs(
            shifted, list("ABCD"), exact_label_tokens=True
        )
        for before, after in zip(raw_probs, shifted_probs):
            self.assertAlmostEqual(before, after)

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

        vertex = normalize_gemini_response(
            response,
            requested_model="gemini-2.5-flash",
            labels=list("ABCD"),
            provider="vertex",
        )
        self.assertEqual(vertex.provider, "vertex")


class ClientBehaviorTest(unittest.TestCase):
    @staticmethod
    def _test_token_encoder(label):
        return [32 + "ABCDE".index(label)]

    def test_default_scoring_preserves_legacy_cache_payload(self):
        with tempfile.TemporaryDirectory() as td:
            client = CommercialAPIClient(
                provider="openai",
                model="gpt-4.1-2025-04-14",
                cache_dir=td,
                max_requests=1,
            )
            payload = client._request_payload(
                [{"role": "user", "content": "q"}], list("ABCD")
            )
            self.assertNotIn("scoring_mode", payload)
            self.assertNotIn("logit_bias", payload)

    def test_equal_label_bias_payload_uses_exact_single_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            client = CommercialAPIClient(
                provider="openai",
                model="gpt-4.1-2025-04-14",
                cache_dir=td,
                max_requests=1,
                scoring_mode="equal_label_bias",
                equal_label_bias=40.0,
                token_encoder=self._test_token_encoder,
            )
            payload = client._request_payload(
                [{"role": "user", "content": "q"}], list("ABCD")
            )
            self.assertEqual(payload["scoring_mode"], "equal_label_bias")
            self.assertEqual(payload["equal_label_bias"], 40.0)
            self.assertEqual(payload["logit_bias"], {
                "32": 40.0, "33": 40.0, "34": 40.0, "35": 40.0,
            })

    def test_equal_label_bias_rejects_multi_token_custom_label(self):
        with tempfile.TemporaryDirectory() as td:
            client = CommercialAPIClient(
                provider="openai",
                model="gpt-4.1-2025-04-14",
                cache_dir=td,
                max_requests=1,
                scoring_mode="equal_label_bias",
                token_encoder=lambda label: [1, 2] if label == "XY" else [1],
            )
            with self.assertRaises(APIInferenceError):
                client._request_payload(
                    [{"role": "user", "content": "q"}], ["A", "XY"]
                )

    def test_luna_uses_live_endpoint_limits_and_current_pricing(self):
        with tempfile.TemporaryDirectory() as td:
            client = CommercialAPIClient(
                provider="openai",
                model="gpt-5.6-luna",
                cache_dir=td,
                max_requests=1,
                transport=lambda **_: _openai_response(model="gpt-5.6-luna"),
            )
            payload = client._request_payload(
                [{"role": "user", "content": "Answer: A/B/C/D"}], list("ABCD")
            )
            self.assertEqual(payload["top_logprobs"], 5)
            self.assertEqual(payload["max_output_tokens"], 8)
            self.assertIsNone(payload["temperature"])
            self.assertEqual(client.pricing["snapshot_date"], "2026-07-21")
            self.assertEqual(client.pricing["input_usd_per_mtok"], 1.0)
            self.assertEqual(client.pricing["cached_input_usd_per_mtok"], 0.1)
            self.assertEqual(client.pricing["output_usd_per_mtok"], 6.0)

    def test_vertex_registry_uses_adc_and_gemini_output_metering(self):
        with tempfile.TemporaryDirectory() as td:
            client = CommercialAPIClient(
                provider="vertex",
                model="gemini-2.5-flash",
                cache_dir=td,
                max_requests=1,
            )
            self.assertIsNone(client.registry["api_key_env"])
            self.assertEqual(client.registry["thinking_budget"], 0)
            self.assertEqual(client.pricing["input_usd_per_mtok"], 0.30)
            self.assertEqual(client.pricing["cached_input_usd_per_mtok"], 0.075)
            self.assertEqual(client.pricing["output_usd_per_mtok"], 2.50)
            usage = TokenUsage(input_tokens=100, cached_input_tokens=20,
                               output_tokens=1, reasoning_tokens=2)
            self.assertAlmostEqual(client._cost(usage), 33.0 / 1_000_000.0)

    def test_luna_chat_kwargs_disable_reasoning_and_request_top5(self):
        class Completions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return _openai_response(model="gpt-5.6-luna")

        completions = Completions()
        sdk = type("SDK", (), {
            "chat": type("Chat", (), {"completions": completions})()
        })()
        with tempfile.TemporaryDirectory() as td:
            client = CommercialAPIClient(
                provider="openai",
                model="gpt-5.6-luna",
                cache_dir=td,
                max_requests=1,
            )
            client._sdk_client = sdk
            client._call_provider(
                [{"role": "user", "content": "Answer: A/B/C/D"}], list("ABCD")
            )

        self.assertEqual(completions.kwargs["top_logprobs"], 5)
        self.assertEqual(completions.kwargs["max_completion_tokens"], 8)
        self.assertEqual(completions.kwargs["reasoning_effort"], "none")
        self.assertNotIn("temperature", completions.kwargs)
        self.assertNotIn("max_tokens", completions.kwargs)

    def test_equal_label_bias_is_sent_to_openai_chat_completions(self):
        class Completions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return _openai_response()

        completions = Completions()
        sdk = type("SDK", (), {
            "chat": type("Chat", (), {"completions": completions})()
        })()
        with tempfile.TemporaryDirectory() as td:
            client = CommercialAPIClient(
                provider="openai",
                model="gpt-4.1-2025-04-14",
                cache_dir=td,
                max_requests=1,
                scoring_mode="equal_label_bias",
                equal_label_bias=80.0,
                token_encoder=self._test_token_encoder,
            )
            client._sdk_client = sdk
            client._call_provider(
                [{"role": "user", "content": "Answer: A/B/C/D"}], list("ABCD")
            )

        self.assertEqual(completions.kwargs["logit_bias"], {
            "32": 80.0, "33": 80.0, "34": 80.0, "35": 80.0,
        })

    def test_equal_label_bias_client_scores_only_exact_tokens(self):
        response = _openai_response()
        first = response["choices"][0]["logprobs"]["content"][0]
        first["token"] = "A"
        first["top_logprobs"] = [
            {"token": label, "logprob": -0.1 - idx}
            for idx, label in enumerate("ABCD")
        ] + [{"token": " A", "logprob": -0.01}]
        with tempfile.TemporaryDirectory() as td:
            client = CommercialAPIClient(
                provider="openai",
                model="gpt-4.1-2025-04-14",
                cache_dir=td,
                max_requests=1,
                scoring_mode="equal_label_bias",
                equal_label_bias=100.0,
                token_encoder=self._test_token_encoder,
                transport=lambda **_: response,
            )
            client.set_context(
                prompt_mode="label_only",
                scoring_mode="equal_label_bias",
                equal_label_bias=100.0,
            )
            out = client.complete_labels(
                [{"role": "user", "content": "q"}], list("ABCD")
            )
            expected_a = math.exp(-0.1) / sum(math.exp(-0.1 - idx) for idx in range(4))
            self.assertAlmostEqual(out.label_probs[0], expected_a)
            call = json.loads(Path(td, "calls.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(call["scoring_mode"], "equal_label_bias")
            self.assertEqual(call["equal_label_bias"], 100.0)

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
            client.set_context(
                task="arc",
                subject="ARC-Challenge-Test",
                run_idx=0,
                execution_mode="offline_sweep",
                prompt_mode="label_only",
            )
            with self.assertRaises(LabelCoverageError):
                client.complete_labels([{"role": "user", "content": "q"}], list("ABCD"))
            diagnostic = json.loads(
                Path(td, "diagnostics.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(diagnostic["type"], "label_coverage_error")
            self.assertEqual(diagnostic["missing_labels"], ["D"])
            self.assertEqual(diagnostic["prompt_mode"], "label_only")
            self.assertEqual(diagnostic["task"], "arc")
            self.assertEqual(diagnostic["requested_model"], "gpt-4.1-2025-04-14")
            self.assertEqual(diagnostic["returned_model"], "gpt-4.1-2025-04-14")
            self.assertEqual(diagnostic["response_id"], "resp-1")
            self.assertEqual(diagnostic["first_token"], " A")
            self.assertTrue(all("probability" in row for row in diagnostic["top_tokens"]))
            self.assertGreater(diagnostic["top_k_mass"], 0.0)
            self.assertGreaterEqual(diagnostic["tail_mass"], 0.0)
            self.assertEqual(diagnostic["input_tokens"], 100)
            self.assertEqual(diagnostic["cached_input_tokens"], 20)
            self.assertEqual(diagnostic["output_tokens"], 1)
            self.assertEqual(diagnostic["reasoning_tokens"], 0)
            self.assertGreater(diagnostic["cost_usd"], 0.0)
            self.assertGreaterEqual(diagnostic["latency_ms"], 0.0)
            self.assertEqual(diagnostic["retry_count"], 0)
            self.assertEqual(diagnostic["physical_after_failure"]["requests"], 1)
            summary = client.summary()
            self.assertEqual(summary["physical"]["requests"], 1)
            self.assertEqual(summary["physical"]["usage"]["input_tokens"], 100)
            self.assertGreater(summary["physical"]["cost_usd"], 0.0)
            self.assertEqual(list(Path(td, "responses").glob("*.json")), [])
            self.assertNotIn(RuntimeError, LabelCoverageError.__mro__)

    def test_success_call_artifact_records_prompt_context_and_probability_mass(self):
        with tempfile.TemporaryDirectory() as td:
            client = CommercialAPIClient(
                provider="openai",
                model="gpt-4.1-2025-04-14",
                cache_dir=td,
                max_requests=1,
                transport=lambda **_: _openai_response(),
            )
            client.set_context(prompt_mode="label_only", execution_mode="adaptive")
            out = client.complete_labels([{"role": "user", "content": "q"}], list("ABCD"))
            self.assertEqual(out.context["prompt_mode"], "label_only")
            self.assertGreater(out.top_k_mass, 0.0)
            self.assertGreaterEqual(out.tail_mass, 0.0)
            call = json.loads(
                Path(td, "calls.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(call["context"]["prompt_mode"], "label_only")
            self.assertEqual(call["context"]["execution_mode"], "adaptive")
            self.assertIn("top_k_mass", call)
            self.assertIn("tail_mass", call)

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
