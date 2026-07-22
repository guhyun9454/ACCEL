"""Run a reproducible single-example equal-label logit-bias capability sweep."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from api_inference import CommercialAPIClient, LabelCoverageError
from eval_clm_utils import _api_messages, prepare_eval


def _parse_biases(value):
    biases = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not biases or any(bias < 0.0 or bias > 100.0 for bias in biases):
        raise ValueError("bias values must be a non-empty comma-separated list in [0, 100]")
    return biases


def _total_variation(left, right):
    return 0.5 * sum(abs(float(a) - float(b)) for a, b in zip(left, right))


def _usage_dict(response):
    return {
        "input_tokens": int(response.usage.input_tokens),
        "cached_input_tokens": int(response.usage.cached_input_tokens),
        "output_tokens": int(response.usage.output_tokens),
        "reasoning_tokens": int(response.usage.reasoning_tokens),
        "total_tokens": int(response.usage.total_tokens),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1-2025-04-14")
    parser.add_argument("--task", choices=["arc", "mmlu", "csqa"], required=True)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--permutation_index", type=int, default=0)
    parser.add_argument("--biases", default="0,20,40,80,100")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_requests", type=int, default=5)
    parser.add_argument("--max_cost_usd", type=float, default=0.05)
    parser.add_argument("--force_requests", action="store_true")
    args = parser.parse_args()

    biases = _parse_biases(args.biases)
    if len(biases) > int(args.max_requests):
        raise ValueError("number of bias values exceeds --max_requests")
    if args.max_cost_usd <= 0.0:
        raise ValueError("--max_cost_usd must be positive")

    option_ids = list("ABCDE" if args.task == "csqa" else "ABCD")
    prep_args = SimpleNamespace(
        inference_backend="api",
        api_execution_mode="offline_sweep",
        api_prompt_mode="label_only",
        model_name=args.model.split("/")[-1],
        option_id_set="".join(option_ids),
        result_tag="equal_label_bias_single_probe",
        skip_full=True,
    )
    subjects, _, prepare_samples, _ = prepare_eval(prep_args, f"{args.task},0,full")
    subject = str(args.subject or subjects[0])
    if subject not in subjects:
        raise ValueError(f"unknown subject {subject!r} for task {args.task!r}")
    samples = prepare_samples(subject)
    if not 0 <= args.sample_index < len(samples):
        raise IndexError(f"sample_index {args.sample_index} outside [0, {len(samples)})")
    inputs, _, ideal = samples[args.sample_index]
    if not isinstance(inputs, list) or not inputs or not isinstance(inputs[0], list):
        permutations = [inputs]
    else:
        permutations = inputs
    if not 0 <= args.permutation_index < len(permutations):
        raise IndexError(
            f"permutation_index {args.permutation_index} outside [0, {len(permutations)})"
        )
    sys_msg, user_prompt = permutations[args.permutation_index]
    messages = _api_messages(
        sys_msg, user_prompt, [], 0, option_ids, api_prompt_mode="label_only"
    )

    rows = []
    total_cost = 0.0
    total_physical_requests = 0
    total_usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
    cache_root = Path(args.cache_dir)
    for bias in biases:
        remaining = float(args.max_cost_usd) - total_cost
        if remaining <= 0.0:
            raise RuntimeError("aggregate cost limit reached before completing bias sweep")
        constrained = bias > 0.0
        mode = "equal_label_bias" if constrained else "topk_strict"
        namespace = "unbiased" if not constrained else f"bias_{bias:g}"
        client = CommercialAPIClient(
            provider="openai",
            model=args.model,
            cache_dir=str(cache_root / namespace),
            max_requests=1,
            max_cost_usd=remaining,
            force_requests=bool(args.force_requests),
            scoring_mode=mode,
            equal_label_bias=bias if constrained else 100.0,
        )
        client.set_context(
            task=args.task,
            subject=subject,
            run_idx=0,
            sample_index=int(args.sample_index),
            permutation_index=int(args.permutation_index),
            execution_mode="single_bias_sweep",
            prompt_mode="label_only",
            scoring_mode=mode,
            equal_label_bias=bias if constrained else None,
        )
        row = {
            "bias": bias,
            "scoring_mode": mode,
            "normalization": "exact_canonical" if constrained else "variant_aggregated",
            "coverage": False,
        }
        try:
            response = client.complete_labels(messages, option_ids)
            row.update({
                "coverage": True,
                "first_token": response.first_token,
                "label_probs": dict(zip(option_ids, response.label_probs)),
                "argmax": option_ids[max(range(len(option_ids)), key=response.label_probs.__getitem__)],
                "top_k_mass": response.top_k_mass,
                "tail_mass": response.tail_mass,
                "top_tokens": response.top_tokens,
                "usage": _usage_dict(response),
                "cost_usd": response.cost_usd,
                "requested_model": response.requested_model,
                "returned_model": response.returned_model,
                "response_id": response.response_id,
                "request_hash": response.request_hash,
                "cache_hit": response.cache_hit,
            })
        except LabelCoverageError as exc:
            row.update({
                "missing_labels": list(exc.missing_labels),
                "top_tokens": list(exc.top_entries),
            })
        summary = client.summary()
        physical = dict(summary.get("physical") or {})
        total_physical_requests += int(physical.get("requests", 0) or 0)
        total_cost += float(physical.get("cost_usd", 0.0) or 0.0)
        usage = dict(physical.get("usage") or {})
        for key in total_usage:
            total_usage[key] += int(usage.get(key, 0) or 0)
        row["physical"] = physical
        rows.append(row)

    successful = [row for row in rows if row.get("coverage") and row["bias"] > 0.0]
    tv = []
    for left, right in zip(successful, successful[1:]):
        tv.append({
            "left_bias": left["bias"],
            "right_bias": right["bias"],
            "total_variation": _total_variation(
                list(left["label_probs"].values()), list(right["label_probs"].values())
            ),
        })
    payload = {
        "protocol": "equal_label_bias_constrained_v1",
        "model": args.model,
        "task": args.task,
        "subject": subject,
        "sample_index": int(args.sample_index),
        "permutation_index": int(args.permutation_index),
        "option_ids": option_ids,
        "ideal": str(ideal),
        "bias_results": rows,
        "pairwise_constrained_tv": tv,
        "physical": {
            "requests": total_physical_requests,
            "usage": total_usage,
            "cost_usd": total_cost,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
