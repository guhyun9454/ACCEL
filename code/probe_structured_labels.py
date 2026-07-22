"""Probe GPT-4.1 label probabilities under strict JSON-schema decoding.

This is an experimental constrained protocol.  It deliberately does not feed
results into the normal PriDe/ACCEL cache or main result files.
"""

import argparse
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from eval_clm_utils import _api_messages, prepare_eval


def _schema(labels):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "mcq_label",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "enum": list(labels)},
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_choice(choice, labels):
    labels = [str(x) for x in labels]
    content = str(_get(_get(choice, "message", {}), "content", "") or "")
    try:
        answer = str(json.loads(content)["answer"])
    except Exception:
        answer = ""
    token_rows = list(_get(_get(choice, "logprobs", {}), "content", []) or [])
    decision = None
    for index, row in enumerate(token_rows):
        if str(_get(row, "token", "")) == answer and answer in labels:
            decision = (index, row)
            break
    result = {
        "content": content,
        "answer": answer,
        "finish_reason": str(_get(choice, "finish_reason", "") or ""),
        "decision_token_index": None,
        "decision_token": None,
        "decision_logprob": None,
        "coverage": False,
        "missing_labels": list(labels),
        "label_logprobs": {},
        "label_probs": {},
        "top_tokens": [],
        "top_k_mass": 0.0,
        "tail_mass": 1.0,
    }
    if decision is None:
        return result
    index, row = decision
    entries = []
    label_logprobs = {}
    for item in list(_get(row, "top_logprobs", []) or []):
        token = str(_get(item, "token", ""))
        logprob = float(_get(item, "logprob", float("-inf")))
        probability = math.exp(logprob) if math.isfinite(logprob) else 0.0
        entries.append({"token": token, "logprob": logprob, "probability": probability})
        if token in labels:
            label_logprobs[token] = logprob
    missing = [label for label in labels if label not in label_logprobs]
    masses = {label: math.exp(label_logprobs[label]) for label in labels if label in label_logprobs}
    denominator = sum(masses.values())
    label_probs = {
        label: masses[label] / denominator for label in labels if label in masses and denominator > 0.0
    }
    top_k_mass = sum(item["probability"] for item in entries)
    result.update({
        "decision_token_index": index,
        "decision_token": str(_get(row, "token", "")),
        "decision_logprob": float(_get(row, "logprob", float("-inf"))),
        "coverage": not missing,
        "missing_labels": missing,
        "label_logprobs": label_logprobs,
        "label_probs": label_probs,
        "top_tokens": entries,
        "top_k_mass": top_k_mass,
        "tail_mass": max(0.0, 1.0 - top_k_mass),
    })
    return result


def _usage(response):
    usage = _get(response, "usage", {}) or {}
    details = _get(usage, "completion_tokens_details", {}) or {}
    return {
        "input_tokens": int(_get(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(_get(usage, "completion_tokens", 0) or 0),
        "cached_input_tokens": int(_get(_get(usage, "prompt_tokens_details", {}) or {}, "cached_tokens", 0) or 0),
        "reasoning_tokens": int(_get(details, "reasoning_tokens", 0) or 0),
        "total_tokens": int(_get(usage, "total_tokens", 0) or 0),
    }


def _cost(usage, input_price, cached_price, output_price):
    uncached = max(0, usage["input_tokens"] - usage["cached_input_tokens"])
    return (
        uncached * input_price
        + usage["cached_input_tokens"] * cached_price
        + usage["output_tokens"] * output_price
    ) / 1_000_000.0


def _prepare_dataset(args, option_ids):
    prep_args = SimpleNamespace(
        inference_backend="api",
        api_execution_mode="offline_sweep",
        api_prompt_mode="label_only",
        model_name=args.model.split("/")[-1],
        option_id_set="".join(option_ids),
        result_tag="structured_label_probe",
        skip_full=True,
    )
    subjects, _, prepare_samples, _ = prepare_eval(prep_args, f"{args.task},0,full")
    selected = list(subjects) if args.all_subjects else [str(args.subject or subjects[0])]
    return selected, prepare_samples


def _iter_prompts(args, option_ids):
    subjects, prepare_samples = _prepare_dataset(args, option_ids)
    for subject in subjects:
        samples = prepare_samples(subject)
        if args.probe_samples is None:
            sample_indices = [args.sample_index]
        else:
            sample_indices = range(min(int(args.probe_samples), len(samples)))
        for sample_index in sample_indices:
            inputs, _, ideal = samples[sample_index]
            permutations = (
                inputs
                if isinstance(inputs, list) and inputs and isinstance(inputs[0], list)
                else [inputs]
            )
            permutation_indices = (
                range(len(permutations)) if args.all_permutations else [args.permutation_index]
            )
            for permutation_index in permutation_indices:
                sys_msg, user_prompt = permutations[permutation_index]
                yield {
                    "subject": str(subject),
                    "sample_index": int(sample_index),
                    "permutation_index": int(permutation_index),
                    "ideal": str(ideal),
                    "messages": _api_messages(
                        sys_msg,
                        user_prompt,
                        [],
                        0,
                        option_ids,
                        api_prompt_mode="label_only",
                    ),
                }


def _softmax(values):
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / weights.sum()


def _reconstruct_pairwise(rows, labels):
    """Fit Bradley--Terry scores to fully observed two-label probabilities."""
    labels = [str(label) for label in labels]
    label_to_index = {label: index for index, label in enumerate(labels)}
    design, targets = [], []
    for row in rows:
        allowed = list(row["allowed_labels"])
        if len(allowed) != 2 or len(row["choices"]) != 1:
            raise ValueError("pairwise reconstruction requires one choice for every label pair")
        choice = row["choices"][0]
        if not choice["coverage"]:
            return {
                "coverage": False,
                "missing_pair": allowed,
                "missing_labels": list(choice["missing_labels"]),
            }
        left, right = allowed
        probs = choice["label_probs"]
        left_p, right_p = float(probs[left]), float(probs[right])
        if left_p <= 0.0 or right_p <= 0.0:
            return {
                "coverage": False,
                "missing_pair": allowed,
                "missing_labels": [label for label in allowed if float(probs[label]) <= 0.0],
            }
        vector = np.zeros(len(labels), dtype=float)
        vector[label_to_index[left]] = 1.0
        vector[label_to_index[right]] = -1.0
        design.append(vector)
        targets.append(math.log(left_p) - math.log(right_p))
    if len(design) != len(labels) * (len(labels) - 1) // 2:
        raise ValueError("pairwise reconstruction requires the complete label-pair graph")
    # Sum-to-zero identifies the otherwise shift-invariant score vector.
    matrix = np.vstack([np.asarray(design), np.ones(len(labels), dtype=float)])
    target = np.asarray([*targets, 0.0], dtype=float)
    scores, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    fitted = np.asarray(design) @ scores
    residuals = fitted - np.asarray(targets)
    probabilities = _softmax(scores)
    return {
        "coverage": True,
        "scores": {label: float(scores[index]) for index, label in enumerate(labels)},
        "probabilities": {
            label: float(probabilities[index]) for index, label in enumerate(labels)
        },
        "argmax": labels[int(np.argmax(probabilities))],
        "log_odds_rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "log_odds_max_abs_error": float(np.max(np.abs(residuals))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1-2025-04-14")
    parser.add_argument("--task", choices=["arc", "mmlu", "csqa"], required=True)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--permutation_index", type=int, default=0)
    parser.add_argument("--probe_samples", type=int, default=None)
    parser.add_argument("--all_permutations", action="store_true")
    parser.add_argument("--all_subjects", action="store_true")
    parser.add_argument(
        "--protocol",
        choices=["multiway", "pairwise", "fallback", "monte_carlo"],
        default="multiway",
    )
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--allowed_labels",
        default=None,
        help="Optional schema-label subset for monte_carlo, for example AB.",
    )
    parser.add_argument("--bias_label", default=None)
    parser.add_argument("--bias_value", type=float, default=0.0)
    parser.add_argument("--max_requests", type=int, default=10)
    parser.add_argument("--max_cost_usd", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.n < 1 or args.n > 128:
        raise ValueError("--n must be in [1, 128]")

    labels = list("ABCDE" if args.task == "csqa" else "ABCD")
    monte_carlo_labels = list(args.allowed_labels) if args.allowed_labels else list(labels)
    if args.protocol != "monte_carlo" and args.allowed_labels:
        raise ValueError("--allowed_labels is supported only by monte_carlo")
    if not monte_carlo_labels or any(label not in labels for label in monte_carlo_labels):
        raise ValueError("--allowed_labels must be a non-empty subset of task labels")
    if args.bias_label is not None and args.bias_label not in monte_carlo_labels:
        raise ValueError("--bias_label must be included in the schema labels")
    logit_bias = None
    if args.bias_label is not None:
        import tiktoken

        token_ids = tiktoken.encoding_for_model(args.model).encode(args.bias_label)
        if len(token_ids) != 1:
            raise ValueError("--bias_label must encode as one canonical token")
        logit_bias = {str(int(token_ids[0])): float(args.bias_value)}
    from openai import OpenAI

    client = OpenAI()
    rows = []
    sample_results = []
    label_pairs = list(itertools.combinations(labels, 2))
    total_cost = 0.0
    total_usage = {key: 0 for key in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens", "total_tokens")}
    strict_pass = True
    stopped_early = False
    for prompt in _iter_prompts(args, labels):
        prompt_start = len(rows)
        counts = {label: 0 for label in labels}
        if args.protocol == "pairwise":
            allowed_groups = list(label_pairs)
        elif args.protocol == "monte_carlo":
            allowed_groups = [tuple(monte_carlo_labels)]
        else:
            allowed_groups = [tuple(labels)]
        for allowed in allowed_groups:
            if len(rows) >= args.max_requests:
                raise RuntimeError("protocol exceeds --max_requests")
            n = args.n if args.protocol == "monte_carlo" else 1
            request_kwargs = dict(
                model=args.model,
                messages=prompt["messages"],
                temperature=args.temperature,
                max_tokens=16,
                response_format=_schema(allowed),
                logprobs=True,
                top_logprobs=20,
                n=n,
            )
            if logit_bias is not None:
                request_kwargs["logit_bias"] = logit_bias
            response = client.chat.completions.create(**request_kwargs)
            usage = _usage(response)
            cost = _cost(usage, 2.0, 0.5, 8.0)
            total_cost += cost
            for key in total_usage:
                total_usage[key] += usage[key]
            choices = [_extract_choice(choice, allowed) for choice in response.choices]
            row = {
                "subject": prompt["subject"],
                "sample_index": prompt["sample_index"],
                "permutation_index": prompt["permutation_index"],
                "allowed_labels": list(allowed),
                "request_id": str(_get(response, "id", "") or ""),
                "requested_model": args.model,
                "returned_model": str(_get(response, "model", "") or ""),
                "usage": usage,
                "estimated_usd": cost,
                "choices": choices,
            }
            rows.append(row)
            if total_cost > args.max_cost_usd:
                raise RuntimeError(f"cost cap exceeded after request: {total_cost:.6f}")
            if args.protocol == "monte_carlo":
                for choice in choices:
                    if choice["answer"] in counts:
                        counts[choice["answer"]] += 1
            elif any(not choice["coverage"] for choice in choices):
                if args.protocol == "fallback" and len(allowed) > 2:
                    # The failed paid multiway response remains in the artifact;
                    # reconstruct this prompt only from a complete pair graph.
                    allowed_groups.extend(label_pairs)
                else:
                    strict_pass = False
                    stopped_early = True
                    break
        prompt_rows = rows[prompt_start:]
        result = {
            key: prompt[key]
            for key in ("subject", "sample_index", "permutation_index", "ideal")
        }
        result["request_row_indices"] = list(range(prompt_start, len(rows)))
        if args.protocol == "pairwise" and len(prompt_rows) == len(label_pairs):
            result["pairwise_reconstruction"] = _reconstruct_pairwise(prompt_rows, labels)
            strict_pass = strict_pass and bool(result["pairwise_reconstruction"]["coverage"])
        elif args.protocol == "fallback":
            multiway = prompt_rows[0]["choices"][0]
            if multiway["coverage"]:
                result["scoring_path"] = "structured_multiway"
                result["probabilities"] = dict(multiway["label_probs"])
                result["argmax"] = max(multiway["label_probs"], key=multiway["label_probs"].get)
            elif len(prompt_rows) == 1 + len(label_pairs):
                reconstruction = _reconstruct_pairwise(prompt_rows[1:], labels)
                result["scoring_path"] = "structured_pairwise_fallback"
                result["multiway_missing_labels"] = list(multiway["missing_labels"])
                result["pairwise_reconstruction"] = reconstruction
                result["probabilities"] = dict(reconstruction.get("probabilities", {}))
                result["argmax"] = reconstruction.get("argmax")
                strict_pass = strict_pass and bool(reconstruction["coverage"])
            else:
                result["scoring_path"] = "failed_pairwise_fallback"
                strict_pass = False
        elif args.protocol == "monte_carlo":
            total = sum(counts.values())
            result["sample_counts"] = counts
            result["sample_probabilities"] = {
                label: (counts[label] / total if total else 0.0) for label in labels
            }
            result["valid_samples"] = total
            strict_pass = strict_pass and total == args.n
        else:
            result["coverage"] = bool(prompt_rows and prompt_rows[0]["choices"][0]["coverage"])
            strict_pass = strict_pass and result["coverage"]
        sample_results.append(result)
        if stopped_early:
            break

    payload = {
        "protocol": f"structured_{args.protocol}_v1",
        "model": args.model,
        "task": args.task,
        "subject": args.subject,
        "sample_index": args.sample_index if args.probe_samples is None else None,
        "permutation_index": args.permutation_index if not args.all_permutations else None,
        "option_ids": labels,
        "allowed_labels": monte_carlo_labels if args.protocol == "monte_carlo" else None,
        "bias_label": args.bias_label,
        "bias_value": float(args.bias_value) if args.bias_label is not None else None,
        "temperature": args.temperature,
        "n": args.n if args.protocol == "monte_carlo" else 1,
        "rows": rows,
        "sample_results": sample_results,
        "strict_pass": bool(strict_pass),
        "stopped_early": bool(stopped_early),
        "physical": {"requests": len(rows), "usage": total_usage, "cost_usd": total_cost},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "strict_pass": payload["strict_pass"],
        "stopped_early": payload["stopped_early"],
        "sample_results": len(payload["sample_results"]),
        "physical": payload["physical"],
    }, ensure_ascii=False))
    if not payload["strict_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
