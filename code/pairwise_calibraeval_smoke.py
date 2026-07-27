"""GPU smoke test for pairwise CalibraEval-style MCQ decomposition.

This first-stage experiment intentionally does not fit the disputed
CalibraEval optimizer.  It measures whether pairwise binary verification itself
is viable and caches both A/B orientations so calibration can be added offline.
"""

import argparse
import csv
import itertools
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from pairwise_aggregation import (
    accuracy,
    bradley_terry_predict,
    copeland_predict,
    count_condorcet_cycle,
    recall_std,
)


OPTION_IDS = "ABCDE"


def build_mcq_prompt(question: str, options: Sequence[str]) -> str:
    labels = OPTION_IDS[: len(options)]
    rendered = "\n".join(f"{label}. {text}" for label, text in zip(labels, options))
    return (
        "The following is a multiple choice science question. "
        "Directly answer by choosing the correct option.\n\n"
        f"Question: {question.strip()}\nOptions:\n{rendered}\nAnswer:"
    )


def build_pair_prompt(question: str, first: str, second: str) -> str:
    return (
        "The following is a science question with two candidate answers. "
        "Choose the candidate that is more likely to be correct. "
        "Directly answer with A or B.\n\n"
        f"Question: {question.strip()}\nOptions:\n"
        f"A. {first}\nB. {second}\nAnswer:"
    )


def option_token_ids(tokenizer, labels: Sequence[str]) -> List[List[int]]:
    """Token IDs for space-prefixed and unprefixed one-token option labels."""
    ids = []
    for label in labels:
        candidates = {
            tokenizer(f": {label}", add_special_tokens=False).input_ids[-1],
            tokenizer(f":{label}", add_special_tokens=False).input_ids[-1],
        }
        ids.append(sorted(candidates))
    return ids


def score_prompts(
    model,
    tokenizer,
    prompts: Sequence[str],
    labels: Sequence[str],
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    token_groups = option_token_ids(tokenizer, labels)
    flat_token_ids = [token for group in token_groups for token in group]
    group_sizes = [len(group) for group in token_groups]
    rows = []

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = pad_token_id
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            if getattr(model.config, "is_encoder_decoder", False):
                decoder_start = model.config.decoder_start_token_id
                if decoder_start is None:
                    decoder_start = model.config.pad_token_id
                decoder_ids = torch.full(
                    (len(batch), 1), decoder_start, dtype=torch.long, device=model.device
                )
                logits = model(**encoded, decoder_input_ids=decoder_ids).logits[:, -1, :]
            else:
                logits = model(**encoded).logits[:, -1, :]

        selected = logits[:, flat_token_ids]
        masses = []
        offset = 0
        for size in group_sizes:
            masses.append(torch.logsumexp(selected[:, offset : offset + size], dim=1))
            offset += size
        probs = torch.softmax(torch.stack(masses, dim=1), dim=1)
        rows.append(probs.detach().cpu().to(torch.float32).numpy())

    tokenizer.padding_side = old_padding_side
    return np.concatenate(rows, axis=0)


def load_arc(path: Path, limit: int) -> List[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) != 6:
                raise ValueError(f"expected 6 ARC columns, got {len(row)}")
            question, *options, answer = row
            rows.append(
                {
                    "question": question,
                    "options": options,
                    "label": OPTION_IDS.index(answer),
                }
            )
            if len(rows) >= limit:
                break
    return rows


def summarize(records: Sequence[dict], elapsed_seconds: float) -> dict:
    labels = [record["label"] for record in records]
    methods = {}
    for name, cost in (
        ("baseline", 1.0),
        ("pairwise_one_view_copeland", 6.0),
        ("pairwise_one_view_bt", 6.0),
        ("pairwise_swap_mean_copeland", 12.0),
        ("pairwise_swap_mean_bt", 12.0),
    ):
        predictions = [record["predictions"][name] for record in records]
        methods[name] = {
            "cost": cost,
            "accuracy": accuracy(labels, predictions),
            "recall_std": recall_std(labels, predictions, 4),
        }

    swap_diffs = [
        abs(pair["forward_p_first"] - pair["swapped_p_first"])
        for record in records
        for pair in record["pairs"]
    ]
    return {
        "n_items": len(records),
        "n_model_calls": len(records) * 13,
        "elapsed_seconds": elapsed_seconds,
        "calls_per_second": len(records) * 13 / elapsed_seconds,
        "methods": methods,
        "swap_consistency": {
            "mean_abs_difference": float(np.mean(swap_diffs)),
            "p95_abs_difference": float(np.quantile(swap_diffs, 0.95)),
            "one_view_cycle_rate": float(
                np.mean([record["cycles"]["one_view"] for record in records])
            ),
            "swap_mean_cycle_rate": float(
                np.mean([record["cycles"]["swap_mean"] for record in records])
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--data", default="data_arc/test/arc_test.csv")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = load_arc(Path(args.data), args.limit)
    config = AutoConfig.from_pretrained(args.model, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        add_bos_token=False,
        add_eos_token=False,
        cache_dir=args.cache_dir,
    )
    model_cls = (
        AutoModelForSeq2SeqLM
        if getattr(config, "is_encoder_decoder", False)
        else AutoModelForCausalLM
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = model_cls.from_pretrained(
        args.model,
        device_map="auto",
        use_safetensors=True,
        torch_dtype=dtype,
        cache_dir=args.cache_dir,
    )
    model.eval()

    prompts: List[str] = []
    layout: List[Tuple[str, int, int, int]] = []
    for item_idx, row in enumerate(rows):
        prompts.append(build_mcq_prompt(row["question"], row["options"]))
        layout.append(("baseline", item_idx, -1, -1))
        for i, j in itertools.combinations(range(4), 2):
            prompts.append(build_pair_prompt(row["question"], row["options"][i], row["options"][j]))
            layout.append(("forward", item_idx, i, j))
            prompts.append(build_pair_prompt(row["question"], row["options"][j], row["options"][i]))
            layout.append(("swapped", item_idx, i, j))

    started = time.time()
    baseline_indices = [idx for idx, entry in enumerate(layout) if entry[0] == "baseline"]
    pair_indices = [idx for idx, entry in enumerate(layout) if entry[0] != "baseline"]
    all_probs = np.zeros((len(prompts), 4), dtype=np.float32)
    all_probs[baseline_indices] = score_prompts(
        model,
        tokenizer,
        [prompts[idx] for idx in baseline_indices],
        list("ABCD"),
        args.batch_size,
        args.max_length,
    )
    pair_probs = score_prompts(
        model,
        tokenizer,
        [prompts[idx] for idx in pair_indices],
        list("AB"),
        args.batch_size,
        args.max_length,
    )
    all_probs[pair_indices, :2] = pair_probs
    elapsed = time.time() - started

    records = [
        {
            "idx": idx,
            "question": row["question"],
            "options": row["options"],
            "label": row["label"],
            "baseline_probs": None,
            "pairs": [],
        }
        for idx, row in enumerate(rows)
    ]
    pair_lookup: List[Dict[Tuple[int, int], dict]] = [dict() for _ in rows]
    for prompt_idx, (kind, item_idx, i, j) in enumerate(layout):
        if kind == "baseline":
            records[item_idx]["baseline_probs"] = all_probs[prompt_idx].tolist()
            continue
        entry = pair_lookup[item_idx].setdefault((i, j), {"i": i, "j": j})
        if kind == "forward":
            entry["forward_p_first"] = float(all_probs[prompt_idx, 0])
        else:
            entry["swapped_p_first"] = float(all_probs[prompt_idx, 1])

    for item_idx, record in enumerate(records):
        one_view = {}
        swap_mean = {}
        for pair in sorted(pair_lookup[item_idx].values(), key=lambda value: (value["i"], value["j"])):
            i, j = pair["i"], pair["j"]
            one_view[(i, j)] = pair["forward_p_first"]
            swap_mean[(i, j)] = 0.5 * (
                pair["forward_p_first"] + pair["swapped_p_first"]
            )
            record["pairs"].append(pair)
        record["predictions"] = {
            "baseline": int(np.argmax(record["baseline_probs"])),
            "pairwise_one_view_copeland": copeland_predict(one_view, 4),
            "pairwise_one_view_bt": bradley_terry_predict(one_view, 4),
            "pairwise_swap_mean_copeland": copeland_predict(swap_mean, 4),
            "pairwise_swap_mean_bt": bradley_terry_predict(swap_mean, 4),
        }
        record["cycles"] = {
            "one_view": count_condorcet_cycle(one_view, 4),
            "swap_mean": count_condorcet_cycle(swap_mean, 4),
        }

    summary = summarize(records, elapsed)
    summary.update(
        {
            "model": args.model,
            "data": str(args.data),
            "limit": args.limit,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "protocol": {
                "baseline_calls_per_item": 1,
                "pairwise_one_view_calls_per_item": 6,
                "pairwise_swap_mean_calls_per_item": 12,
                "calibraeval_optimizer_applied": False,
            },
        }
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": records}, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

