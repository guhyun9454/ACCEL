"""One-vs-rest (OvR) binary verification sweep on ARC.

Implements the cost-4 protocol from docs/worklog/260727-guhyun-calibraeval-ovr-idea.md:
for every item, each option's TEXT (not its letter, to avoid re-introducing
option-ID bias) is presented as a proposed answer and the model answers
Yes/No; the prediction is argmax over the four Yes log-odds. A standard 4-way
MCQ prompt is scored as the cost-1 baseline. Per-item Yes/No probabilities are
cached in the output JSON so CalibraEval-style calibration can be fitted
offline without re-running inference.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from pairwise_aggregation import accuracy, recall_std
from pairwise_calibraeval_smoke import build_mcq_prompt, load_arc, score_prompts


def build_ovr_prompt(question: str, candidate: str) -> str:
    return (
        "The following is a science question with a proposed answer. "
        "Decide whether the proposed answer is correct. "
        "Directly answer with Yes or No.\n\n"
        f"Question: {question.strip()}\nProposed answer: {candidate.strip()}\nAnswer:"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data_arc/test/arc_test.csv")
    parser.add_argument("--limit", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"[ovr] loading ARC rows: data={args.data} limit={args.limit}", flush=True)
    rows = load_arc(Path(args.data), args.limit)
    print(f"[ovr] {len(rows)} items", flush=True)

    config = AutoConfig.from_pretrained(args.model, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, add_bos_token=False, add_eos_token=False, cache_dir=args.cache_dir
    )
    model_cls = (
        AutoModelForSeq2SeqLM
        if getattr(config, "is_encoder_decoder", False)
        else AutoModelForCausalLM
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[ovr] loading model weights: {args.model} dtype={dtype}", flush=True)
    model = model_cls.from_pretrained(
        args.model,
        device_map="auto",
        use_safetensors=True,
        torch_dtype=dtype,
        cache_dir=args.cache_dir,
    )
    model.eval()
    print(f"[ovr] model ready on {model.device}", flush=True)

    mcq_prompts: List[str] = []
    ovr_prompts: List[str] = []
    ovr_layout: List[Tuple[int, int]] = []
    for item_idx, row in enumerate(rows):
        mcq_prompts.append(build_mcq_prompt(row["question"], row["options"]))
        for opt_idx, option in enumerate(row["options"]):
            ovr_prompts.append(build_ovr_prompt(row["question"], option))
            ovr_layout.append((item_idx, opt_idx))

    started = time.time()
    print(
        f"[ovr] scoring {len(mcq_prompts)} baseline and {len(ovr_prompts)} OvR prompts",
        flush=True,
    )
    baseline_probs = score_prompts(
        model, tokenizer, mcq_prompts, list("ABCD"), args.batch_size, args.max_length
    )
    yes_no = score_prompts(
        model, tokenizer, ovr_prompts, ["Yes", "No"], args.batch_size, args.max_length
    )
    elapsed = time.time() - started
    print(f"[ovr] scoring complete in {elapsed:.1f}s", flush=True)

    p_yes = np.zeros((len(rows), 4), dtype=np.float64)
    for (item_idx, opt_idx), probs in zip(ovr_layout, yes_no):
        p_yes[item_idx, opt_idx] = float(probs[0])

    labels = [row["label"] for row in rows]
    baseline_preds = [int(np.argmax(p)) for p in baseline_probs]
    ovr_preds = [int(np.argmax(p)) for p in p_yes]

    records = [
        {
            "idx": idx,
            "label": row["label"],
            "baseline_probs": [float(x) for x in baseline_probs[idx]],
            "ovr_p_yes": [float(x) for x in p_yes[idx]],
            "baseline_pred": baseline_preds[idx],
            "ovr_pred": ovr_preds[idx],
        }
        for idx, row in enumerate(rows)
    ]
    summary = {
        "model": args.model,
        "data": args.data,
        "n_items": len(rows),
        "n_model_calls": len(rows) * 5,
        "elapsed_seconds": elapsed,
        "methods": {
            "baseline": {
                "cost": 1.0,
                "accuracy": accuracy(labels, baseline_preds),
                "recall_std": recall_std(labels, baseline_preds, 4),
            },
            "ovr_yes_argmax": {
                "cost": 4.0,
                "accuracy": accuracy(labels, ovr_preds),
                "recall_std": recall_std(labels, ovr_preds, 4),
            },
        },
        "ovr_agreement_with_baseline": float(
            np.mean([b == o for b, o in zip(baseline_preds, ovr_preds)])
        ),
        "mean_yes_mass": float(np.mean(np.sum(p_yes, axis=1))),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": records}, handle)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[ovr] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
