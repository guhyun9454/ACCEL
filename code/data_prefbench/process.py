"""Convert Prometheus-2's Preference-Bench into the headerless CSV the loader expects.

`prometheus-eval/Preference-Bench` is the relative-grading test set for
Prometheus 2: an instruction plus two responses, with `orig_preference` naming
the better one. Another k=2 task, so it shares `pairwise_data_utils` and the
`prefbench` branch in `eval_clm_utils.prepare_eval`.

`orig_preference` already labels a slot ("A" or "B"), and that labelling is not
balanced in the source. We therefore resolve it to chosen/rejected here and let
`pairwise_data_utils.to_row` re-randomize the slot, so gold ends up ~50/50 and a
model that always answers "A" scores chance.

Usage:
    python process.py
    python process.py --limit 1200
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pairwise_data_utils import write_dataset  # noqa: E402

HF_DATASET = "prometheus-eval/Preference-Bench"


def to_triples(dataset):
    """(query, chosen, rejected) per item; items without a clear preference are dropped."""
    triples, skipped = [], 0
    for sample in dataset:
        query = sample.get("orig_instruction", "")
        response_a = sample.get("orig_response_A", "")
        response_b = sample.get("orig_response_B", "")
        preference = str(sample.get("orig_preference", "")).strip().upper()
        if preference == "A":
            triples.append((query, response_a, response_b))
        elif preference == "B":
            triples.append((query, response_b, response_a))
        else:
            skipped += 1
    return triples, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_few_shot_pool", type=int, default=5)
    args = parser.parse_args()

    from datasets import load_dataset

    data = load_dataset(HF_DATASET, split=args.split)
    triples, skipped = to_triples(data)
    print(f"items: {len(triples)} (no clear preference, skipped: {skipped})")

    write_dataset("prefbench", triples, os.path.dirname(os.path.abspath(__file__)),
                  limit=args.limit, seed=args.seed,
                  num_few_shot_pool=args.num_few_shot_pool)


if __name__ == "__main__":
    main()
