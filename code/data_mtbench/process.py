"""Convert MT-Bench human judgments into the headerless CSV the loader expects.

`lmsys/mt_bench_human_judgments` holds 3.3K expert pairwise preferences over
MT-Bench answers. Like RewardBench it is a k=2 task, so it reuses
`pairwise_data_utils` and the `mtbench` branch in `eval_clm_utils.prepare_eval`.

Two dataset-specific decisions:

* **Turn 1 only.** For `turn == 2` the two conversations share the first user
  message but each model produced its *own* first answer, so the context leading
  into the compared answer differs between A and B. There is no single "Question"
  text that fairly serves both, and the loader has exactly one question column.
* **Majority vote across judges, ties dropped.** Each row is one human judgment,
  so the same (question, model_a, model_b) pair appears several times and judges
  disagree. Keeping the rows as-is would put contradictory gold labels on
  identical items. `winner == "tie"` carries no preference and is discarded.

Usage:
    python process.py                 # human split, turn 1
    python process.py --split gpt4_pair
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pairwise_data_utils import majority_vote, write_dataset  # noqa: E402

HF_DATASET = "lmsys/mt_bench_human_judgments"


def last_assistant(conversation):
    """Final assistant message — the answer being judged."""
    for message in reversed(conversation or []):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def first_user(conversation):
    for message in conversation or []:
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def to_triples(dataset, turn=1):
    """Aggregate judgments into (query, chosen, rejected) per unique pair."""
    grouped = {}
    for sample in dataset:
        if int(sample.get("turn", 1)) != turn:
            continue
        winner = str(sample.get("winner", ""))
        if winner not in ("model_a", "model_b"):      # drop ties
            continue
        key = (sample.get("question_id"), sample.get("model_a"), sample.get("model_b"))
        entry = grouped.setdefault(key, {"votes": [], "sample": sample})
        entry["votes"].append(winner)

    triples, no_majority = [], 0
    for entry in grouped.values():
        winner = majority_vote(entry["votes"])
        if winner is None:
            no_majority += 1
            continue
        sample = entry["sample"]
        answer_a = last_assistant(sample.get("conversation_a"))
        answer_b = last_assistant(sample.get("conversation_b"))
        query = first_user(sample.get("conversation_a"))
        if winner == "model_a":
            triples.append((query, answer_a, answer_b))
        else:
            triples.append((query, answer_b, answer_a))
    return triples, len(grouped), no_majority


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="human", choices=["human", "gpt4_pair"])
    parser.add_argument("--turn", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_few_shot_pool", type=int, default=5)
    args = parser.parse_args()

    from datasets import load_dataset

    data = load_dataset(HF_DATASET, split=args.split)
    triples, n_pairs, no_majority = to_triples(data, turn=args.turn)
    print(f"turn-{args.turn} pairs: {n_pairs}, no clear majority: {no_majority}, "
          f"usable: {len(triples)}")

    write_dataset("mtbench", triples, os.path.dirname(os.path.abspath(__file__)),
                  limit=args.limit, seed=args.seed,
                  num_few_shot_pool=args.num_few_shot_pool)


if __name__ == "__main__":
    main()
