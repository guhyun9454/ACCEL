import os
import json
import argparse
import logging
from types import SimpleNamespace
from datetime import datetime

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from eval_clm_utils import prepare_eval
from utils import eval_all_samples
from debias_utils import simple as debias_simple


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--task", type=str, default="arc", choices=["arc", "mmlu"],
                        help="evaluation dataset; uses 4 choices")
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--ratio_prefix_samples", type=float, default=0.05)
    parser.add_argument("--option_id_sets", type=str, nargs='+',
                        default=["ABCD", "abcd", "1234"],
                        help="token ID sets to probe (e.g., ABCD abcd 1234). Length must match #options")
    parser.add_argument("--cache_dir", type=str, default="models",
                        help="Hugging Face cache directory where models/tokenizers are stored")
    return parser.parse_args()


def main():
    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    args = parse_args()
    model_name = args.pretrained_model_path.split("/")[-1]

    # Resolve and set cache directory for this process
    cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    logger.info(f"Using HF cache dir: {cache_dir}")

    # Load model/tokenizer once
    toker = AutoTokenizer.from_pretrained(
        args.pretrained_model_path,
        use_fast=False,
        add_bos_token=False,
        add_eos_token=False,
        cache_dir=cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_path,
        device_map='auto',
        use_safetensors=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        cache_dir=cache_dir,
    )

    save_dir = f"results_{args.task}/{args.num_few_shot}s_{model_name}/{args.task}_token_pref"
    os.makedirs(save_dir, exist_ok=True)

    for option_id_set in args.option_id_sets:
        logger.info(f"Probing token preference with option IDs: {option_id_set}")

        # Build a lightweight args object expected by prepare_eval
        pe_args = SimpleNamespace()
        pe_args.pretrained_model_path = args.pretrained_model_path
        pe_args.model_name = model_name
        pe_args.option_id_set = option_id_set

        eval_name = f"{args.task},{args.num_few_shot},cyclic"
        subjects, prepare_few_shot_samples, prepare_eval_samples, prepare_eval_fn = prepare_eval(pe_args, eval_name)

        # Compute total number of samples (across subjects)
        total_num = 0
        per_subject_counts = {}
        for subject in subjects:
            cnt = len(prepare_eval_samples(subject))
            per_subject_counts[subject] = cnt
            total_num += cnt

        target_prefix = max(1, int(total_num * args.ratio_prefix_samples))
        logger.info(f"Total samples={total_num}, using prefix subset={target_prefix}")

        # Gather prefix subset results across subjects until target_prefix is reached
        gathered = 0
        priors = []
        for subject in subjects:
            if gathered >= target_prefix:
                break
            few_shot_samples = prepare_few_shot_samples(subject)
            eval_samples = prepare_eval_samples(subject)
            eval_fn = prepare_eval_fn(model, toker, few_shot_samples)

            remaining = min(per_subject_counts[subject], target_prefix - gathered)
            results = eval_all_samples(
                eval_fn, eval_samples,
                name=f'{args.task},{args.num_few_shot},cyclic,{subject},{option_id_set}',
                max_num_samples=remaining,
                threads=torch.cuda.device_count() if torch.cuda.is_available() else 1,
            )

            for r in results:
                if r['type'] != 'result':
                    continue
                observed = np.array(r['data']['probs'])  # shape: (#cycles, 4)
                _, _, prior = debias_simple(observed)
                priors.append(prior)
                gathered += 1
                if gathered >= target_prefix:
                    break

        if gathered == 0:
            logger.info("No results gathered; skipping.")
            continue

        prior_mean = np.mean(np.array(priors), axis=0).tolist()

        # Build labels for option_ids from option_id_set
        option_ids = list(option_id_set)
        result = {
            'model': model_name,
            'pretrained_model_path': args.pretrained_model_path,
            'task': args.task,
            'setting': 'cyclic',
            'num_few_shot': args.num_few_shot,
            'ratio_prefix_samples': args.ratio_prefix_samples,
            'num_samples_used': gathered,
            'option_id_set': option_id_set,
            'option_ids': option_ids,
            'prior': prior_mean,
            'prior_map': {k: float(v) for k, v in zip(option_ids, prior_mean)},
            'note': 'PRIDE-style prior estimated via cyclic permutations',
            'saved_at': datetime.utcnow().isoformat() + 'Z',
        }

        save_path = os.path.join(save_dir, f"{option_id_set}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved token preference report: {save_path}")


if __name__ == "__main__":
    main()


