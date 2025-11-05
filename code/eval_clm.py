
import os
import sys
import json
import logging
from eval_clm_utils import (
    parse_arguments,
    prepare_eval,
)
from utils import (
    _orange, _blue, _purple,
    eval_all_samples,
    get_accuracy,
    get_bootstrap_accuracy_std,
    save_results,
    patch_open,
)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import numpy as np
from types import SimpleNamespace

import gc

import pynvml
pynvml.nvmlInit()

logger = logging.getLogger(__name__)


def logging_cuda_memory_usage():
    logger.info("******** Memory usage ********")
    n_gpus = pynvml.nvmlDeviceGetCount()
    for i in range(n_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.info("GPU {}: {:.2f} GB / {:.2f} GB".format(i, meminfo.used / 1024 ** 3, meminfo.total / 1024 ** 3))


def main():
    patch_open()

    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    args = parse_arguments()
    if len(args.eval_names) == 0:
        exit()

    toker = AutoTokenizer.from_pretrained(
        args.pretrained_model_path,
        use_fast=False,
        add_bos_token=False,
        add_eos_token=False,
        cache_dir=args.cache_dir,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_path,
        device_map='auto',
        use_safetensors=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        cache_dir=args.cache_dir,
    )
    logging_cuda_memory_usage()

    printed_example = False

    # Comparative mode: run with two option ID sets and compute matching ratio and flip stats
    if getattr(args, 'option_id_sets', None) and len(args.option_id_sets) == 2:
        id_set_a, id_set_b = args.option_id_sets[0], args.option_id_sets[1]

        overall_total = 0
        overall_matches = 0
        overall_c2i = 0
        overall_i2c = 0
        overall_both_correct = 0
        overall_both_incorrect = 0

        def _read_results_file(file_path):
            try:
                lines = [json.loads(line) for line in open(file_path)]
                lines = [e for e in lines if e.get('type') == 'result']
                lines = sorted(lines, key=lambda x: int(x['data']['idx']))
                return lines
            except FileNotFoundError:
                return None

        for eval_name in args.eval_names[::1]:
            eval_args = eval_name.split(',')
            setting = eval_args[2] if len(eval_args) > 2 else None
            if setting in ['perm', 'cyclic']:
                logger.info(_orange(f"Skipping compare for setting '{setting}' (not supported)."))
                continue
            args_a = SimpleNamespace(pretrained_model_path=args.pretrained_model_path, model_name=args.model_name, option_id_set=id_set_a)
            args_b = SimpleNamespace(pretrained_model_path=args.pretrained_model_path, model_name=args.model_name, option_id_set=id_set_b)

            (subjects_a, prepare_fewshot_a, prepare_eval_samples_a, prepare_eval_fn_a) = prepare_eval(args_a, eval_name)
            (subjects_b, prepare_fewshot_b, prepare_eval_samples_b, prepare_eval_fn_b) = prepare_eval(args_b, eval_name)

            assert subjects_a == subjects_b, "Subjects mismatch across option ID sets"

            for subject in subjects_a[::1]:
                logger.info(_blue(f"Preparing (compare): {subject}"))
                few_a = prepare_fewshot_a(subject)
                few_b = prepare_fewshot_b(subject)
                eval_samples_a = prepare_eval_samples_a(subject)
                eval_samples_b = prepare_eval_samples_b(subject)
                eval_fn_a = prepare_eval_fn_a(model, toker, few_a)
                eval_fn_b = prepare_eval_fn_b(model, toker, few_b)

                # Prompt example (only once, from set A)
                if getattr(args, 'print_prompt_example', False) and not printed_example and len(eval_samples_a) > 0:
                    try:
                        first_input, _first_options, _first_ideal = eval_samples_a[0]
                        def build_input_text(pair):
                            sys_msg, eval_sample = pair
                            text = sys_msg + '\n\n'
                            if args_a.num_few_shot > 0:
                                for s in few_a[:args_a.num_few_shot]:
                                    text += s + '\n\n'
                            text += eval_sample
                            return text
                        if isinstance(first_input, list) and len(first_input) > 0 and isinstance(first_input[0], list):
                            input_text = build_input_text(first_input[0])
                            bpe_has_space_prefix = toker(': A').input_ids[-1] != toker(':A').input_ids[-1]
                            if not bpe_has_space_prefix:
                                input_text += ' '
                        else:
                            input_text = build_input_text(first_input)
                            bpe_has_space_prefix = toker(': A').input_ids[-1] != toker(':A').input_ids[-1]
                            if not bpe_has_space_prefix:
                                input_text += ' '
                        logger.info(_purple("==== Prompt example ===="))
                        logger.info("\n" + input_text)
                        logger.info(_purple("==== End prompt example ===="))
                        printed_example = True
                    except Exception as e:
                        logger.warning(f"Failed to build prompt example: {e}")

                # Try cached results first
                path_a = f'{args_a.save_path}/{subject}.jsonl'
                path_b = f'{args_b.save_path}/{subject}.jsonl'
                results_a = _read_results_file(path_a)
                results_b = _read_results_file(path_b)

                if results_a is not None:
                    logger.info(_blue(f"Using cached results (A): {path_a}"))
                else:
                    logger.info(_blue(f"Run started (A): {subject} [{id_set_a}]"))
                    results_a = eval_all_samples(
                        eval_fn_a, eval_samples_a,
                        name=f'{args_a.task},{args_a.num_few_shot},{args_a.setting},{subject},{id_set_a}',
                        threads=torch.cuda.device_count() if 'falcon' not in args.pretrained_model_path else 1,
                    )
                    save_results(path_a, results_a, metrics=None)
                    logger.info(f"Results saved (A): {subject}")

                if results_b is not None:
                    logger.info(_blue(f"Using cached results (B): {path_b}"))
                else:
                    logger.info(_blue(f"Run started (B): {subject} [{id_set_b}]"))
                    results_b = eval_all_samples(
                        eval_fn_b, eval_samples_b,
                        name=f'{args_b.task},{args_b.num_few_shot},{args_b.setting},{subject},{id_set_b}',
                        threads=torch.cuda.device_count() if 'falcon' not in args.pretrained_model_path else 1,
                    )
                    save_results(path_b, results_b, metrics=None)
                    logger.info(f"Results saved (B): {subject}")

                # Align by idx
                map_a = {int(r['data']['idx']): r for r in results_a if r['type'] == 'result'}
                map_b = {int(r['data']['idx']): r for r in results_b if r['type'] == 'result'}
                common = sorted(set(map_a.keys()) & set(map_b.keys()))

                matches = 0
                total = 0
                c2i = 0
                i2c = 0
                both_correct = 0
                both_incorrect = 0

                for idx in common:
                    ra = map_a[idx]['data']
                    rb = map_b[idx]['data']
                    # Predicted option indices via argmax over probs
                    pa = int(np.argmax(np.array(ra['probs'])))
                    pb = int(np.argmax(np.array(rb['probs'])))
                    if pa == pb:
                        matches += 1
                    ca = bool(ra['correct'])
                    cb = bool(rb['correct'])
                    if ca and not cb:
                        c2i += 1
                    elif (not ca) and cb:
                        i2c += 1
                    elif ca and cb:
                        both_correct += 1
                    elif (not ca) and (not cb):
                        both_incorrect += 1
                    total += 1

                overall_total += total
                overall_matches += matches
                overall_c2i += c2i
                overall_i2c += i2c
                overall_both_correct += both_correct
                overall_both_incorrect += both_incorrect

                acc_a = get_accuracy(results_a)
                acc_b = get_accuracy(results_b)
                mr = (matches / total) if total > 0 else float('nan')
                logger.info(_purple(f"[{subject}] matching_ratio={mr:.4f} (matches/total={matches}/{total})"))
                logger.info(f"accuracy_A({id_set_a})={acc_a:.4f}, accuracy_B({id_set_b})={acc_b:.4f}")
                logger.info(f"flip correct->incorrect={c2i}, incorrect->correct={i2c}, both_correct={both_correct}, both_incorrect={both_incorrect}")

                gc.collect()
                torch.cuda.empty_cache()

        # Overall summary
        if overall_total > 0:
            overall_mr = overall_matches / overall_total
            logger.info(_purple("==== Overall compare summary ====\n" +
                                f"matching_ratio={overall_mr:.4f} (matches/total={overall_matches}/{overall_total})\n" +
                                f"correct->incorrect={overall_c2i}, incorrect->correct={overall_i2c}, both_correct={overall_both_correct}, both_incorrect={overall_both_incorrect}"))
        return

    # Single-run mode (original)
    for eval_name in args.eval_names[::1]:
        (
            subjects, prepare_few_shot_samples, prepare_eval_samples, prepare_eval_fn
        ) = prepare_eval(args, eval_name)
        for subject in subjects[::1]:
            if os.path.exists(f'{args.save_path}/{subject}.jsonl'):
                logger.info(f"Results already exist: {args.save_path}/{subject}.jsonl")
                continue

            logger.info(_blue(f"Preparing: {subject}"))
            few_shot_samples = prepare_few_shot_samples(subject)
            eval_samples = prepare_eval_samples(subject)
            eval_fn = prepare_eval_fn(model, toker, few_shot_samples)

            # Print one prompt example if requested
            if getattr(args, 'print_prompt_example', False) and not printed_example and len(eval_samples) > 0:
                try:
                    first_input, _first_options, _first_ideal = eval_samples[0]
                    # Build input_text similarly to eval fns for accuracy of display
                    def build_input_text(pair):
                        sys_msg, eval_sample = pair
                        text = sys_msg + '\n\n'
                        if args.num_few_shot > 0:
                            for s in few_shot_samples[:args.num_few_shot]:
                                text += s + '\n\n'
                        text += eval_sample
                        return text

                    if isinstance(first_input, list) and len(first_input) > 0 and isinstance(first_input[0], list):
                        # perm/cyclic case: list of [sys_msg, prompt] pairs; show the first
                        input_text = build_input_text(first_input[0])
                        # Match model-space behavior: add trailing space if tokenizer lacks space-prefix
                        bpe_has_space_prefix = toker(': A').input_ids[-1] != toker(':A').input_ids[-1]
                        if not bpe_has_space_prefix:
                            input_text += ' '
                    else:
                        # base/noid case
                        input_text = build_input_text(first_input)
                        if args.setting not in ['noid']:
                            bpe_has_space_prefix = toker(': A').input_ids[-1] != toker(':A').input_ids[-1]
                            if not bpe_has_space_prefix:
                                input_text += ' '

                    logger.info(_purple("==== Prompt example ===="))
                    logger.info("\n" + input_text)
                    logger.info(_purple("==== End prompt example ===="))
                    printed_example = True
                except Exception as e:
                    logger.warning(f"Failed to build prompt example: {e}")

            logger.info(_blue(f"Run started: {subject}"))
            results = eval_all_samples(
                eval_fn, eval_samples,
                name=f'{args.task},{args.num_few_shot},{args.setting},{subject}',
                threads=torch.cuda.device_count() if 'falcon' not in args.pretrained_model_path else 1,
            )
            gc.collect()
            torch.cuda.empty_cache()

            metrics = None
            if args.setting not in ['perm', 'cyclic'] and len(results) > 0:
                metrics = {'type': 'metric', 'data': {}}
                metrics['data']['accuracy'] = get_accuracy(results)
                metrics['data']['boostrap_std'] = get_bootstrap_accuracy_std(results)
                logger.info("Final report:")
                for key, value in metrics['data'].items():
                    logger.info(f"{key}: {value}")
            logger.info(_orange(f"Run completed: {subject}"))

            save_results(f'{args.save_path}/{subject}.jsonl', results, metrics)
            logger.info(f"Results saved: {subject}")

            logging_cuda_memory_usage()


if __name__ == "__main__":
    main()

