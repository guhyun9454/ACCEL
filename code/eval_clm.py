
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
from itertools import permutations
import matplotlib.pyplot as plt
import math

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


def _rotations(k: int):
    return [tuple((i + s) % k for i in range(k)) for s in range(k)]


def _aggregate_probs_over_permutations(probs_seq, permuted_indices, k: int):
    """
    Map letter-indexed probs from each permutation to content-indexed probs,
    then average across permutations.
    probs_seq: list of length (#perms), each a length-k list of probs for letters
    permuted_indices: list of tuples, permutation p: letter j corresponds to content index p[j]
    """
    agg = np.zeros(k, dtype=np.float64)
    for perm_idx, p in enumerate(permuted_indices):
        letter_probs = np.asarray(probs_seq[perm_idx], dtype=np.float64)
        for j in range(k):
            agg[p[j]] += letter_probs[j]
    if len(permuted_indices) > 0:
        agg /= float(len(permuted_indices))
    return agg


def main():
    patch_open()

    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    args = parse_arguments()
    if len(args.eval_names) == 0:
        exit()

    # Optional: W&B init
    wandb_run = None
    if getattr(args, 'wandb', False):
        try:
            import wandb
            project = args.wandb_project 
            run_name = args.wandb_run_name or f"{args.model_name}-{args.eval_names[0]}"
            wandb_run = wandb.init(entity = "capde", project=project, name=run_name, config={
                "pretrained_model_path": args.pretrained_model_path,
                "model_name": args.model_name,
                "eval_names": args.eval_names,
                "option_id_set": args.option_id_set,
            })
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")
            wandb_run = None

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
                    max_samples = 100 if getattr(args, 'test', False) else None
                    results_a = eval_all_samples(
                        eval_fn_a, eval_samples_a,
                        name=f'{args_a.task},{args_a.num_few_shot},{args_a.setting},{subject},{id_set_a}',
                        threads=torch.cuda.device_count() if 'falcon' not in args.pretrained_model_path else 1,
                        max_num_samples=max_samples,
                    )
                    save_results(path_a, results_a, metrics=None)
                    logger.info(f"Results saved (A): {subject}")

                if results_b is not None:
                    logger.info(_blue(f"Using cached results (B): {path_b}"))
                else:
                    logger.info(_blue(f"Run started (B): {subject} [{id_set_b}]"))
                    max_samples = 100 if getattr(args, 'test', False) else None
                    results_b = eval_all_samples(
                        eval_fn_b, eval_samples_b,
                        name=f'{args_b.task},{args_b.num_few_shot},{args_b.setting},{subject},{id_set_b}',
                        threads=torch.cuda.device_count() if 'falcon' not in args.pretrained_model_path else 1,
                        max_num_samples=max_samples,
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
            max_samples = 100 if getattr(args, 'test', False) else None
            results = eval_all_samples(
                eval_fn, eval_samples,
                name=f'{args.task},{args.num_few_shot},{args.setting},{subject}',
                threads=torch.cuda.device_count() if 'falcon' not in args.pretrained_model_path else 1,
                max_num_samples=max_samples,
            )
            gc.collect()
            torch.cuda.empty_cache()

            metrics = None
            if len(results) > 0:
                if args.setting in ['perm', 'full', 'cyclic']:
                    # Ensemble over permutations/cycles with content mapping
                    if getattr(args, 'option_id_set', None):
                        option_ids = list(args.option_id_set)
                    else:
                        k_guess = len(results[0]['data']['options'])
                        option_ids = list('ABCDE' if k_guess == 5 else 'ABCD')
                    k = len(option_ids)

                    if args.setting in ['perm', 'full']:
                        perm_list = list(sorted(permutations(range(k))))
                    else:
                        perm_list = _rotations(k)

                    total = 0
                    corrects = 0
                    for r in results:
                        if r.get('type') != 'result':
                            continue
                        data = r['data']
                        probs_seq = data.get('probs', None)
                        if not isinstance(probs_seq, list) or len(probs_seq) != len(perm_list):
                            continue
                        agg = _aggregate_probs_over_permutations(probs_seq, perm_list, k)
                        pred_letter = option_ids[int(np.argmax(agg))]
                        if pred_letter == data['ideal']:
                            corrects += 1
                        total += 1
                    acc = (corrects / total) if total > 0 else float('nan')
                    metrics = {'type': 'metric', 'data': {'accuracy': acc}}
                    logger.info(_purple(f"==== Ensemble report ({args.setting}) ===="))
                    logger.info(f"accuracy: {acc:.4f}")
                else:
                    metrics = {'type': 'metric', 'data': {}}
                    metrics['data']['accuracy'] = get_accuracy(results)
                    metrics['data']['boostrap_std'] = get_bootstrap_accuracy_std(results)
                    logger.info("Final report:")
                    for key, value in metrics['data'].items():
                        logger.info(f"{key}: {value}")
            logger.info(_orange(f"Run completed: {subject}"))

            save_results(f'{args.save_path}/{subject}.jsonl', results, metrics)
            logger.info(f"Results saved: {subject}")

            # Derive cyclic and default (base) outputs automatically from FULL permutation runs
            if args.setting == 'full':
                try:
                    # Determine option ids (respect custom option_id_set)
                    if getattr(args, 'option_id_set', None):
                        option_ids = list(args.option_id_set)
                    else:
                        # Fallback to infer from first result options length
                        k = len(results[0]['data']['options']) if len(results) > 0 and results[0]['type'] == 'result' else 4
                        option_ids = list('ABCDE' if k == 5 else 'ABCD')
                    k = len(option_ids)

                    # Build permutation index lookup (identity and cyclic rotations)
                    perm_list = list(sorted(permutations(range(k))))
                    identity_idx = perm_list.index(tuple(range(k)))
                    cyclic_indices = [perm_list.index(tuple((i + s) % k for i in range(k))) for s in range(k)]

                    # Build derived results
                    cyclic_results = []
                    base_results = []
                    # For reporting ensemble accuracies
                    full_total = 0
                    full_corrects = 0
                    cyclic_total = 0
                    cyclic_corrects = 0
                    # Per-sample correctness lists for beta curves
                    base_correct_list = []
                    cyclic_correct_list = []
                    full_correct_list = []
                    for r in results:
                        if r.get('type') != 'result':
                            continue
                        data = r['data']
                        probs_seq = data['probs']  # list of length (#perms) each with length k
                        if not isinstance(probs_seq, list) or len(probs_seq) <= identity_idx:
                            continue

                        # Cyclic subset
                        cyclic_probs = [probs_seq[idx] for idx in cyclic_indices]
                        cyclic_results.append({
                            'type': 'result',
                            'data': {
                                'idx': data['idx'],
                                'prompt': data.get('prompt'),
                                'options': data['options'],
                                'probs': cyclic_probs,
                                'ideal': data['ideal'],
                            },
                        })
                        # Cyclic ensemble vote for this sample
                        agg_cyc = _aggregate_probs_over_permutations(cyclic_probs, [tuple((i + s) % k for i in range(k)) for s in range(k)], k)
                        pred_cyc = option_ids[int(np.argmax(agg_cyc))]
                        cyclic_correct_list.append(pred_cyc == data['ideal'])
                        if pred_cyc == data['ideal']:
                            cyclic_corrects += 1
                        cyclic_total += 1

                        # Default (identity) subset -> base-style result
                        base_probs = probs_seq[identity_idx]
                        sampled = option_ids[int(np.argmax(np.array(base_probs)))]
                        correct = (sampled == data['ideal'])
                        base_correct_list.append(correct)
                        base_results.append({
                            'type': 'result',
                            'data': {
                                'idx': data['idx'],
                                'prompt': data.get('prompt'),
                                'options': data['options'],
                                'probs': base_probs,
                                'sampled': sampled,
                                'ideal': data['ideal'],
                                'correct': correct,
                            },
                        })

                        # Full ensemble vote for this sample
                        agg_full = _aggregate_probs_over_permutations(probs_seq, perm_list, k)
                        pred_full = option_ids[int(np.argmax(agg_full))]
                        full_correct_list.append(pred_full == data['ideal'])
                        if pred_full == data['ideal']:
                            full_corrects += 1
                        full_total += 1

                    # Save cyclic-derived results
                    cyclic_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_cyclic'
                    if getattr(args, 'option_id_set', None):
                        cyclic_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(cyclic_save_path, exist_ok=True)
                    cyclic_metrics = None
                    if cyclic_total > 0:
                        cyclic_acc = cyclic_corrects / cyclic_total
                        cyclic_metrics = {'type': 'metric', 'data': {'accuracy': cyclic_acc}}
                        logger.info(_purple(f"[{subject}] Cyclic ensemble accuracy: {cyclic_acc:.4f}"))
                    save_results(f'{cyclic_save_path}/{subject}.jsonl', cyclic_results, metrics=cyclic_metrics)
                    logger.info(_orange(f"Derived and saved cyclic results (with metrics): {subject}"))

                    # Save base-derived results with metrics
                    base_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}'
                    if getattr(args, 'option_id_set', None):
                        base_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(base_save_path, exist_ok=True)
                    base_metrics = None
                    if len(base_results) > 0:
                        base_metrics = {'type': 'metric', 'data': {}}
                        base_metrics['data']['accuracy'] = get_accuracy(base_results)
                        base_metrics['data']['boostrap_std'] = get_bootstrap_accuracy_std(base_results)
                        logger.info("Derived base report:")
                        for key, value in base_metrics['data'].items():
                            logger.info(f"{key}: {value}")
                    save_results(f'{base_save_path}/{subject}.jsonl', base_results, base_metrics)
                    logger.info(_orange(f"Derived and saved base results (with metrics): {subject}"))

                    # Report FULL ensemble accuracy
                    if full_total > 0:
                        full_acc = full_corrects / full_total
                        logger.info(_purple(f"[{subject}] Full permutation ensemble accuracy: {full_acc:.4f}"))

                    # Unified 3-accuracy summary (Full, Cyclic, Default)
                    summary_full = full_acc if full_total > 0 else float('nan')
                    summary_cyc = cyclic_acc if cyclic_total > 0 else float('nan')
                    summary_base = base_metrics['data']['accuracy'] if (base_metrics is not None and 'accuracy' in base_metrics['data']) else float('nan')
                    logger.info(_purple(f"[{subject}] Accuracies — Full: {summary_full:.4f}, Cyclic: {summary_cyc:.4f}, Default: {summary_base:.4f}"))

                    # Compute beta curves (0.0, 0.1, ..., 1.0) with cost on x-axis
                    if len(base_correct_list) == len(cyclic_correct_list) == len(full_correct_list) and len(base_correct_list) > 0:
                        N = len(base_correct_list)
                        betas = [i / 10.0 for i in range(11)]
                        # Cost per sample: beta*C + (1-beta)*1 (C = k or k!)
                        C_cyc = float(k)
                        # factorial for k
                        C_full = float(math.factorial(k))
                        curve_cyc = []
                        curve_full = []
                        # Deterministic subset: use first n indices (results already sorted by idx)
                        for beta in betas:
                            n = int(N * beta + 1e-9)
                            # Cyclic mix accuracy
                            if n > 0:
                                acc_cyc = (sum(cyclic_correct_list[:n]) + sum(base_correct_list[n:])) / float(N)
                            else:
                                acc_cyc = sum(base_correct_list) / float(N)
                            cost_cyc = (beta * C_cyc) + ((1.0 - beta) * 1.0)
                            curve_cyc.append((cost_cyc, acc_cyc))
                            # Full mix accuracy
                            if n > 0:
                                acc_full_mix = (sum(full_correct_list[:n]) + sum(base_correct_list[n:])) / float(N)
                            else:
                                acc_full_mix = sum(base_correct_list) / float(N)
                            cost_full = (beta * C_full) + ((1.0 - beta) * 1.0)
                            curve_full.append((cost_full, acc_full_mix))

                        logger.info(_purple(f"[{subject}] Beta curve (Cyclic): " + ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_cyc])))
                        logger.info(_purple(f"[{subject}] Beta curve (Full): " + ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_full])))

                        # Save curve data
                        curve_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full'
                        if getattr(args, 'option_id_set', None):
                            curve_save_path += f'_id-{args.option_id_set}'
                        os.makedirs(curve_save_path, exist_ok=True)
                        curve_obj = {
                            'subject': subject,
                            'k': k,
                            'betas': betas,
                            'default_accuracy': summary_base,
                            'cyclic': {
                                'costs': [c for c, _ in curve_cyc],
                                'accuracies': [a for _, a in curve_cyc],
                            },
                            'full': {
                                'costs': [c for c, _ in curve_full],
                                'accuracies': [a for _, a in curve_full],
                            },
                        }
                        save_results(f'{curve_save_path}/{subject}_beta_curve.jsonl', [curve_obj], metrics=None)

                        # W&B logging: one sample's prompts + per-permutation probs, and the beta curve figure
                        if wandb_run is not None:
                            try:
                                import wandb

                                # Choose sample to log
                                target_idx = args.wandb_sample_idx
                                chosen = None
                                for r in results:
                                    if r.get('type') != 'result':
                                        continue
                                    if target_idx is None or int(r['data']['idx']) == int(target_idx):
                                        chosen = r['data']
                                        break
                                if chosen is None and len(results) > 0 and results[0].get('type') == 'result':
                                    chosen = results[0]['data']

                                # Log prompts/probs table
                                if chosen is not None and 'prompts' in chosen and 'probs' in chosen:
                                    prompts_list = chosen['prompts']
                                    probs_seq = chosen['probs']
                                    cols = ['perm_idx', 'ideal', 'prompt'] + [f'prob_{oid}' for oid in option_ids]
                                    rows = []
                                    for pi, (ptext, pvec) in enumerate(zip(prompts_list, probs_seq)):
                                        rows.append([pi, chosen['ideal'], ptext] + [float(x) for x in pvec])
                                    table = wandb.Table(columns=cols, data=rows)
                                    wandb.log({f"{subject}/sample_prompts": table})

                                # Build and log beta curve figure
                                fig = plt.figure(figsize=(7.5, 5.0), dpi=160)
                                cyc_costs = [c for c, _ in curve_cyc]
                                cyc_accs = [a for _, a in curve_cyc]
                                full_costs = [c for c, _ in curve_full]
                                full_accs = [a for _, a in curve_full]
                                plt.plot(cyc_costs, cyc_accs, marker='o', label='Cyclic (k rotations)')
                                plt.plot(full_costs, full_accs, marker='o', label='Full (k! permutations)')
                                plt.scatter([1.0], [summary_base], marker='*', s=180, c='black', label='Default')
                                plt.xlabel("Computational Cost (× of default)")
                                plt.ylabel("Accuracy")
                                plt.title(f"Accuracy vs. Cost — {subject}")
                                plt.grid(True, linestyle='--', alpha=0.4)
                                plt.legend()
                                plt.tight_layout()
                                wandb.log({f"{subject}/beta_curve": wandb.Image(fig)})
                                plt.close(fig)

                                # Log scalar metrics as well
                                payload = {
                                    f"{subject}/acc_full": summary_full,
                                    f"{subject}/acc_cyclic": summary_cyc,
                                    f"{subject}/acc_default": summary_base,
                                }
                                wandb.log(payload)
                            except Exception as e:
                                logger.warning(f"W&B logging failed: {e}")
                except Exception as e:
                    logger.warning(f"Failed to derive cyclic/base from full for subject '{subject}': {e}")

            logging_cuda_memory_usage()

    # Finalize W&B
    try:
        if wandb_run is not None:
            import wandb
            wandb.finish()
    except Exception:
        pass

if __name__ == "__main__":
    main()

