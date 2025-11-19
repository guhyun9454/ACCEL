# eval_clm.py
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
from types import SimpleNamespace
from itertools import permutations
from collections import Counter
import math
import gc

import numpy as np
import torch
import matplotlib.pyplot as plt
import pynvml

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
    content-인덱스로 재정렬 후 평균
    probs_seq: list of length (#perms), each a length-k probs for letters
    permuted_indices: list of tuples, permutation p: letter j -> content idx p[j]
    """
    agg = np.zeros(k, dtype=np.float64)
    for perm_idx, p in enumerate(permuted_indices):
        letter_probs = np.asarray(probs_seq[perm_idx], dtype=np.float64)
        for j in range(k):
            agg[p[j]] += letter_probs[j]
    if len(permuted_indices) > 0:
        agg /= float(len(permuted_indices))
    return agg


def _read_results_file(file_path):
    try:
        lines = [json.loads(line) for line in open(file_path)]
        lines = [e for e in lines if e.get('type') == 'result']
        lines = sorted(lines, key=lambda x: int(x['data']['idx']))
        return lines
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _conf_gap(pvec: np.ndarray) -> float:
    vals = np.sort(pvec)[::-1]
    return float(vals[0] - vals[1]) if vals.shape[0] >= 2 else 0.0


def _warn_and_clamp_beta(beta_raw: float) -> float:
    if beta_raw < 0.0 or beta_raw > 1.0:
        logger.warning(_orange(f"[cascade_beta] expected 0.0~1.0, got {beta_raw}. Clamping into range."))
    return max(0.0, min(1.0, float(beta_raw)))


def _log_conf_stats(tag: str, confs: np.ndarray):
    if confs.size == 0:
        return
    q = np.quantile(confs, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    logger.info(_purple(
        f"[{tag}] conf_gap stats — "
        f"min={q[0]:.4f}, p10={q[1]:.4f}, p25={q[2]:.4f}, "
        f"p50={q[3]:.4f}, p75={q[4]:.4f}, p90={q[5]:.4f}, max={q[6]:.4f}"
    ))


# ==============================
# Token-bias offset helpers
# ==============================

def _get_option_ids_from_results(args, results):
    """
    결과에서 options 길이를 보고 ABCD/ABCDE를 추론.
    args.option_id_set이 있으면 그걸 우선 사용.
    """
    if getattr(args, 'option_id_set', None):
        return list(args.option_id_set)
    for r in results:
        if r.get('type') == 'result':
            try:
                k_guess = len(r['data']['options'])
            except Exception:
                continue
            if k_guess == 5:
                return list('ABCDE')
            else:
                return list('ABCD')
    return list('ABCD')


def _compute_token_bias_mu(results, k: int) -> np.ndarray:
    """
    μ_k = mean log-prob per option index k.
    - base: probs shape (k,)
    - perm/cyclic/full: probs shape (num_perm, k)
    """
    logs_sum = np.zeros(k, dtype=np.float64)
    count = 0
    for r in results:
        if r.get('type') != 'result':
            continue
        p = np.asarray(r['data']['probs'], dtype=np.float64)

        if p.ndim == 1:
            if p.shape[0] != k:
                continue
            logp = np.log(p + 1e-12)
            logs_sum += logp
            count += 1

        elif p.ndim == 2:
            if p.shape[1] != k:
                continue
            logp = np.log(p + 1e-12)          # (num_perm, k)
            logs_sum += logp.sum(axis=0)
            count += p.shape[0]

    if count == 0:
        return np.zeros(k, dtype=np.float64)
    return logs_sum / float(count)


def _apply_token_bias_offset_inplace(results, option_ids, lam: float):
    """
    log p'_k = log p_k - λ * μ_k  적용 후, 다시 softmax해서 probs 업데이트.
    - base/run: probs shape (k,)
    - perm/full/cyclic: probs shape (num_perm, k)
    """
    if lam <= 0.0:
        return

    k = len(option_ids)
    mu = _compute_token_bias_mu(results, k)
    logger.info(_purple(f"[token_bias] lambda={lam:.3f}, mu={mu.tolist()}"))

    for r in results:
        if r.get('type') != 'result':
            continue
        data = r['data']
        p = np.asarray(data['probs'], dtype=np.float64)

        # 단일 벡터 (base / cascade export 등)
        if p.ndim == 1:
            if p.shape[0] != k:
                logger.warning(f"[token_bias] unexpected prob length {p.shape[0]} (k={k})")
                continue
            logp = np.log(p + 1e-12) - lam * mu
            logp = logp - logp.max()
            new_p = np.exp(logp)
            new_p = new_p / (new_p.sum() + 1e-12)
            data['probs'] = new_p.tolist()

            if 'ideal' in data:
                idx_max = int(np.argmax(new_p))
                sampled = option_ids[idx_max]
                data['sampled'] = sampled
                data['correct'] = (sampled == data['ideal'])

        # perm/cyclic/full: probs shape (num_perm, k)
        elif p.ndim == 2:
            if p.shape[1] != k:
                logger.warning(f"[token_bias] unexpected prob shape {p.shape} (k={k})")
                continue
            logp = np.log(p + 1e-12) - lam * mu.reshape(1, -1)
            logp = logp - logp.max(axis=1, keepdims=True)
            new_p = np.exp(logp)
            new_p = new_p / (new_p.sum(axis=1, keepdims=True) + 1e-12)
            data['probs'] = new_p.tolist()

        else:
            logger.warning(f"[token_bias] unsupported prob ndim {p.ndim}")


# ==============================
# main
# ==============================

def main():
    patch_open()
    logging.basicConfig(format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
                        level=logging.INFO)

    args = parse_arguments()
    if len(args.eval_names) == 0:
        sys.exit(0)

    # ---- W&B init (optional) ----
    wandb_run = None
    if getattr(args, 'wandb', False):
        try:
            import wandb
            project = args.wandb_project
            run_name = args.wandb_run_name or f"{args.model_name}-{args.eval_names[0]}"
            wandb_run = wandb.init(
                entity="capde",
                project=project,
                name=run_name,
                config={
                    "pretrained_model_path": args.pretrained_model_path,
                    "model_name": args.model_name,
                    "eval_names": args.eval_names,
                    "option_id_set": args.option_id_set,
                    "token_bias_lambda": getattr(args, "token_bias_lambda", 0.0),
                },
            )
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")
            wandb_run = None

    # ---- tokenizer / model ----
    from transformers import AutoTokenizer, AutoModelForCausalLM

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

    # ---- Compare mode (two option sets) ----
    if getattr(args, 'option_id_sets', None) and len(args.option_id_sets) == 2:
        id_set_a, id_set_b = args.option_id_sets[0], args.option_id_sets[1]

        overall_total = overall_matches = overall_c2i = overall_i2c = 0
        overall_both_correct = overall_both_incorrect = 0

        def _read_results_file_local(file_path):
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

            args_a = SimpleNamespace(pretrained_model_path=args.pretrained_model_path,
                                     model_name=args.model_name, option_id_set=id_set_a)
            args_b = SimpleNamespace(pretrained_model_path=args.pretrained_model_path,
                                     model_name=args.model_name, option_id_set=id_set_b)

            (subjects_a, prepare_fewshot_a, prepare_eval_samples_a, prepare_eval_fn_a) = prepare_eval(args_a, eval_name)
            (subjects_b, prepare_fewshot_b, prepare_eval_samples_b, prepare_eval_fn_b) = prepare_eval(args_b, eval_name)
            assert subjects_a == subjects_b, "Subjects mismatch across option ID sets"

            for subject in subjects_a[::1]:
                logger.info(_blue(f"Preparing (compare): {subject}"))
                few_a = prepare_fewshot_a(subject)
                few_b = prepare_fewshot_b(subject)
                eval_samples_a = prepare_eval_samples_a(subject)
                eval_samples_b = prepare_eval_samples_b(subject)

                # noise 없이 eval fn 래핑
                def make_eval_fn(fn_base, few):
                    def f(model_, toker_, few_):
                        base = fn_base(model_, toker_, few_)
                        def wrapped(sample, rng):
                            return base(sample, rng)
                        return wrapped
                    return f

                eval_fn_a = make_eval_fn(prepare_eval_fn_a, few_a)(model, toker, few_a)
                eval_fn_b = make_eval_fn(prepare_eval_fn_b, few_b)(model, toker, few_b)

                # Prompt example
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

                # cache
                path_a = f'{args_a.save_path}/{subject}.jsonl'
                path_b = f'{args_b.save_path}/{subject}.jsonl'
                results_a = None if getattr(args, 'force', False) else _read_results_file_local(path_a)
                results_b = None if getattr(args, 'force', False) else _read_results_file_local(path_b)

                if results_a is None:
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
                else:
                    logger.info(_blue(f"Using cached results (A): {path_a}"))

                if results_b is None:
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
                else:
                    logger.info(_blue(f"Using cached results (B): {path_b}"))

                # Align by idx
                map_a = {int(r['data']['idx']): r for r in results_a if r['type'] == 'result'}
                map_b = {int(r['data']['idx']): r for r in results_b if r['type'] == 'result'}
                common = sorted(set(map_a.keys()) & set(map_b.keys()))

                matches = total = c2i = i2c = 0
                both_correct = both_incorrect = 0

                for idx in common:
                    ra = map_a[idx]['data']
                    rb = map_b[idx]['data']
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

        if overall_total > 0:
            overall_mr = overall_matches / overall_total
            logger.info(_purple("==== Overall compare summary ====\n" +
                                f"matching_ratio={overall_mr:.4f} (matches/total={overall_matches}/{overall_total})\n" +
                                f"correct->incorrect={overall_c2i}, incorrect->correct={overall_i2c}, both_correct={overall_both_correct}, both_incorrect={overall_both_incorrect}"))
        return

    # ---- Single-run mode ----
    for eval_name in args.eval_names[::1]:
        (subjects, prepare_few_shot_samples, prepare_eval_samples, prepare_eval_fn_raw) = prepare_eval(args, eval_name)

        # noise 없이 eval fn 래핑
        def make_eval_fn(fn_raw, few):
            def f(model_, toker_, few_):
                base = fn_raw(model_, toker_, few_)
                def wrapped(sample, rng):
                    return base(sample, rng)
                return wrapped
            return f

        for subject in subjects[::1]:
            cached_path = f'{args.save_path}/{subject}.jsonl'
            use_cached = (not getattr(args, 'force', False)) and os.path.exists(cached_path)

            logger.info(_blue(f"Preparing: {subject}"))
            few_shot_samples = prepare_few_shot_samples(subject)
            eval_samples = prepare_eval_samples(subject)
            eval_fn = make_eval_fn(prepare_eval_fn_raw, few_shot_samples)(model, toker, few_shot_samples)

            # Prompt example
            if getattr(args, 'print_prompt_example', False) and not printed_example and len(eval_samples) > 0:
                try:
                    first_input, _first_options, _first_ideal = eval_samples[0]

                    def build_input_text(pair):
                        sys_msg, eval_sample = pair
                        text = sys_msg + '\n\n'
                        if args.num_few_shot > 0:
                            for s in few_shot_samples[:args.num_few_shot]:
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

            if use_cached:
                logger.info(_blue(f"Using cached results: {cached_path}"))
                results = _read_results_file(cached_path) or []
            else:
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
                # ---- Token-bias offset (λ > 0일 때만 적용) ----
                option_ids_for_bias = None
                if args.setting != 'noid':
                    option_ids_for_bias = _get_option_ids_from_results(args, results)
                    lam = float(getattr(args, 'token_bias_lambda', 0.0))
                    if lam != 0.0:
                        _apply_token_bias_offset_inplace(results, option_ids_for_bias, lam)

                # ---- Metrics 계산 ----
                if args.setting in ['perm', 'full', 'cyclic']:
                    if option_ids_for_bias is None:
                        option_ids_for_bias = _get_option_ids_from_results(args, results)
                    option_ids = option_ids_for_bias
                    k = len(option_ids)

                    perm_list = list(sorted(permutations(range(k)))) if args.setting in ['perm', 'full'] else _rotations(k)

                    total = corrects = 0
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

            if not use_cached:
                save_results(f'{args.save_path}/{subject}.jsonl', results, metrics)
                logger.info(f"Results saved: {subject}")

            # ========== Derive cyclic/base from FULL runs ==========
            if args.setting == 'full':
                try:
                    if getattr(args, 'option_id_set', None):
                        option_ids = list(args.option_id_set)
                    else:
                        if len(results) > 0 and results[0]['type'] == 'result':
                            k_guess = len(results[0]['data']['options'])
                            option_ids = list('ABCDE' if k_guess == 5 else 'ABCD')
                        else:
                            option_ids = list('ABCD')
                    k = len(option_ids)

                    perm_list = list(sorted(permutations(range(k))))
                    identity_idx = perm_list.index(tuple(range(k)))
                    cyclic_indices = [perm_list.index(tuple((i + s) % k for i in range(k))) for s in range(k)]

                    cyclic_results, base_results = [], []
                    full_total = full_corrects = 0
                    cyclic_total = cyclic_corrects = 0
                    base_correct_list, cyclic_correct_list, full_correct_list = [], [], []

                    for r in results:
                        if r.get('type') != 'result':
                            continue
                        data = r['data']
                        probs_seq = data['probs']
                        if not isinstance(probs_seq, list) or len(probs_seq) <= identity_idx:
                            continue

                        # cyclic subset
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
                        agg_cyc = _aggregate_probs_over_permutations(
                            cyclic_probs,
                            [tuple((i + s) % k for i in range(k)) for s in range(k)],
                            k,
                        )
                        pred_cyc = option_ids[int(np.argmax(agg_cyc))]
                        ok = (pred_cyc == data['ideal'])
                        cyclic_correct_list.append(ok)
                        cyclic_corrects += int(ok)
                        cyclic_total += 1

                        # default(identity)
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

                        # full ensemble
                        agg_full = _aggregate_probs_over_permutations(probs_seq, perm_list, k)
                        pred_full = option_ids[int(np.argmax(agg_full))]
                        okf = (pred_full == data['ideal'])
                        full_correct_list.append(okf)
                        full_corrects += int(okf)
                        full_total += 1

                    # save cyclic
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

                    # save base
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

                    if full_total > 0:
                        full_acc = full_corrects / full_total
                        logger.info(_purple(f"[{subject}] Full permutation ensemble accuracy: {full_acc:.4f}"))

                    summary_full = full_acc if full_total > 0 else float('nan')
                    summary_cyc = cyclic_acc if cyclic_total > 0 else float('nan')
                    summary_base = base_metrics['data']['accuracy'] if (base_metrics is not None and 'accuracy' in base_metrics['data']) else float('nan')
                    logger.info(_purple(f"[{subject}] Accuracies — Full: {summary_full:.4f}, Cyclic: {summary_cyc:.4f}, Default: {summary_base:.4f}"))

                    # ---- Beta curves ----
                    if len(base_correct_list) == len(cyclic_correct_list) == len(full_correct_list) and len(base_correct_list) > 0:
                        N = len(base_correct_list)
                        betas = [i / 10.0 for i in range(11)]
                        C_cyc = float(k)
                        C_full = float(math.factorial(k))
                        curve_cyc, curve_full = [], []
                        for beta in betas:
                            n = int(N * beta + 1e-9)
                            acc_cyc = (sum(cyclic_correct_list[:n]) + sum(base_correct_list[n:])) / float(N) if n > 0 else sum(base_correct_list) / float(N)
                            cost_cyc = (beta * C_cyc) + ((1.0 - beta) * 1.0)
                            curve_cyc.append((cost_cyc, acc_cyc))
                            acc_full_mix = (sum(full_correct_list[:n]) + sum(base_correct_list[n:])) / float(N) if n > 0 else sum(base_correct_list) / float(N)
                            cost_full = (beta * C_full) + ((1.0 - beta) * 1.0)
                            curve_full.append((cost_full, acc_full_mix))
                        logger.info(_purple(f"[{subject}] Beta curve (Cyclic): " + ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_cyc])))
                        logger.info(_purple(f"[{subject}] Beta curve (Full): " + ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_full])))

                        # Ours (dynamic cascading)
                        curve_ours = []
                        ours_cascade_counts_list = []
                        try:
                            order_indices = list(range(len(perm_list)))
                            identity_idx = perm_list.index(tuple(range(k)))
                            if identity_idx != 0:
                                order_indices = [identity_idx] + [i for i in order_indices if i != identity_idx]
                            per_sample_probs, base_probs_list, ideals = [], [], []
                            for r in results:
                                if r.get('type') != 'result':
                                    continue
                                data = r['data']
                                probs_seq = np.asarray(data['probs'], dtype=np.float64)
                                per_sample_probs.append(probs_seq)
                                base_probs_list.append(probs_seq[identity_idx])
                                ideals.append(data['ideal'])

                            default_conf = np.array([_conf_gap(bp) for bp in base_probs_list], dtype=np.float64)
                            _log_conf_stats(f"{subject}/default_conf_all", default_conf)

                            perc = max(min(getattr(args, 'ours_low_conf_percent', 10.0), 100.0), 0.0) / 100.0

                            for beta in betas:
                                n = int(N * beta + 1e-9)

                                # 임계치는 “캐스케이드 후보 집합”에서 측정
                                eval_confs = default_conf[n:] if n < N else default_conf
                                if eval_confs.size == 0:
                                    thresh = float(np.quantile(default_conf, perc))
                                else:
                                    thresh = float(np.quantile(eval_confs, perc))

                                total_cost = 0.0
                                corrects = 0
                                cascade_counts = []

                                # 앞쪽 n개는 1스텝 고정
                                for i in range(0, n):
                                    bp = base_probs_list[i]
                                    pred_letter = option_ids[int(np.argmax(bp))]
                                    if pred_letter == ideals[i]:
                                        corrects += 1
                                    total_cost += 1.0
                                    cascade_counts.append(1)

                                # 나머지는 컨피던스 기준 캐스케이드
                                for i in range(n, N):
                                    probs_seq = per_sample_probs[i]
                                    selected = [order_indices[0]]
                                    agg = _aggregate_probs_over_permutations(
                                        [probs_seq[j].tolist() for j in selected],
                                        [perm_list[j] for j in selected],
                                        k,
                                    )
                                    current_conf = _conf_gap(agg)
                                    t = 1
                                    while (current_conf < thresh) and (t < len(order_indices)):
                                        selected.append(order_indices[t])
                                        agg = _aggregate_probs_over_permutations(
                                            [probs_seq[j].tolist() for j in selected],
                                            [perm_list[j] for j in selected],
                                            k,
                                        )
                                        current_conf = _conf_gap(agg)
                                        t += 1
                                    pred_letter = option_ids[int(np.argmax(agg))]
                                    if pred_letter == ideals[i]:
                                        corrects += 1
                                    total_cost += float(len(selected))
                                    cascade_counts.append(int(len(selected)))

                                acc_ours = (corrects / float(N)) if N > 0 else float('nan')
                                cost_ours = (total_cost / float(N)) if N > 0 else float('nan')
                                curve_ours.append((cost_ours, acc_ours))
                                ours_cascade_counts_list.append(cascade_counts)
                            logger.info(_purple(f"[{subject}] Beta curve (Ours): " + ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_ours])))
                        except Exception as e:
                            logger.warning(f"Failed to compute Ours curve: {e}")
                            curve_ours = []
                            ours_cascade_counts_list = []

                        # Ablations (switch-full / switch-cyclic)
                        curve_ours_switch_full = []
                        curve_ours_switch_cyc = []
                        try:
                            if 'per_sample_probs' not in locals():
                                per_sample_probs, base_probs_list, ideals = [], [], []
                                for r in results:
                                    if r.get('type') != 'result':
                                        continue
                                    data = r['data']
                                    probs_seq = np.asarray(data['probs'], dtype=np.float64)
                                    per_sample_probs.append(probs_seq)
                                    base_probs_list.append(probs_seq[identity_idx])
                                    ideals.append(data['ideal'])

                            default_conf = np.array([_conf_gap(bp) for bp in base_probs_list], dtype=np.float64)
                            perc = max(min(getattr(args, 'ours_low_conf_percent', 10.0), 100.0), 0.0) / 100.0

                            for beta in betas:
                                n = int(N * beta + 1e-9)
                                eval_confs = default_conf[n:] if n < N else default_conf
                                if eval_confs.size == 0:
                                    thresh = float(np.quantile(default_conf, perc))
                                else:
                                    thresh = float(np.quantile(eval_confs, perc))

                                # switch-full
                                total_cost_sf = corrects_sf = 0.0
                                for i in range(0, n):
                                    bp = base_probs_list[i]
                                    pred_letter = option_ids[int(np.argmax(bp))]
                                    if pred_letter == ideals[i]:
                                        corrects_sf += 1
                                    total_cost_sf += 1.0
                                for i in range(n, N):
                                    probs_seq = per_sample_probs[i]
                                    if default_conf[i] < thresh:
                                        agg = _aggregate_probs_over_permutations(
                                            [probs_seq[j].tolist() for j in range(len(perm_list))],
                                            [perm_list[j] for j in range(len(perm_list))],
                                            k,
                                        )
                                        total_cost_sf += float(len(perm_list))
                                    else:
                                        agg = probs_seq[identity_idx]
                                        total_cost_sf += 1.0
                                    pred_letter = option_ids[int(np.argmax(agg))]
                                    if pred_letter == ideals[i]:
                                        corrects_sf += 1
                                acc_sf = (corrects_sf / float(N)) if N > 0 else float('nan')
                                cost_sf = (total_cost_sf / float(N)) if N > 0 else float('nan')
                                curve_ours_switch_full.append((cost_sf, acc_sf))

                                # switch-cyclic
                                total_cost_sc = corrects_sc = 0.0
                                cyclic_indices = [perm_list.index(tuple((i + s) % k for i in range(k))) for s in range(k)]
                                for i in range(0, n):
                                    bp = base_probs_list[i]
                                    pred_letter = option_ids[int(np.argmax(bp))]
                                    if pred_letter == ideals[i]:
                                        corrects_sc += 1
                                    total_cost_sc += 1.0
                                for i in range(n, N):
                                    probs_seq = per_sample_probs[i]
                                    if default_conf[i] < thresh:
                                        agg = _aggregate_probs_over_permutations(
                                            [probs_seq[j].tolist() for j in cyclic_indices],
                                            [perm_list[j] for j in cyclic_indices],
                                            k,
                                        )
                                        total_cost_sc += float(k)
                                    else:
                                        agg = probs_seq[identity_idx]
                                        total_cost_sc += 1.0
                                    pred_letter = option_ids[int(np.argmax(agg))]
                                    if pred_letter == ideals[i]:
                                        corrects_sc += 1
                                acc_sc = (corrects_sc / float(N)) if N > 0 else float('nan')
                                cost_sc = (total_cost_sc / float(N)) if N > 0 else float('nan')
                                curve_ours_switch_cyc.append((cost_sc, acc_sc))

                            logger.info(_purple(f"[{subject}] Beta curve (Ours switch-full): " + ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_ours_switch_full])))
                            logger.info(_purple(f"[{subject}] Beta curve (Ours switch-cyclic): " + ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_ours_switch_cyc])))
                        except Exception as e:
                            logger.warning(f"Failed to compute Ours ablation curves: {e}")
                            curve_ours_switch_full = []
                            curve_ours_switch_cyc = []

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
                        if len(curve_ours) == len(betas):
                            curve_obj['ours'] = {'costs': [c for c, _ in curve_ours],
                                                 'accuracies': [a for _, a in curve_ours]}
                        if len(curve_ours_switch_full) == len(betas):
                            curve_obj['ours_switch_full'] = {'costs': [c for c, _ in curve_ours_switch_full],
                                                             'accuracies': [a for _, a in curve_ours_switch_full]}
                        if len(curve_ours_switch_cyc) == len(betas):
                            curve_obj['ours_switch_cyclic'] = {'costs': [c for c, _ in curve_ours_switch_cyc],
                                                               'accuracies': [a for _, a in curve_ours_switch_cyc]}

                        # Oracle
                        try:
                            default_confs, default_corrects = [], []
                            for r in results:
                                if r.get('type') != 'result':
                                    continue
                                data = r['data']
                                probs_seq = np.asarray(data['probs'], dtype=np.float64)
                                base_probs = probs_seq[identity_idx]
                                default_confs.append(_conf_gap(base_probs))
                                pred_letter = option_ids[int(np.argmax(base_probs))]
                                default_corrects.append(int(pred_letter == data['ideal']))
                            default_confs = np.asarray(default_confs, dtype=np.float64)
                            default_corrects = np.asarray(default_corrects, dtype=np.int32)
                            order = np.argsort(default_confs)
                            oracle_percentiles = list(range(1, 101))
                            oracle_bottom_accs = []
                            for p in oracle_percentiles:
                                n2 = max(1, int(N * (p / 100.0) + 1e-9))
                                sel = order[:n2]
                                acc_bottom = float(default_corrects[sel].mean())
                                oracle_bottom_accs.append(acc_bottom)
                            bottom10 = oracle_bottom_accs[9] if len(oracle_bottom_accs) >= 10 else float('nan')
                            logger.info(_purple(f"[{subject}] Oracle bottom-10% accuracy (default): {bottom10:.4f}"))
                            curve_obj['oracle_low_conf'] = {
                                'percentiles': oracle_percentiles,
                                'accuracies': oracle_bottom_accs,
                                'bottom10_acc': bottom10,
                            }
                        except Exception as e:
                            logger.warning(f"Failed to compute oracle low-confidence accuracy curve: {e}")

                        save_results(f'{curve_save_path}/{subject}_beta_curve.jsonl', [curve_obj], metrics=None)

                        # W&B figures
                        if wandb_run is not None:
                            try:
                                import wandb
                                fig = plt.figure(figsize=(7.5, 5.0), dpi=160)
                                cyc_costs = [c for c, _ in curve_cyc];  cyc_accs = [a for _, a in curve_cyc]
                                full_costs = [c for c, _ in curve_full]; full_accs = [a for _, a in curve_full]
                                plt.plot(cyc_costs, cyc_accs, marker='o', label='Cyclic (k rotations)')
                                plt.plot(full_costs, full_accs, marker='o', label='Full (k! permutations)')
                                if len(curve_ours) == len(betas):
                                    ours_costs = [c for c, _ in curve_ours]; ours_accs = [a for _, a in curve_ours]
                                    plt.plot(ours_costs, ours_accs, marker='o', label='Ours (cascading)')
                                if len(curve_ours_switch_full) == len(betas):
                                    sf_costs = [c for c, _ in curve_ours_switch_full]; sf_accs = [a for _, a in curve_ours_switch_full]
                                    plt.plot(sf_costs, sf_accs, marker='o', label='Ours (switch-full)')
                                if len(curve_ours_switch_cyc) == len(betas):
                                    sc_costs = [c for c, _ in curve_ours_switch_cyc]; sc_accs = [a for _, a in curve_ours_switch_cyc]
                                    plt.plot(sc_costs, sc_accs, marker='o', label='Ours (switch-cyclic)')
                                plt.scatter([1.0], [summary_base], marker='*', s=180, c='black', label='Default')
                                plt.xlabel("Computational Cost (× of default)"); plt.ylabel("Accuracy")
                                plt.title(f"Accuracy vs. Cost — {subject}")
                                plt.grid(True, linestyle='--', alpha=0.4); plt.legend(); plt.tight_layout()
                                out_png = f"{curve_save_path}/{subject}_beta_curve.png"
                                fig.savefig(out_png, dpi=160, bbox_inches='tight')
                                wandb.log({f"{subject}/beta_curve": wandb.Image(out_png)})
                                plt.close(fig)
                            except Exception as e:
                                logger.warning(f"W&B logging failed: {e}")
                except Exception as e:
                    logger.warning(f"Failed to derive cyclic/base from full for subject '{subject}': {e}")

            # ===== [PATCH] Cascade export (token-bias 적용된 probs 기반) =====
            try:
                if getattr(args, 'cascade_export', False) and args.setting in ['full', 'cyclic'] and len(results) > 0:
                    if getattr(args, 'option_id_set', None):
                        option_ids = list(args.option_id_set)
                    else:
                        k_guess = len(results[0]['data']['options'])
                        option_ids = list('ABCDE' if k_guess == 5 else 'ABCD')
                    k = len(option_ids)

                    # Permutation list
                    if args.setting == 'full':
                        perm_list = list(sorted(permutations(range(k))))
                        identity_idx = perm_list.index(tuple(range(k)))
                        cyclic_indices = [perm_list.index(tuple((i + s) % k for i in range(k))) for s in range(k)]
                        order_indices = list(range(len(perm_list)))
                        if identity_idx != 0:
                            order_indices = [identity_idx] + [i for i in order_indices if i != identity_idx]
                    else:  # cyclic
                        perm_list = _rotations(k)
                        identity_idx = 0
                        cyclic_indices = list(range(k))
                        order_indices = list(range(k))

                    # collect per-sample seq
                    per_sample_probs, base_probs_list, ideals = [], [], []
                    for r in results:
                        if r.get('type') != 'result':
                            continue
                        data = r['data']
                        probs_seq = np.asarray(data['probs'], dtype=np.float64)
                        per_sample_probs.append(probs_seq)
                        base_probs_list.append(probs_seq[identity_idx])
                        ideals.append(data['ideal'])
                    N = len(base_probs_list)

                    default_conf = np.array([_conf_gap(bp) for bp in base_probs_list], dtype=np.float64)
                    _log_conf_stats(f"{subject}/cascade_default_conf_all", default_conf)

                    beta = _warn_and_clamp_beta(float(getattr(args, 'cascade_beta', 0.0)))
                    n = int(N * beta + 1e-9)
                    perc = max(min(getattr(args, 'ours_low_conf_percent', 10.0), 100.0), 0.0) / 100.0

                    eval_confs = default_conf[n:] if n < N else default_conf
                    if eval_confs.size == 0:
                        thresh = float(np.quantile(default_conf, perc))
                    else:
                        thresh = float(np.quantile(eval_confs, perc))
                    policy = str(getattr(args, 'cascade_policy', 'ours'))

                    logger.info(_purple(
                        f"[{subject}] Cascade params — beta={beta:.3f} (n={n}/{N}), "
                        f"low_conf_percent={perc*100:.1f}%, thresh={thresh:.5f}, policy={policy}"
                    ))

                    cascade_results = []
                    corrects = 0
                    total_cost = 0.0
                    escalated = 0
                    step_hist = Counter()

                    for i in range(N):
                        probs_seq = per_sample_probs[i]

                        if i < n:
                            agg = probs_seq[identity_idx]
                            steps = 1
                        else:
                            if policy == 'switch_full' and args.setting == 'full':
                                selected = list(range(len(perm_list)))
                                agg = _aggregate_probs_over_permutations(
                                    [probs_seq[j].tolist() for j in selected],
                                    [perm_list[j] for j in selected],
                                    k,
                                )
                                steps = len(selected)
                                escalated += 1
                            elif policy == 'switch_cyclic':
                                if args.setting == 'full':
                                    selected = cyclic_indices
                                    selected_perms = [perm_list[j] for j in cyclic_indices]
                                else:
                                    selected = list(range(k))
                                    selected_perms = [perm_list[j] for j in selected]
                                agg = _aggregate_probs_over_permutations(
                                    [probs_seq[j].tolist() for j in selected],
                                    selected_perms,
                                    k,
                                )
                                steps = len(selected)
                                escalated += 1
                            else:
                                selected = [order_indices[0]]
                                agg = _aggregate_probs_over_permutations(
                                    [probs_seq[selected[0]].tolist()],
                                    [perm_list[selected[0]]],
                                    k,
                                )
                                t = 1
                                while (_conf_gap(agg) < thresh) and (t < len(order_indices)):
                                    selected.append(order_indices[t])
                                    agg = _aggregate_probs_over_permutations(
                                        [probs_seq[j].tolist() for j in selected],
                                        [perm_list[j] for j in selected],
                                        k,
                                    )
                                    t += 1
                                steps = len(selected)
                                if steps > 1:
                                    escalated += 1

                        pred_letter = option_ids[int(np.argmax(agg))]
                        correct = (pred_letter == ideals[i])
                        if correct:
                            corrects += 1
                        total_cost += float(steps)
                        step_hist[steps] += 1

                        cascade_results.append({
                            'type': 'result',
                            'data': {
                                'idx': i,
                                'options': option_ids,
                                'probs': agg.tolist(),
                                'sampled': pred_letter,
                                'ideal': ideals[i],
                                'correct': correct,
                                'cascade_steps': steps,
                            },
                        })

                    acc = (corrects / float(N)) if N > 0 else float('nan')
                    avg_cost = (total_cost / float(N)) if N > 0 else float('nan')
                    cascade_metrics = {'type': 'metric', 'data': {
                        'accuracy': acc,
                        'avg_cost_x_of_default': avg_cost,
                        'beta': beta,
                        'low_conf_percent': float(getattr(args, 'ours_low_conf_percent', 10.0)),
                        'policy': policy,
                        'escalation_rate': (escalated / float(N)) if N > 0 else 0.0,
                        'step_hist': dict(sorted(step_hist.items())),
                    }}

                    base_dir = f"results_{args.task}/{args.num_few_shot}s_{args.model_name}"
                    sub = f"{args.task}_cascade_{policy}_beta{beta:.1f}_p{int(getattr(args,'ours_low_conf_percent',10.0))}"
                    if getattr(args, 'option_id_set', None):
                        sub += f"_id-{args.option_id_set}"
                    save_dir = os.path.join(base_dir, sub)
                    os.makedirs(save_dir, exist_ok=True)
                    save_results(os.path.join(save_dir, f"{subject}.jsonl"), cascade_results, metrics=cascade_metrics)
                    logger.info(_purple(
                        f"[{subject}] Cascade ({policy}, beta={beta:.1f}, p={int(getattr(args,'ours_low_conf_percent',10.0))}): "
                        f"acc={acc:.4f}, avg_cost={avg_cost:.2f}x, escalated={escalated}/{N} "
                        f"({(escalated/float(N) if N>0 else 0.0):.3f}), steps={dict(sorted(step_hist.items()))}, "
                        f"saved → {save_dir}/{subject}.jsonl"
                    ))
            except Exception as e:
                logger.warning(f"Cascade export failed: {e}")
            # ===== [END PATCH] =====

            logging_cuda_memory_usage()

    # ---- finalize W&B ----
    try:
        if wandb_run is not None:
            import wandb
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
