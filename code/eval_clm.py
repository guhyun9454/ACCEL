
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
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

import gc

try:
    import pynvml
except Exception:  # pragma: no cover - optional dependency
    pynvml = None

logger = logging.getLogger(__name__)


def _is_safetensors_load_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    markers = [
        "safetensors",
        "model.safetensors",
        "safe_serialization",
        "cannot be loaded with `safetensors`",
        "does not appear to have a file named",
    ]
    return any(m in msg for m in markers)


def _load_model_with_fallback(model_cls, model_path: str, load_kwargs: dict):
    try:
        return model_cls.from_pretrained(model_path, **load_kwargs)
    except Exception as e:
        if load_kwargs.get("use_safetensors", False) and _is_safetensors_load_error(e):
            retry_kwargs = dict(load_kwargs)
            retry_kwargs.pop("use_safetensors", None)
            logger.warning(
                "Model load with use_safetensors=True failed (%s). Retrying without use_safetensors.",
                e,
            )
            return model_cls.from_pretrained(model_path, **retry_kwargs)
        raise


def logging_cuda_memory_usage():
    if pynvml is None:
        logger.info("pynvml is not installed; skipping GPU memory usage logging.")
        return
    logger.info("******** Memory usage ********")
    try:
        n_gpus = pynvml.nvmlDeviceGetCount()
        for i in range(n_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            logger.info("GPU {}: {:.2f} GB / {:.2f} GB".format(i, meminfo.used / 1024 ** 3, meminfo.total / 1024 ** 3))
    except Exception as e:
        logger.warning(f"Failed to query GPU memory usage via pynvml: {e}")


def main():
    patch_open()

    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    args = parse_arguments()
    if len(args.eval_names) == 0:
        exit()

    if pynvml is not None:
        try:
            pynvml.nvmlInit()
        except Exception as e:
            logger.warning(f"pynvml init failed: {e}")

    config = AutoConfig.from_pretrained(args.pretrained_model_path)
    args.is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
    arch = "encoder-decoder (Seq2SeqLM)" if args.is_encoder_decoder else "decoder-only (CausalLM)"
    logger.info(f"Detected model architecture: {arch}")

    toker = AutoTokenizer.from_pretrained(
        args.pretrained_model_path, use_fast=False,
        add_bos_token=False, add_eos_token=False,
    )

    model_cls = AutoModelForSeq2SeqLM if args.is_encoder_decoder else AutoModelForCausalLM
    load_kwargs = {
        "config": config,
        "device_map": "auto",
        "use_safetensors": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    }
    model = _load_model_with_fallback(model_cls, args.pretrained_model_path, load_kwargs)
    logging_cuda_memory_usage()

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

            logger.info(_blue(f"Run started: {subject}"))
            results = eval_all_samples(
                eval_fn, eval_samples,
                name=f'{args.task},{args.num_few_shot},{args.setting},{subject}',
                threads=max(1, torch.cuda.device_count()) if 'falcon' not in args.pretrained_model_path else 1,
            )
            gc.collect()
            torch.cuda.empty_cache()

            metrics = None
            if args.setting not in ['perm', 'cyclic', 'cyclic_swap_text', 'cyclic_swap_id'] and len(results) > 0:
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
