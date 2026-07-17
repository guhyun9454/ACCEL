# README

You can find the scripts of running gpt-3.5-turbo or causal language models in `scripts`. Note that for local running you should set the env variable `HF_MODELS` that indicates the save folder of LLMs.

For PriDe, simply modify the hyperparameters in `debias_pride.py` and run it.

## Commercial API logprob evaluation

`eval_clm.py` keeps the existing Hugging Face path as the default and enables
commercial inference only with `--inference_backend api`. Supported pairs are:

- `openai` / `gpt-4.1-2025-04-14`
- `gemini` / `gemini-2.5-flash-lite`
- `deepseek` / `deepseek-v4-flash`
- `together` / `Qwen/Qwen3.5-397B-A17B`

Set only the matching environment variable (`OPENAI_API_KEY`, `GEMINI_API_KEY`,
`DEEPSEEK_API_KEY`, or `TOGETHER_API_KEY`). Credentials are never written to
result files or W&B. Every API run requires a cost or request guard.

Run a 10-sample capability probe first:

```bash
python eval_clm.py \
  --pretrained_model_path gpt-4.1-2025-04-14 \
  --inference_backend api --api_provider openai \
  --api_execution_mode offline_sweep \
  --api_probe_only --api_probe_samples 10 --api_max_requests 100 \
  --eval_names arc,0,full --option_id_set ABCD --skip_full \
  --empirical_pride --empirical_residual_model empirical \
  --plot_empirical_prefix_fractions 2 \
  --empirical_sweep_mode percentile --empirical_stage_schedule flat \
  --empirical_transition_mode latin
```

For a physical adaptive run, the percentile is deliberately required rather
than defaulted. Calibration-prefix questions collect all Latin stages; other
questions stop before the next paid request as soon as the confidence rule is
satisfied:

```bash
python eval_clm.py \
  --pretrained_model_path gpt-4.1-2025-04-14 \
  --inference_backend api --api_provider openai \
  --api_execution_mode adaptive --api_adaptive_percentile 80 \
  --api_max_cost_usd 10 --api_max_requests 20000 \
  --eval_names arc,0,full --option_id_set ABCD --skip_full \
  --n_runs 3 --wandb --wandb_project 3_arc_api_adaptive \
  --empirical_pride --empirical_residual_model empirical \
  --plot_empirical_prefix_fractions 2 \
  --empirical_sweep_mode percentile --empirical_stage_schedule flat \
  --empirical_transition_mode latin \
  --result_tag api_adaptive_p80
```

`--force` recomputes result files but reuses the durable paid-response cache.
Add `--api_force_requests` only when intentional new provider calls are needed.
With `--n_runs 3`, the run index is part of the cache namespace, so all three
runs are independent API repetitions. Missing option labels in first-token
top-20 logprobs fail immediately and are recorded in `diagnostics.jsonl`.

If you find this repository useful or our work is related to your research, please kindly cite it:

```latex
@inproceedings{
  llm-mcq-bias,
  title={Large Language Models Are Not Robust Multiple Choice Selectors},
  author={Chujie Zheng and Hao Zhou and Fandong Meng and Jie Zhou and Minlie Huang},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=shr9PXz7T0}
}
```

## Experimental Results

Our experimental results are released in another data repo: https://github.com/chujiezheng/LLM-MCQ-Bias_data 
