# README

You can find the scripts of running gpt-3.5-turbo or causal language models in `scripts`. Note that for local running you should set the env variable `HF_MODELS` that indicates the save folder of LLMs.

For PriDe, simply modify the hyperparameters in `debias_pride.py` and run it.

## Commercial API logprob evaluation

`eval_clm.py` keeps the existing Hugging Face path as the default and enables
commercial inference only with `--inference_backend api`. Supported pairs are:

- `openai` / `gpt-4.1-2025-04-14`
- `gemini` / `gemini-2.5-flash-lite` (legacy accounts only; probe required)
- `vertex` / `gemini-2.5-flash` (Vertex AI ADC; probe required)
- `deepseek` / `deepseek-v4-flash`
- `together` / `Qwen/Qwen3.5-397B-A17B`

Set only the matching environment variable (`OPENAI_API_KEY`, `GEMINI_API_KEY`,
`DEEPSEEK_API_KEY`, or `TOGETHER_API_KEY`). Vertex uses Application Default
Credentials plus `GOOGLE_CLOUD_PROJECT` and optional
`GOOGLE_CLOUD_LOCATION` (default `global`); point
`GOOGLE_APPLICATION_CREDENTIALS` at a protected service-account file when ADC
is not otherwise configured. Credentials are never written to result files or
W&B. Every API run requires a cost or request guard.

Run a 10-sample capability probe first:

```bash
python eval_clm.py \
  --pretrained_model_path gpt-4.1-2025-04-14 \
  --inference_backend api --api_provider openai \
  --api_prompt_mode label_only \
  --api_execution_mode offline_sweep \
  --api_probe_only --api_probe_samples 10 --api_max_requests 100 \
  --eval_names arc,0,full --option_id_set ABCD --skip_full \
  --empirical_pride --empirical_residual_model empirical \
  --plot_empirical_prefix_fractions 2 \
  --empirical_sweep_mode percentile --empirical_stage_schedule flat \
  --empirical_transition_mode latin
```

For datasets with multiple subjects, the default probe intentionally keeps the
legacy behavior of checking only the first subject. Add
`--api_probe_all_subjects` to stratify the gate across every subject; for
example, `--api_probe_samples 1 --api_probe_all_subjects` checks one question
per MMLU subject and still stops immediately on the first strict coverage
failure.

Gemini capability was rechecked on 2026-07-19. New accounts receive `404` for
Gemini 2.5 models, while the available Gemini 3.1/3.5 and Gemma 4 text models
return `400 Logprobs is not enabled for this model`. The Gemini adapter remains
for legacy accounts, but no current Gemini model should be promoted to a full
PriDe/ACCEL run unless a fresh strict capability probe succeeds.

Vertex AI's native Gemini path is separate from the AI Studio key path. The
adapter requests `response_logprobs`, top-20 candidates, and a zero thinking
budget for `gemini-2.5-flash`. It still requires the same strict A-D/E coverage
probe before any paid sweep.

For a physical adaptive run, the percentile is deliberately required rather
than defaulted. Calibration-prefix questions collect all Latin stages; other
questions stop before the next paid request as soon as the confidence rule is
satisfied:

```bash
python eval_clm.py \
  --pretrained_model_path gpt-4.1-2025-04-14 \
  --inference_backend api --api_provider openai \
  --api_prompt_mode label_only \
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
`--api_prompt_mode baseline` preserves the original PriDe prompt and remains
the default. `label_only` adds the strict one-label instruction only to API
system messages; local Hugging Face prompts and label-restricted scoring are
unchanged. Coverage failures remain invalid and uncached, while their raw top
probabilities, usage, and estimated physical USD are retained in diagnostics.

### Experimental equal-label logit bias

`--api_scoring_mode equal_label_bias --api_equal_label_bias 100` applies the
same OpenAI `logit_bias` to every exact canonical single-token label and records
the result as a separate constrained protocol. It requires
`--api_prompt_mode label_only` and `tiktoken`; the default `topk_strict` cache
payload and the local Hugging Face path remain unchanged.

The live GPT-4.1 probe on 2026-07-22 found that bias changes token selection but
does not expand the returned `top_logprobs`. Forcing `A` changed the emitted
token to `A`, while its emitted logprob was `-9999` and the top-20 still began
with the unbiased `C`. Equal bias did not restore the missing `A` at any tested
value from 20 through 100. This mode is diagnostic only and must not be promoted
to a full PriDe/ACCEL run unless a future endpoint exposes every exact label.

```bash
python probe_equal_label_bias.py \
  --task arc --sample_index 0 --permutation_index 0 \
  --biases 0,20,40,80,100 \
  --cache_dir /path/to/separate/cache \
  --output /path/to/summary.json \
  --max_requests 5 --max_cost_usd 0.05
```

### Experimental structured-label protocols

`probe_structured_labels.py` keeps JSON-schema multiway, pairwise, and
Monte-Carlo experiments outside the normal PriDe/ACCEL result and durable cache
namespaces. Pairwise mode requires every binary label pair to be observed, then
fits sum-to-zero Bradley--Terry scores and records log-odds residuals. It is a
separate constrained protocol: it must not be described as the original
unconstrained first-token distribution.

```bash
python probe_structured_labels.py \
  --task arc --protocol pairwise \
  --probe_samples 10 --all_permutations \
  --max_requests 240 --max_cost_usd 0.20 \
  --output /path/to/arc_10xcyclic_pairwise.json
```

The live GPT-4.1 check found that a four/five-way JSON enum can still omit a
low-probability label from returned top-20 candidates. Complete pairwise enums
improve coverage, but the reported Bradley--Terry residuals must be inspected
because changing the allowed pair can change the model's odds.

The 2026-07-22 all-subject check also found a binary failure: MMLU
`college_chemistry`, sample 0, cyclic permutation 3 returned only `A` in the
top-20 for an A/B enum. Temperature 1 and 2 Monte-Carlo runs (`n=128`) returned
the winning label 128/128 times. Targeting B with positive bias exposed a
sampling transition around bias 40.67, but that is a repeated-sampling inverse
estimate rather than an exact API logprob. None of these structured protocols
is promoted to the main PriDe/ACCEL pipeline.

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
