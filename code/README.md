# README

You can find the scripts of running gpt-3.5-turbo or causal language models in `scripts`. Note that for local running you should set the env variable `HF_MODELS` that indicates the save folder of LLMs.

For PriDe, simply modify the hyperparameters in `debias_pride.py` and run it.

## Added: Ablation study helpers (for Ours / PriDe-style analyses)

### New datasets

* **HellaSwag** support is added as `hellaswag`.
  * Create CSVs via:
    ```bash
    pip install datasets pandas
    python data_hellaswag/process.py
    ```

### New evaluation settings

These settings are intended to help separate **option-ID token bias** vs
**position bias**, and to reproduce PriDe-style ablations:

* `shuffle_both` (existing): shuffle *(ID, option text)* pairs (breaks ID–position coupling).
* `noid` (existing): remove option IDs; evaluate by option-text likelihood.

Additional settings added for deeper ablations:

* `swap_text`: keep IDs fixed (`A/B/C/D`), deterministically shuffle **option texts**.
  * Note: the *correct* ID changes under this condition.
* `swap_id`: keep option texts fixed, deterministically shuffle **option IDs**.
  * Note: the *correct* ID changes under this condition.
* `cyclic_swap_text`: cyclic-probing prompts built on top of `swap_text`.
* `cyclic_swap_id`: cyclic-probing prompts built on top of `swap_id`.

### CSV + plots for ablation runs

The script `ablation_study.py` aggregates results and writes a single CSV. It
also optionally produces per-task bar plots.

Example:

```bash
python ablation_study.py \
  --tasks mmlu arc csqa hellaswag \
  --model llama-7b \
  --num_shot 0 \
  --output_csv ablation_results.csv \
  --make_plots
```

If you want to include your **Ours** method in the CSV/plots, pass a callable:

```bash
python ablation_study.py ... \
  --ours_callable path/to/ours_impl.py:predict
```

See `ablation_study.py:evaluate_ours()` for the expected signature.

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
