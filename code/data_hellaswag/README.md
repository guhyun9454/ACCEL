# HellaSwag

This directory follows the same CSV format as the other tasks in this repo.

## Create CSVs

```bash
pip install datasets pandas
python data_hellaswag/process.py
```

If you keep datasets under the official `LLM-MCQ-Bias_data/` repo (recommended for large data), you can write there directly:

```bash
python data_hellaswag/process.py --out_dir ../LLM-MCQ-Bias_data/data_hellaswag
```

This writes:
- `data_hellaswag/dev/hellaswag_dev.csv`
- `data_hellaswag/test/hellaswag_test.csv`

Columns (no header):

`Question, A, B, C, D, Answer`

> Note: the official HellaSwag `test` split is unlabeled. The script uses the
> HuggingFace `validation` split as the repo's `test` split (common practice).
