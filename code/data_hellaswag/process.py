"""Preprocess HellaSwag into the CSV format expected by this repo.

This repo expects:
  - data_<task>/dev/<subject>_dev.csv
  - data_<task>/test/<subject>_test.csv

For HellaSwag we create a single subject file:
  - data_hellaswag/dev/hellaswag_dev.csv
  - data_hellaswag/test/hellaswag_test.csv

CSV columns (no header):
  Question, A, B, C, D, Answer

Answer is the correct option ID in {A,B,C,D}.

Usage:
  python data_hellaswag/process.py

Optional:
  python data_hellaswag/process.py --max_rows 5000
  python data_hellaswag/process.py --out_dir ../LLM-MCQ-Bias_data/data_hellaswag

Notes:
  * We use the HuggingFace `datasets` package.
  * The official HellaSwag "test" split does not contain labels; we therefore
    use `validation` as our "test" here (common evaluation practice).
"""

import argparse
import os
from typing import Any, Dict, List, Optional

import pandas as pd


def _build_question(ex: Dict[str, Any]) -> str:
    # Different dataset versions expose slightly different fields.
    if 'ctx' in ex and ex['ctx']:
        return str(ex['ctx']).strip()

    ctx_a = str(ex.get('ctx_a', '')).strip()
    ctx_b = str(ex.get('ctx_b', '')).strip()
    if ctx_a and ctx_b:
        return f"{ctx_a} {ctx_b}".strip()
    return (ctx_a or ctx_b).strip()


def _convert_split(split, out_csv_path: str, max_rows: Optional[int] = None) -> None:
    rows: List[List[str]] = []

    for i, ex in enumerate(split):
        if max_rows is not None and i >= max_rows:
            break

        q = _build_question(ex)
        endings = ex['endings']
        if not isinstance(endings, list) or len(endings) != 4:
            raise ValueError(f"Unexpected endings format at row {i}: {type(endings)} / len={len(endings)}")

        label = int(ex['label'])
        if label < 0 or label > 3:
            raise ValueError(f"Unexpected label at row {i}: {label}")

        ans = 'ABCD'[label]
        rows.append([
            q,
            str(endings[0]).strip(),
            str(endings[1]).strip(),
            str(endings[2]).strip(),
            str(endings[3]).strip(),
            ans,
        ])

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df = pd.DataFrame(rows, columns=['Question', 'A', 'B', 'C', 'D', 'Answer'])
    df.to_csv(out_csv_path, index=False, header=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_rows', type=int, default=None, help='Optional cap for quick debugging')
    parser.add_argument(
        '--out_dir',
        type=str,
        default='data_hellaswag',
        help=(
            'Output directory where dev/ and test/ will be created. '
            'Default: data_hellaswag (relative to current working directory).'
        ),
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "`datasets` is required. Install with: pip install datasets"
        ) from e

    dataset = load_dataset('hellaswag')

    # Common practice: use train for dev (few-shot pool), validation for test.
    dev_split = dataset['train']
    test_split = dataset['validation']

    out_dir = args.out_dir
    _convert_split(dev_split, os.path.join(out_dir, 'dev', 'hellaswag_dev.csv'), max_rows=args.max_rows)
    _convert_split(test_split, os.path.join(out_dir, 'test', 'hellaswag_test.csv'), max_rows=args.max_rows)

    print('Done.')


if __name__ == '__main__':
    main()
