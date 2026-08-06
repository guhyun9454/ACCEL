<!-- Parent: ../AGENTS.md -->

# ACCEL / LLM-MCQ-Bias (PriDe baseline)

## Purpose
Forked baseline for the ICLR 2024 Spotlight paper "Large Language Models Are Not Robust Multiple Choice Selectors" (Zheng et al.). Used as the PriDe reference implementation underlying ACCEL. Remote: `guhyun9454/ACCEL` (renamed 2026-05 from `guhyun9454/LLM-MCQ-Bias`; old URL 301-redirects). ACCEL-side baseline notes and key-script index live in `../../docs/baselines.md`; this file documents how to *operate inside this repo*.

## Layout

| Path | Role |
|------|------|
| `code/` | PriDe / base / instruct-chat eval implementations. See `code/README.md` (upstream). |
| `code/debias_pride.py` | PriDe debiasing — primary reference for ACCEL's calibration logic. |
| `code/debias_base.py` | Base (no-debias) eval. |
| `code/eval_clm.py` / `eval_clm_utils.py` | Causal-LM scoring (log-likelihood over option-ID tokens). |
| `code/eval_ichat.py` / `eval_ichat_utils.py` | Instruct/chat-model eval path. |
| `code/load_model_simple.py` | Project-local helper to load HF models (added in this fork; see git log). |
| `code/run_all.py` | Orchestrator that loops over datasets/models. |
| `code/scripts/` | Upstream shell entry points (`run_debias.sh`, `run_ichat.sh`, `run_llama-7b.sh`). |
| `experiments/` | **ACCEL-side scripts** — Korean prompt variants (`arc-csqa-ko*.sh`), `pride.bash`, multi-GPU launchers, model-download helpers. Do not push these upstream. |
| `data/data_arc/`, `data/data_csqa/`, `data/data_mmlu/` | Dataset payloads checked into the repo. |
| `models.txt` | List of HF model IDs that scripts iterate over. |
| `requirements.txt` | Upstream pinned deps. |

Note: this table reflects the `main` (upstream PriDe) view. The active `sm/table_3_4_1` branch adds `tests/`, `code/api_inference.py` (commercial-API backend + model registry), `code/eval_clm_online.py`/`eval_clm_reporting.py`/`eval_clm_plots.py`, and Streamlit result tooling.

## For AI Agents

### Testing
- Canonical unit-test command (validated 2026-07-23, 39 tests, **no API keys needed** — fake-response tests): `python -m unittest tests.test_api_inference`. Use unittest, not pytest (pytest is not in `requirements.txt`).
- This local machine (Jetson) has no ready experiment env. Working recipe: `python -m venv --system-site-packages <tmp>` over the `torch-gpu` conda env (`~/miniconda3/envs/torch-gpu`), then `pip install tqdm pandas google-auth`. Real API/GPU runs belong on Seraph or a machine with keys.

### Running
- Set `HF_MODELS` env var before any local run — it points at the on-disk model cache (Seraph: typically `/data/$USER/g/models` or similar; confirm with the user). The upstream README requires this.
- Conda env name is **TBD** — see root `../../AGENTS.md` "To Build" list. Confirm with the user before invoking `conda activate`.
- Multi-GPU launches go through `experiments/run_8gpu.sh` / `_grab_y1_8gpu.sh`. Single-GPU debug via `experiments/test_single_gpu.sh`.
- For Korean-prompt experiments use `experiments/arc-csqa-ko*.sh`. Token-ID conventions (가나다라 vs ABCD) are encoded inside each script.

### Modifying
- Anything genuinely useful upstream (bug fix, broader feature) → commit on a topic branch and consider a PR; everything ACCEL-specific stays under `experiments/` so the upstream diff is small.
- `code/__pycache__/` is ignored. Do not check it in.

### Where context lives
- Method-level "what PriDe does and why ACCEL extends it" → `../../docs/method.md`.
- "Which experiment to run for X result" → `../../docs/baselines.md`.
- Run IDs / wandb sweep IDs → `../../docs/results.md`.

## Dependencies
- HuggingFace `transformers`, `torch` (versions in `requirements.txt`).
- Wandb (entity `capde`).
- KHU Seraph cluster for GPUs (see the `seraph` Claude skill).
