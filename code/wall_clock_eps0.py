"""Wall-clock cost of ACCEL (final eps=0 method) and its baselines, executed online.

Production runs compute every permutation view up front and simulate the routing offline, so they record no wall
time. Here each method runs only the forward passes it needs, in stream order, and is timed end to end (tokenisation,
forward passes, prior estimation, posterior updates, Latin scheduling, gating). The routing, prompts and posteriors
are the production functions imported from eval_clm.py / eval_clm_utils.py (sm/raceall-eps0 a74db32), so the
sequential mode reproduces the production run exactly; a gate checks this against the stored eps0 trajectories.

Methods (per subject, stream order = production run `--run_idx`):
  base            identity view only (1 pass/question)
  pride           PriDe alpha=2%: k cyclic views on the random prefix, identity view elsewhere, corrected by the prior
  partial_cyclic  identity + (m-1) random non-identity cyclic views on every question (cost m)
  cyclic          all k cyclic views on every question (cost k)
  accel_b<beta>   ACCEL: prefix -> k cyclic views (prior) + its Latin views; others: identity view, then targeted /
                  Latin-completing views while the stage confidence is below the online percentile threshold
Modes:
  seq    one prompt per forward pass (= production scoring)
  batch  up to --batch_size prompts per forward pass (left padding). ACCEL batches the questions of an arrival chunk
         that reach the same stage; gate decisions are still taken in stream order, so routing is unchanged.

usage: python wall_clock_eps0.py --model <hf id> --task mmlu --mode seq --out <json> [--ref_dir <eps0 results dir>]
"""
import argparse
import glob
import json
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_clm import (  # noqa: E402
    _aggregate_probs_over_permutations, _build_option_user_prompt, _build_targeted_latin_schedule,
    _compute_empirical_stage_posteriors, _estimate_empirical_pride_bank, _estimate_pride_prior_random_prefix_mean,
    _extract_question_from_user_prompt, _pride_correct_row, _rotations, _stable_u32_seed,
)
from eval_clm_utils import prepare_eval  # noqa: E402

TASK_IDS = {"arc": "ABCD", "csqa": "ABCDE", "mmlu": "ABCD"}


class Scorer:
    """Option-ID probabilities exactly as eval_clm_utils.prepare_eval_fn_perm (0-shot), with an optional batch path."""

    def __init__(self, model, toker, option_ids):
        self.model, self.toker, self.k = model, toker, len(option_ids)
        toker.padding_side = "left"
        if toker.pad_token is None:
            toker.pad_token = toker.eos_token
        self.space = toker(": A", add_special_tokens=False).input_ids[-1] != toker(":A", add_special_tokens=False).input_ids[-1]
        self.idx = ([toker(f": {e}", add_special_tokens=False).input_ids[-1] for e in option_ids]
                    + [toker(f":{e}", add_special_tokens=False).input_ids[-1] for e in option_ids])
        self.n_forward = 0
        self.n_prompts = 0

    def _text(self, sys_msg, user):
        return sys_msg + "\n\n" + user + ("" if self.space else " ")

    def _probs(self, logits):
        p = F.softmax(logits[..., self.idx], dim=-1).detach().to(torch.float32).cpu().numpy()
        return p.reshape(p.shape[0], 2, self.k).sum(axis=1)

    @torch.no_grad()
    def score(self, prompts, batch_size):
        """prompts: list of (sys_msg, user_prompt). Returns (n, k) letter-space probabilities."""
        out = []
        dev = self.model.device
        if batch_size <= 1:
            for s, u in prompts:
                ids = self.toker(self._text(s, u), truncation=False, return_tensors="pt").input_ids.to(dev)[..., -1536:]
                out.append(self._probs(self.model(input_ids=ids).logits[:, -1]))
                self.n_forward += 1
        else:
            for i in range(0, len(prompts), batch_size):
                chunk = [self._text(s, u) for s, u in prompts[i:i + batch_size]]
                enc = self.toker(chunk, truncation=False, padding=True, return_tensors="pt")
                ids, mask = enc.input_ids[..., -1536:].to(dev), enc.attention_mask[..., -1536:].to(dev)
                out.append(self._probs(self.model(input_ids=ids, attention_mask=mask).logits[:, -1]))
                self.n_forward += 1
        self.n_prompts += len(prompts)
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.k))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def view_prompt(meta, slot_to_content, option_ids):
    opts = [meta["options"][c] for c in slot_to_content]
    return (meta["sys_msg"], _build_option_user_prompt(meta["question"], opts, option_ids))


def latin_posteriors(stage_probs, schedule, mu):
    return _compute_empirical_stage_posteriors(stage_probs=np.asarray(stage_probs), slot_to_content_schedule=schedule,
                                               mu_hat=mu, residual_bank=np.zeros((1, len(mu))))


def run_subject(method, samples, metas, labels, scorer, option_ids, k, subject, run_idx, bs, m_partial):
    """Returns (preds, per-question cost in prompts, extra) for one subject in stream order."""
    N = len(samples)
    perm_list = _rotations(k)
    ident = perm_list.index(tuple(range(k)))
    cyc_prompts = [[tuple(p) for p in s[0]] for s in samples]          # the k cyclic views as production renders them
    seed = _stable_u32_seed(str(subject), 0 + run_idx)                 # empirical/pride seed (pride_seed=0)
    m = int(max(1, int(round(N * 0.02))))
    prefix = [int(x) for x in np.random.default_rng(seed).choice(np.arange(N, dtype=np.int64), size=m, replace=False)]
    pset = set(prefix)

    if method == "base":
        p = scorer.score([cyc_prompts[i][ident] for i in range(N)], bs)
        return [int(np.argmax(r)) for r in p], [1] * N, {}

    if method == "cyclic" or method == "partial_cyclic":
        if method == "cyclic":
            views = [list(range(k)) for _ in range(N)]
        else:
            views = []
            for i in range(N):
                rng = random.Random(_stable_u32_seed(f"{subject}:{i}:partial", run_idx))
                views.append([ident] + rng.sample([j for j in range(k) if j != ident], m_partial - 1))
        flat = [cyc_prompts[i][v] for i in range(N) for v in views[i]]
        p = scorer.score(flat, bs)
        preds, o = [], 0
        for i in range(N):
            rows = p[o:o + len(views[i])]; o += len(views[i])
            agg = _aggregate_probs_over_permutations(rows, [perm_list[v] for v in views[i]], k)
            preds.append(int(np.argmax(agg)))
        return preds, [len(v) for v in views], {}

    # prefix: all k cyclic views (prior estimation), shared by pride and accel
    pref_rows = scorer.score([cyc_prompts[i][v] for i in prefix for v in range(k)], bs)
    per_sample = [None] * N
    for j, i in enumerate(prefix):
        per_sample[i] = pref_rows[j * k:(j + 1) * k]
    cyclic_indices = [perm_list.index(tuple((i + s) % k for i in range(k))) for s in range(k)]

    if method == "pride":
        prior, meta = _estimate_pride_prior_random_prefix_mean(per_sample, cyclic_indices, k, 0.02, seed)
        assert sorted(meta["prefix_ids"]) == sorted(prefix)
        rest = [i for i in range(N) if i not in pset]
        p = scorer.score([cyc_prompts[i][ident] for i in rest], bs)
        base = {i: per_sample[i][ident] for i in prefix}
        base.update({i: r for i, r in zip(rest, p)})
        preds = [int(np.argmax(_pride_correct_row(base[i], prior))) for i in range(N)]
        return preds, [k if i in pset else 1 for i in range(N)], {}

    # ACCEL
    beta = float(method.split("_b")[1])
    _, mu, _, meta = _estimate_empirical_pride_bank(per_sample, cyclic_indices, k, 0.02, seed)
    assert sorted(meta["prefix_ids"]) == sorted(prefix)
    q = max(0.0, min(1.0, beta / 100.0))
    hist = [[] for _ in range(k)]
    state = {}                                                          # i -> dict(probs, schedule, confs, preds)

    def stage1(idxs, rows):
        for i, r in zip(idxs, rows):
            post, pred, conf = latin_posteriors(r.reshape(1, -1), [tuple(range(k))], mu)
            top1 = int(pred[0]); runner = int(np.argsort(post[0])[::-1][1])
            state[i] = {"probs": [r], "sched": _build_targeted_latin_schedule(k, top1, runner),
                        "conf": [conf[0]], "pred": [pred[0]]}

    def extend(i, row):
        st = state[i]; st["probs"].append(row)
        post, pred, conf = latin_posteriors(st["probs"], st["sched"][:len(st["probs"])], mu)
        st["conf"].append(conf[-1]); st["pred"].append(pred[-1])

    # prefix questions: identity row from the cyclic views, then every Latin view (forced to full depth)
    stage1(prefix, [per_sample[i][ident] for i in prefix])
    lat = [(i, t) for i in prefix for t in range(1, k)]
    rows = scorer.score([view_prompt(metas[i], state[i]["sched"][t], option_ids) for i, t in lat], bs)
    for (i, t), r in zip(lat, rows):
        extend(i, r)
    extra_prefix_latin = len(lat)

    preds, cost = [None] * N, [0] * N
    chunk = max(1, bs)
    for c0 in range(0, N, chunk):
        idxs = list(range(c0, min(N, c0 + chunk)))
        new = [i for i in idxs if i not in pset]
        stage1(new, scorer.score([cyc_prompts[i][ident] for i in new], bs))
        active = list(idxs)                                             # still undecided, in stream order
        for t in range(k):                                              # decide stage t+1 in stream order
            routed = []
            for i in active:
                st = state[i]
                if i in pset:
                    hist[t].append(st["conf"][t])
                    if t == k - 1:
                        preds[i], cost[i] = st["pred"][-1], k
                    else:
                        routed.append(i)
                    continue
                thr = float(np.quantile(np.asarray(hist[t]), q)) if hist[t] else 0.0
                hist[t].append(st["conf"][t])
                if st["conf"][t] >= thr or t == k - 1:
                    preds[i], cost[i] = st["pred"][t], t + 1
                else:
                    routed.append(i)
            if t == k - 1:
                break
            need = [i for i in routed if i not in pset]
            rows = scorer.score([view_prompt(metas[i], state[i]["sched"][t + 1], option_ids) for i in need], bs)
            for i, r in zip(need, rows):
                extend(i, r)
            active = routed
    return preds, cost, {"extra_prefix_latin_prompts": extra_prefix_latin}


def production_reference(ref_dir, task, subject, run_idx, k, betas):
    """Production run: identity-view probs and route_ours (latin_cges_eps0 logic) per beta, from stored files."""
    rows = [json.loads(l) for l in open(f"{ref_dir}/{subject}_run{run_idx}.jsonl") if l.strip()]
    rows = [r for r in rows if r.get("type") == "result"]
    traj = {}
    for l in open(f"{ref_dir}/empirical_analysis/{subject}_run{run_idx}_empirical_alpha2_trajectories.jsonl"):
        if l.strip():
            o = json.loads(l); traj[int(o["sample_pos"])] = o
    infos = [traj[i] for i in range(len(rows))]
    out = {"identity_probs": [r["data"]["probs"][0] for r in rows]}
    for b in betas:
        hist = [[] for _ in range(k)]; q = b / 100.0; preds, cost = [], []
        for t in infos:
            confs, pbs, ds = t["conf_by_stage"], t["pred_by_stage"], t["decision_stages"]
            stop = ds[-1]
            if not t["prefix_forced"]:
                for li, sid in enumerate(ds):
                    h = hist[sid - 1]
                    if confs[li] >= (float(np.quantile(np.asarray(h), q)) if h else 0.0):
                        stop = sid; break
            sli = ds.index(stop); preds.append(int(pbs[sli])); cost.append(stop)
            for li, sid in enumerate(ds[:sli + 1]):
                hist[sid - 1].append(confs[li])
        out[b] = (preds, cost)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--task", required=True, choices=sorted(TASK_IDS))
    ap.add_argument("--mode", choices=["seq", "batch"], required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--run_idx", type=int, default=0)
    ap.add_argument("--betas", default="10,40,80")
    ap.add_argument("--partial_m", type=int, default=2)
    ap.add_argument("--methods", default="base,pride,partial_cyclic,cyclic,accel")
    ap.add_argument("--ref_dir", default=None, help="eps0 production results dir (seq-mode gate)")
    ap.add_argument("--max_subjects", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bs = 1 if a.mode == "seq" else a.batch_size
    ids = TASK_IDS[a.task]; option_ids = list(ids); k = len(ids)
    betas = [float(x) for x in a.betas.split(",") if x]
    methods = []
    for m in a.methods.split(","):
        methods += [f"accel_b{b:g}" for b in betas] if m == "accel" else [m]

    toker = AutoTokenizer.from_pretrained(a.model, add_bos_token=False, add_eos_token=False, cache_dir=a.cache_dir)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    AutoConfig.from_pretrained(a.model, cache_dir=a.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map="auto", use_safetensors=True,
                                                 torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
                                                 cache_dir=a.cache_dir).eval()
    scorer = Scorer(model, toker, option_ids)

    ev = SimpleNamespace(task=a.task, option_id_set=ids, skip_full=True, result_tag="wallclock_tmp",
                         inference_backend="local", num_few_shot=0, setting="full")
    subjects, _, prep_samples, _ = prepare_eval(ev, f"{a.task},0,full")
    if a.max_subjects:
        subjects = subjects[:a.max_subjects]
    data = []
    for s in subjects:
        smp = prep_samples(s)
        random.Random(a.run_idx + 42).shuffle(smp)                     # production stream order (n_runs > 1)
        metas = [{"sys_msg": e[0][0][0], "question": _extract_question_from_user_prompt(e[0][0][1]),
                  "options": list(e[1])} for e in smp]
        labels = [option_ids.index(e[2]) for e in smp]
        data.append((s, smp, metas, labels))

    # warm-up (excluded from timing)
    scorer.score([tuple(data[0][1][0][0][0])] * max(2, bs), bs); sync()

    out = {"model": a.model, "task": a.task, "mode": a.mode, "batch_size": bs, "run_idx": a.run_idx,
           "n_items": sum(len(d[1]) for d in data), "n_subjects": len(data),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "methods": {}}
    ref = {}
    if a.ref_dir:
        ref = {s: production_reference(a.ref_dir, a.task, s, a.run_idx, k, betas) for s, *_ in data}
    for meth in methods:
        f0, p0 = scorer.n_forward, scorer.n_prompts
        sync(); t0 = time.perf_counter()
        per_subj = []
        for s, smp, metas, labels in data:
            preds, cost, extra = run_subject(meth, smp, metas, labels, scorer, option_ids, k, s, a.run_idx, bs, a.partial_m)
            per_subj.append((s, preds, cost, labels, extra))
        sync(); wall = time.perf_counter() - t0
        accs = [float(np.mean(np.asarray(p) == np.asarray(l))) for _, p, _, l, _ in per_subj]
        n = out["n_items"]
        row = {"wall_sec": wall, "prompts": scorer.n_prompts - p0, "forward_calls": scorer.n_forward - f0,
               "sec_per_question": wall / n, "cost_accounted": float(np.mean([c for _, _, cs, _, _ in per_subj for c in cs])),
               "acc_macro": float(np.mean(accs)),
               "extra_prefix_latin_prompts": int(sum(e.get("extra_prefix_latin_prompts", 0) for *_, e in per_subj))}
        if ref and meth.startswith("accel_b"):
            b = float(meth.split("_b")[1])
            mism = sum(int(np.sum(np.asarray(p) != np.asarray(ref[s][b][0]))) for s, p, *_ in per_subj)
            cmis = sum(int(np.sum(np.asarray(c) != np.asarray(ref[s][b][1]))) for s, _, c, *_ in per_subj)
            row["gate_vs_production"] = {"pred_mismatch": mism, "stop_stage_mismatch": cmis}
        out["methods"][meth] = row
        print(f"[{a.task} {a.mode}] {meth}: {wall:.1f}s prompts={row['prompts']} cost={row['cost_accounted']:.3f} "
              f"acc={row['acc_macro']*100:.2f} {row.get('gate_vs_production', '')}", flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    if ref and a.mode == "seq":
        # identity-view probabilities of the base method vs production run (numerics check)
        mx = 0.0
        for s, smp, *_ in data:
            p = scorer.score([tuple(e[0][0]) for e in smp[:50]], 1)
            mx = max(mx, float(np.max(np.abs(p - np.asarray(ref[s]["identity_probs"][:50])))))
        out["identity_prob_max_abs_diff_first50"] = mx
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"identity prob max |diff| vs production (first 50/subject): {mx:.2e}", flush=True)


if __name__ == "__main__":
    main()
