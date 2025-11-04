#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt

def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_preds(dir_model, token):
    """returns: (qid_list, pred_list, gold_list or None) from <dir_model>/<token>/P0.jsonl"""
    jf = os.path.join(dir_model, token, "P0.jsonl")
    if not os.path.exists(jf):
        return [], [], None
    rows = read_jsonl(jf)
    qids, preds, golds = [], [], []
    has_gold = True
    for i, r in enumerate(rows):
        qid = r.get("qid", f"{i}")
        pred = r.get("pred")
        gold = r.get("gold", None)
        if pred is None:
            # 호환: prediction_idx 등 다른 키가 들어온 경우
            for k in ["pred_idx","prediction_idx","prediction"]:
                if k in r:
                    pred = r[k]; break
        if pred is None:
            continue
        qids.append(qid)
        preds.append(int(pred))
        if gold is None:
            has_gold = False
            golds.append(-1)
        else:
            golds.append(int(gold))
    return qids, preds, (golds if has_gold else None)

def intersect_align(token_dict):
    """
    token_dict: {token: (qids, preds, golds or None)}
    returns aligned arrays (N x T), gold_aligned (N) or None, and kept qids
    """
    tokens = list(token_dict.keys())
    # 공통 qid 교집합
    sets = [set(token_dict[t][0]) for t in tokens]
    common = set.intersection(*sets) if sets else set()
    qid_order = sorted(common)
    # 인덱스 맵
    idx_maps = {}
    for t in tokens:
        qids = token_dict[t][0]
        idx_maps[t] = {q:i for i,q in enumerate(qids)}

    preds_mat = []
    gold_vec = None
    for t in tokens:
        qids, preds, golds = token_dict[t]
        idx_map = idx_maps[t]
        aligned = [preds[idx_map[q]] for q in qid_order]
        preds_mat.append(aligned)
        if golds is not None:
            if gold_vec is None:
                gold_vec = [golds[idx_map[q]] for q in qid_order]
    preds_mat = np.array(preds_mat, dtype=int).T  # shape (N, T)
    gold_vec = (np.array(gold_vec, dtype=int) if gold_vec is not None else None)
    return preds_mat, gold_vec, qid_order

def agreement_matrix(preds_mat):
    """preds_mat: (N, T) -> (T, T) agreement ratio"""
    T = preds_mat.shape[1]
    M = np.zeros((T, T), dtype=float)
    for i in range(T):
        for j in range(T):
            M[i, j] = np.mean(preds_mat[:, i] == preds_mat[:, j]) if preds_mat.size else 0.0
    return M

def accuracies(preds_mat, gold_vec):
    if gold_vec is None:
        return None
    T = preds_mat.shape[1]
    acc = np.zeros(T, dtype=float)
    for i in range(T):
        acc[i] = np.mean(preds_mat[:, i] == gold_vec)
    return acc

def pair_deltas(preds_mat, gold_vec, i, j):
    """returns (agree, only_i_correct, only_j_correct, both_correct)"""
    agree = float(np.mean(preds_mat[:, i] == preds_mat[:, j]))
    if gold_vec is None:
        return agree, None, None, None
    pi = preds_mat[:, i]; pj = preds_mat[:, j]; g = gold_vec
    only_i = float(np.mean((pi == g) & (pj != g)))
    only_j = float(np.mean((pj == g) & (pi != g)))
    both   = float(np.mean((pi == g) & (pj == g)))
    return agree, only_i, only_j, both

def save_heatmap(M, tokens, title, out_png):
    plt.figure(figsize=(3.2, 2.8))
    plt.imshow(M, vmin=0.0, vmax=1.0)
    plt.xticks(range(len(tokens)), tokens)
    plt.yticks(range(len(tokens)), tokens)
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            plt.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center")
    plt.title(title)
    plt.colorbar()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="예: routes_out/token_only/pride_on/arc")
    ap.add_argument("--outdir", default="viz_out/matrix")
    ap.add_argument("--tokens", nargs="+", default=["T0","T1","T2"])
    args = ap.parse_args()

    models = [m for m in sorted(os.listdir(args.root)) if os.path.isdir(os.path.join(args.root, m))]
    os.makedirs(args.outdir, exist_ok=True)

    # 전체 요약 CSV
    summary_rows = []
    csv_path = os.path.join(args.outdir, "summary.csv")
    with open(csv_path, "w", encoding="utf-8") as w:
        header = ["model"] + [f"acc_{t}" for t in args.tokens] + \
                 [f"agree_{a}_{b}" for a in args.tokens for b in args.tokens if a<b] + \
                 [f"only_{a}_corr_vs_{b}" for a in args.tokens for b in args.tokens if a<b] + \
                 [f"only_{b}_corr_vs_{a}" for a in args.tokens for b in args.tokens if a<b]
        w.write(",".join(header) + "\n")

        for m in models:
            dir_model = os.path.join(args.root, m)
            token_dict = {}
            for t in args.tokens:
                token_dict[t] = load_preds(dir_model, t)  # (qids, preds, golds)
            preds_mat, gold_vec, _ = intersect_align(token_dict)
            if preds_mat.size == 0:
                print(f"[SKIP] no common qids: {m}")
                continue

            M = agreement_matrix(preds_mat)
            acc = accuracies(preds_mat, gold_vec)

            # 저장: heatmap (일치율)
            save_dir = os.path.join(args.outdir, m)
            os.makedirs(save_dir, exist_ok=True)
            save_heatmap(M, args.tokens, title=f"{m} agreement", out_png=os.path.join(save_dir, "agreement.png"))

            # 한 줄 요약 작성
            row = [m]
            if acc is None:
                row += [""]*len(args.tokens)
            else:
                row += [f"{a:.4f}" for a in acc]
            # pairwise
            only_A, only_B = [], []
            agree_pairs = []
            for i in range(len(args.tokens)):
                for j in range(i+1, len(args.tokens)):
                    ag, oi, oj, _ = pair_deltas(preds_mat, gold_vec, i, j)
                    agree_pairs.append(f"{ag:.4f}")
                    only_A.append("" if oi is None else f"{oi:.4f}")
                    only_B.append("" if oj is None else f"{oj:.4f}")
            row += agree_pairs + only_A + only_B
            w.write(",".join(row) + "\n")

    print(f"[DONE] Per-model heatmaps → {args.outdir}/<MODEL>/agreement.png")
    print(f"[DONE] Summary CSV       → {csv_path}")

if __name__ == "__main__":
    main()
