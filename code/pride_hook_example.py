
"""
pride_hook_example.py

Drop the core pieces into eval_clm.py to enable: --pride "method=paraphrase,k=3,seed=42"

Requires your existing debias_pride.py to expose a function like:
    generate_pride_variants(prompt: str, method: str, k: int, seed: int) -> list[str]

If your API is different, just edit the wrapper below.
"""

import re

def parse_pride_cfg(cfg_str: str):
    """
    "method=paraphrase,k=3,seed=42" -> {"method":"paraphrase","k":3,"seed":42}
    """
    out = {"method": "paraphrase", "k": 3, "seed": 42}
    if not cfg_str:
        return out
    for kv in cfg_str.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            k = k.strip(); v = v.strip()
            if k in ("k", "seed"):
                try:
                    out[k] = int(v)
                except:
                    pass
            else:
                out[k] = v
    return out

def build_prompts_with_pride(base_prompt: str, cfg: str):
    """
    Returns [base_prompt] + (k-1) PriDe variants
    """
    try:
        from debias_pride import generate_pride_variants
    except Exception as e:
        # Fallback: no PriDe available -> just return [base_prompt]
        return [base_prompt]

    opt = parse_pride_cfg(cfg)
    method = opt.get("method", "paraphrase")
    k = int(opt.get("k", 3))
    seed = int(opt.get("seed", 42))

    variants = generate_pride_variants(base_prompt, method=method, k=k, seed=seed)
    out = [base_prompt] + variants[:max(0, k-1)]
    return out

# Example usage in eval loop
# ---------------------------------------
# if args.pride:
#     prompts = build_prompts_with_pride(prompt, args.pride)
# else:
#     prompts = [prompt]
#
# # then score each prompt, average the probabilities before deciding
# all_probs = []
# for pr in prompts:
#     probs = run_model(pr)  # (C,) normalized
#     all_probs.append(probs)
# P = np.stack(all_probs, axis=0).mean(axis=0)
# pred = int(np.argmax(P))
# ---------------------------------------
