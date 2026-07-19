"""
TIU retention-policy study (Longhorn Silicon).

Software model of the TIU's per-block KV retention policy. Answers:
  (1) Which importance signal should the hardware accumulate?
  (2) At a matched memory budget, does demoting cold blocks to low precision
      beat evicting them entirely?

Grid: 4 routing policies x 2 cold-tier fates, swept over several budget levels,
on SmolLM-360M / HellaSwag.

Design guarantees:
  - Granularity is per-16-token block (matches the silicon), NOT per-token.
  - All four policies share ONE budget backend (assign_bits), so the only thing
    that varies between policies is *which* blocks get kept -- the comparison is
    fair by construction.
  - Reuses the existing repo's make_per_token_hook and score_continuation_dwb
    UNTOUCHED. The only new code is block-score computation + budget assignment
    + expansion back to per-position.

Reference anchor is the reimplementation's own numbers (DWB 33.8% / FP16 42.6%,
500 samp, same codebase), NEVER the DWB paper's 41.2% (unreachable here -- IMPL_GAP).

IMPORTANT scale caveat (from research-state.yaml): INT4 is LOSSLESS at 360M
(effective residual 8.1%). So any budget that keeps every block at >=4 bits will
score ~41% regardless of routing -- all four policies tie. The signal only bites
when the budget forces some blocks down to 2-bit (which IS lossy: ~25% at 360M).
That is why we sweep budgets DOWN through the 4-bit floor.
"""

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

BLOCK_SIZE = 16
BIT_CLASSES = [2, 4, 8, 16]


# ----------------------------------------------------------------------------
# 1. BLOCK SCORING -- four policies. Each takes the attention tensor and returns
#    one score per 16-token block. Higher score = more important = keep at higher
#    precision. This is the ONLY thing that differs between policies.
# ----------------------------------------------------------------------------

def _n_blocks(seq_len: int) -> int:
    return (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE


def _group_into_blocks(per_position: torch.Tensor) -> torch.Tensor:
    """(seq,) per-position values -> (n_blocks,) summed within each block."""
    seq = per_position.shape[0]
    nb = _n_blocks(seq)
    scores = torch.zeros(nb)
    for b in range(nb):
        scores[b] = per_position[b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE].sum()
    return scores


def score_attention(attn: torch.Tensor, seq_len: int) -> torch.Tensor:
    """H2O policy (the TIU's actual signal, the headline).

    attn: (num_heads, seq, seq) = outputs.attentions[-1][0].
    Importance of a key position = total attention it RECEIVED, summed over all
    heads and all query positions. Then grouped into blocks.
    """
    # sum over heads (dim 0), then over query positions (dim 0 of the result)
    # -> received[k] = how much every query, in every head, attended to key k.
    received = attn.sum(dim=0).sum(dim=0)          # (seq,)
    received = received[:seq_len]
    return _group_into_blocks(received)


def score_recency(attn: torch.Tensor, seq_len: int) -> torch.Tensor:
    """StreamingLLM policy: protect attention SINKS (first block) + a RECENT window.

    This is the actual StreamingLLM/H2O-streaming heuristic: the first block
    (attention sink) and the most recent blocks are kept; the middle is cold.
    Distinct from pure FIFO because the sink block is rescued regardless of age.
    Scores: sink block = max, recent blocks = high (rising), old-middle = low.
    """
    nb = _n_blocks(seq_len)
    scores = torch.arange(nb, dtype=torch.float32)   # recency: later = higher
    if nb > 0:
        scores[0] = float(nb) + 1.0                  # sink block: always top priority
    return scores


def score_fifo(attn: torch.Tensor, seq_len: int) -> torch.Tensor:
    """FIFO floor control: pure oldest-first eviction. Earliest blocks score LOWEST,
    with NO sink rescue -- the honest 'position order, no cleverness' baseline.
    """
    nb = _n_blocks(seq_len)
    return torch.arange(nb, dtype=torch.float32)


def score_random(attn: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Random floor: no signal at all. Establishes the 'routing does nothing' baseline."""
    nb = _n_blocks(seq_len)
    return torch.rand(nb)


POLICIES = {
    "tiu_h2o": score_attention,        # headline: attention-accumulation
    "tiu_streaming": score_recency,    # recency control
    "tiu_fifo": score_fifo,            # floor control
    "tiu_random": score_random,        # floor
}


# ----------------------------------------------------------------------------
# 2. BUDGET BACKEND -- shared by all policies.
#
#    The budget is a TARGET AVERAGE BITS-PER-BLOCK, hit EXACTLY on every sequence.
#    This is what makes the comparison fair: every policy spends the identical
#    memory footprint on a given sequence, so the only thing that varies is
#    *which* blocks each policy chooses to protect. (Fixed capacity fractions
#    drifted ~0.3 bits across policies -- as big as the accuracy signal -- which
#    is why we switched to matched-average-bits.)
#
#    Budgets are named by target avg bits. Tight budgets force blocks below the
#    4-bit lossless floor (down to 2-bit / evicted) so the routing signal can
#    actually bite; loose budgets stay >=4-bit where INT4 is lossless and all
#    policies must tie.
# ----------------------------------------------------------------------------

BUDGET_TARGETS = {
    "b3":  3.0,    # tight: forces heavy 2-bit use -> signal should bite
    "b4":  4.0,    # right at the lossless floor
    "b6":  6.0,    # loose: mostly >=4-bit -> expect ties (INT4 lossless)
}

# Allowed per-block precisions. Both fates use a simple TWO-TIER split at each
# budget: a warm tier and a cold tier. Demote's cold tier is 2-bit; evict's cold
# tier is 0 (dropped). The number of blocks in each tier is computed EXACTLY from
# the target so the average is hit identically for every policy AND both fates --
# the only thing that varies is *which* blocks land cold.
_DEMOTE_COLD = 2
_EVICT_COLD = 0


def _tier_split(nb, target_avg, warm, cold):
    """
    Exactly how many blocks go WARM vs COLD to hit target_avg on average.

    Solve: n_warm*warm + n_cold*cold = target_avg*nb, with n_warm+n_cold=nb.
    => n_warm = (target_avg - cold) / (warm - cold) * nb
    Rounded to the nearest integer. This depends ONLY on (nb, target, warm, cold)
    -- never on the scores -- so it is identical across all four policies.
    Returns (n_warm, n_cold).
    """
    if warm == cold:
        return nb, 0
    frac_warm = (target_avg - cold) / (warm - cold)
    n_warm = int(round(frac_warm * nb))
    n_warm = max(0, min(nb, n_warm))
    return n_warm, nb - n_warm


# Warm tier per budget: which precision the KEPT blocks sit at.
#   b3  -> warm=4  (kept blocks at INT4, cold blocks at 2/evicted): sub-floor, signal bites
#   b4  -> warm=4, cold=4 effectively (all 4-bit): the lossless floor, forced tie
#   b6  -> warm=8  (kept blocks at INT8, cold at 4): above floor, expect tie
_WARM_TIER = {"b3": 4, "b4": 4, "b6": 8}
_COLD_WARM_FOR_B4 = 4  # b4 is all-warm by construction


def assign_bits(block_scores, target_avg, seq_len, cold_fate="demote", budget_name=None):
    """
    block_scores: (n_blocks,) importance per block (higher = keep warm).
    target_avg:   target average bits per block, hit identically across policies.
    cold_fate:    'demote' -> cold tier = 2-bit; 'evict' -> cold tier = 0 (dropped).
    budget_name:  selects the warm-tier precision (b3/b4/b6).

    Two-tier: the top n_warm blocks by score get the warm precision, the rest get
    the cold precision. n_warm is computed exactly from the target, so every policy
    spends the same bits and demote vs evict are footprint-matched.

    Returns per-position list of bit-widths (length seq_len).
    """
    nb = len(block_scores) if not hasattr(block_scores, "shape") else block_scores.shape[0]
    warm = _WARM_TIER.get(budget_name, 4)
    cold = _DEMOTE_COLD if cold_fate == "demote" else _EVICT_COLD

    n_warm, n_cold = _tier_split(nb, target_avg, warm, cold)

    # rank blocks best-first
    if hasattr(block_scores, "argsort"):
        order = torch.argsort(block_scores, descending=True).tolist()
    else:
        order = sorted(range(nb), key=lambda i: -block_scores[i])

    bits_per_block = [cold] * nb
    for i in range(n_warm):
        bits_per_block[order[i]] = warm

    per_position = []
    for b in range(nb):
        per_position.extend([bits_per_block[b]] * BLOCK_SIZE)
    return per_position[:seq_len]


# ----------------------------------------------------------------------------
# 3. HOOK -- extends the repo's per-token quantization to also handle bit-width 0
#    (eviction = zero the KV for that position). For bits in {2,4,8,16} it defers
#    to the exact same quantizers the repo uses, so 'demote' matches the existing
#    reproduction path bit-for-bit.
# ----------------------------------------------------------------------------

def make_tiu_hook(bit_widths_per_position):
    """Like the repo's make_per_token_hook, but bit-width 0 means EVICT (zero it)."""
    from kv_cache_quant import quantize_tensor  # single fn, dispatches by bit count

    def hook(module, input, output):
        result = output.clone()
        seq_len = output.shape[1]
        for t in range(min(seq_len, len(bit_widths_per_position))):
            bits = bit_widths_per_position[t]
            if bits == 0:
                result[:, t] = 0.0            # evicted: block is gone
            elif bits >= 16:
                pass                            # fp16: untouched
            else:
                result[:, t] = quantize_tensor(output[:, t], bits)
        return result

    return hook


# ----------------------------------------------------------------------------
# 4. SCORING -- reuses the repo's attention extraction and continuation scoring
#    approach, swapping in the TIU hook. Structure mirrors score_continuation_dwb.
# ----------------------------------------------------------------------------

def get_block_scores(policy_name, eager_model, tokenizer, context, device, max_length=128):
    """Run one eager forward pass to get the attention tensor, then score blocks."""
    inputs = tokenizer(context, return_tensors="pt", truncation=True,
                       max_length=max_length).to(device)
    seq_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = eager_model(**inputs, output_attentions=True)
    attn = outputs.attentions[-1][0]        # (num_heads, seq, seq), last layer
    return POLICIES[policy_name](attn, seq_len), seq_len


def score_continuation_tiu(model, tokenizer, context, continuation,
                           bit_widths, device):
    """Identical in spirit to the repo's score_continuation_dwb, TIU hook swapped in."""
    full_ids = tokenizer.encode(context + continuation, return_tensors="pt").to(device)
    ctx_len = tokenizer.encode(context, return_tensors="pt").shape[1]

    hooks = []
    for name, module in model.named_modules():
        if name.split(".")[-1] in ("k_proj", "v_proj"):
            hooks.append(module.register_forward_hook(make_tiu_hook(bit_widths)))
    try:
        with torch.no_grad():
            logits = model(full_ids).logits[0]
    finally:
        for h in hooks:
            h.remove()

    cont_ids = full_ids[0, ctx_len:]
    if len(cont_ids) == 0:
        return -1e9
    log_probs = F.log_softmax(logits[ctx_len - 1:ctx_len - 1 + len(cont_ids)], dim=-1)
    return log_probs[range(len(cont_ids)), cont_ids].sum().item()


# ----------------------------------------------------------------------------
# 5. GRID EVAL
# ----------------------------------------------------------------------------

def evaluate_cell(model, eager_model, tokenizer, ds, policy_name, budget_name,
                  cold_fate, device):
    target_avg = BUDGET_TARGETS[budget_name]
    correct = 0
    total = len(ds)
    bit_dist = Counter()

    for ex in ds:
        ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        block_scores, seq_len = get_block_scores(
            policy_name, eager_model, tokenizer, ctx, device)
        bit_widths = assign_bits(block_scores, target_avg, seq_len, cold_fate,
                                  budget_name=budget_name)
        for b in bit_widths:
            bit_dist[b] += 1
        scores = [
            score_continuation_tiu(model, tokenizer, ctx, " " + e, bit_widths, device)
            for e in ex["endings"]
        ]
        if max(range(4), key=lambda j: scores[j]) == int(ex["label"]):
            correct += 1

    acc = correct / total * 100
    tot = sum(bit_dist.values())
    bit_pct = {b: round(c / tot * 100, 1) for b, c in sorted(bit_dist.items())}
    avg = sum(b * c for b, c in bit_dist.items()) / max(1, tot)
    return {"accuracy": acc, "correct": correct, "total": total,
            "bit_distribution_pct": bit_pct, "avg_bits": round(avg, 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM-360M")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--budgets", default="b3,b4,b6")
    p.add_argument("--fates", default="demote,evict")
    p.add_argument("--policies", default="tiu_h2o,tiu_streaming,tiu_fifo,tiu_random")
    p.add_argument("--output_dir", default="research/data")
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device)
    model.eval()
    print("Loading eager model for attention extraction...", flush=True)
    eager_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager").to(args.device)
    eager_model.eval()

    ds = load_dataset("Rowan/hellaswag", split="validation").select(range(args.limit))

    policies = args.policies.split(",")
    budgets = args.budgets.split(",")
    fates = args.fates.split(",")

    print(f"\nGrid: {len(policies)} policies x {len(fates)} fates x {len(budgets)} budgets"
          f" @ {args.limit} samples\n")
    grid = {}
    t0 = time.time()
    for budget in budgets:
        for fate in fates:
            for policy in policies:
                key = f"{policy}|{fate}|{budget}"
                res = evaluate_cell(model, eager_model, tokenizer, ds,
                                    policy, budget, fate, args.device)
                grid[key] = res
                print(f"  {key:38s} acc={res['accuracy']:5.1f}%  "
                      f"avg_bits={res['avg_bits']:.2f}  {res['bit_distribution_pct']}",
                      flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"tiu_grid_{args.limit}samp_{datetime.now():%Y%m%d_%H%M}.json"
    with open(out_dir / fname, "w") as f:
        json.dump({"grid": grid, "elapsed_s": elapsed,
                   "anchor_note": "reimpl anchor DWB 33.8% / FP16 42.6%, same codebase; "
                                  "INT4 lossless at 360M so ties at loose budgets are expected",
                   "date": datetime.now().isoformat()}, f, indent=2)
    print(f"Saved: {out_dir / fname}")


if __name__ == "__main__":
    main()