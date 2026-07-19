"""
TIU retention policies -- LONG-CONTEXT evaluation (WikiText-2 perplexity).

Why this exists: the HellaSwag grid (tiu_policies.py) returned a clean null on
the policy comparison at 500 samples -- routing choice does not separate at 360M
on short contexts, even below the 4-bit lossless floor. HellaSwag contexts are
~100 tokens (~8 blocks); importance-based retention is designed for LONG contexts
where most of the cache is cold. This script asks the same question where the
signal is supposed to live: 1024-2048 token sequences (64-128 blocks -- at 2048
this exactly matches the TIU spec's 128 tracked blocks).

Metric: token-level perplexity (teacher-forced) on WikiText-2 test chunks.
Continuous metric -> far more sensitive than HellaSwag's binary accuracy.
Every run also reports the unquantized FP16 PPL on the same chunks as anchor.

Reuses POLICIES / assign_bits / BUDGET_TARGETS from tiu_policies.py unchanged --
same policies, same matched-budget backend, same fates. Only the benchmark differs.

Anchoring discipline: compare cells only against each other and the FP16 anchor
computed in the same run on the same chunks. No external targets.
"""

import argparse
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))

from tiu_policies import (BLOCK_SIZE, BUDGET_TARGETS, POLICIES, assign_bits)


# ----------------------------------------------------------------------------
# Vectorized per-position quantization hook.
# Mathematically identical to the per-token loop (batch=1: per-position scale
# over the feature dim), but masked per bit-class instead of a Python loop over
# 1024+ positions x 64 modules -- required to make long-context runs tractable.
# ----------------------------------------------------------------------------

def make_fast_tiu_hook(bit_widths_per_position):
    bits_t = torch.tensor(bit_widths_per_position)

    def hook(module, input, output):
        # output: (batch, seq_len, d)
        seq_len = output.shape[1]
        n = min(seq_len, len(bit_widths_per_position))
        result = output.clone()
        bits_here = bits_t[:n].to(output.device)

        for bits in torch.unique(bits_here).tolist():
            if bits >= 16:
                continue                       # fp16: untouched
            mask = (bits_here == bits)         # (n,)
            if bits == 0:
                result[:, :n][:, mask] = 0.0   # evicted
                continue
            x = output[:, :n][:, mask]         # (batch, n_sel, d)
            levels = 2 ** (bits - 1) - 1
            # per-position scale over batch+features (batch=1 -> feature amax)
            scale = x.abs().amax(dim=(0, 2), keepdim=True) / max(levels, 1)
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            q = (x / scale).round().clamp(-levels - 1, levels) * scale
            result[:, :n][:, mask] = q
        return result

    return hook


# ----------------------------------------------------------------------------
# Chunked WikiText-2
# ----------------------------------------------------------------------------

def load_wikitext_chunks(tokenizer, seq_len, n_chunks):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    chunks = []
    for i in range(n_chunks):
        start = i * seq_len
        if start + seq_len > len(ids):
            break
        chunks.append(ids[start:start + seq_len])
    return chunks


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def block_scores_for_chunk(policy_name, eager_model, chunk_ids, device):
    """One eager pass -> last-layer attention -> block scores for this policy."""
    input_ids = chunk_ids.unsqueeze(0).to(device)
    seq_len = input_ids.shape[1]
    with torch.no_grad():
        out = eager_model(input_ids, output_attentions=True)
    attn = out.attentions[-1][0].float()      # (heads, seq, seq)
    scores = POLICIES[policy_name](attn, seq_len)
    del out, attn
    if device == "cuda":
        torch.cuda.empty_cache()
    return scores, seq_len


def ppl_with_bits(model, chunk_ids, bit_widths, device):
    """Teacher-forced perplexity over the chunk with TIU quantization applied."""
    input_ids = chunk_ids.unsqueeze(0).to(device)
    hooks = []
    for name, module in model.named_modules():
        if name.split(".")[-1] in ("k_proj", "v_proj"):
            hooks.append(module.register_forward_hook(make_fast_tiu_hook(bit_widths)))
    try:
        with torch.no_grad():
            logits = model(input_ids).logits[0]        # (seq, vocab)
    finally:
        for h in hooks:
            h.remove()
    # predict token t+1 from position t
    ce = F.cross_entropy(logits[:-1], input_ids[0, 1:], reduction="mean")
    return ce.item()


def ppl_fp16_anchor(model, chunk_ids, device):
    input_ids = chunk_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]
    ce = F.cross_entropy(logits[:-1], input_ids[0, 1:], reduction="mean")
    return ce.item()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM-360M")
    p.add_argument("--seq_len", type=int, default=1024,
                   help="1024 = 64 blocks (fits 6GB). 2048 = 128 blocks (TIU spec-exact; may OOM).")
    p.add_argument("--n_chunks", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--budgets", default="b3")
    p.add_argument("--fates", default="demote,evict")
    p.add_argument("--policies", default="tiu_h2o,tiu_streaming,tiu_fifo,tiu_random")
    p.add_argument("--output_dir", default="research/data")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16).to(args.device)
    model.eval()
    print("Loading eager model (fp16) for attention extraction...", flush=True)
    eager_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="eager").to(args.device)
    eager_model.eval()

    print(f"Loading WikiText-2 chunks (seq_len={args.seq_len}, n={args.n_chunks})...",
          flush=True)
    chunks = load_wikitext_chunks(tokenizer, args.seq_len, args.n_chunks)
    print(f"  {len(chunks)} chunks of {args.seq_len} tokens "
          f"({args.seq_len // BLOCK_SIZE} blocks each)", flush=True)

    policies = args.policies.split(",")
    budgets = args.budgets.split(",")
    fates = args.fates.split(",")

    # FP16 anchor on the same chunks
    print("\nFP16 anchor (no quantization)...", flush=True)
    fp16_ces = [ppl_fp16_anchor(model, c, args.device) for c in chunks]
    fp16_ppl = math.exp(sum(fp16_ces) / len(fp16_ces))
    print(f"  FP16 PPL = {fp16_ppl:.2f}", flush=True)

    # Precompute block scores per (policy, chunk) -- one eager pass per pair.
    # (Attention scores are policy-independent; positional policies don't even
    # use them -- but we recompute per policy for simplicity. The eager pass is
    # only needed for tiu_h2o; skip it for the rest.)
    print("\nExtracting attention scores for tiu_h2o...", flush=True)
    h2o_scores = {}
    if "tiu_h2o" in policies:
        for ci, chunk in enumerate(chunks):
            h2o_scores[ci], _ = block_scores_for_chunk(
                "tiu_h2o", eager_model, chunk, args.device)
            if (ci + 1) % 10 == 0:
                print(f"  {ci + 1}/{len(chunks)}", flush=True)
    del eager_model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    grid = {}
    t0 = time.time()
    print(f"\nGrid: {len(policies)} policies x {len(fates)} fates x "
          f"{len(budgets)} budgets @ {len(chunks)} chunks\n", flush=True)

    torch.manual_seed(0)  # fix tiu_random's draws for reproducibility

    for budget in budgets:
        target = BUDGET_TARGETS[budget]
        for fate in fates:
            for policy in policies:
                ces = []
                bit_dist = Counter()
                for ci, chunk in enumerate(chunks):
                    seq_len = chunk.shape[0]
                    if policy == "tiu_h2o":
                        scores = h2o_scores[ci]
                    else:
                        # positional/random policies never read the attention tensor
                        scores = POLICIES[policy](None, seq_len)
                    bit_widths = assign_bits(scores, target, seq_len, fate,
                                             budget_name=budget)
                    for b in bit_widths:
                        bit_dist[b] += 1
                    ces.append(ppl_with_bits(model, chunk, bit_widths, args.device))
                mean_ce = sum(ces) / len(ces)
                ppl = math.exp(mean_ce)
                # per-chunk PPL std for CI reading
                chunk_ppls = [math.exp(c) for c in ces]
                mean_ppl_chunks = sum(chunk_ppls) / len(chunk_ppls)
                var = sum((x - mean_ppl_chunks) ** 2 for x in chunk_ppls) / max(1, len(chunk_ppls) - 1)
                sem = math.sqrt(var / len(chunk_ppls))
                tot = sum(bit_dist.values())
                bit_pct = {b: round(c / tot * 100, 1) for b, c in sorted(bit_dist.items())}
                avg = sum(b * c for b, c in bit_dist.items()) / max(1, tot)
                key = f"{policy}|{fate}|{budget}"
                grid[key] = {"ppl": round(ppl, 3),
                             "ppl_sem_across_chunks": round(sem, 3),
                             "delta_vs_fp16": round(ppl - fp16_ppl, 3),
                             "bit_distribution_pct": bit_pct,
                             "avg_bits": round(avg, 2),
                             "n_chunks": len(chunks)}
                print(f"  {key:34s} ppl={ppl:8.2f} (+{ppl - fp16_ppl:6.2f} vs fp16)"
                      f"  avg_bits={avg:.2f}  {bit_pct}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. FP16 anchor PPL = {fp16_ppl:.2f}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"tiu_longctx_{args.seq_len}tok_{len(chunks)}chunks_{datetime.now():%Y%m%d_%H%M}.json"
    with open(out_dir / fname, "w") as f:
        json.dump({"grid": grid, "fp16_ppl": round(fp16_ppl, 3),
                   "seq_len": args.seq_len, "n_chunks": len(chunks),
                   "elapsed_s": elapsed,
                   "note": "Long-context companion to the HellaSwag grid. PPL lower=better. "
                           "Compare cells only within this run (same chunks, same anchor).",
                   "date": datetime.now().isoformat()}, f, indent=2)
    print(f"Saved: {out_dir / fname}")


if __name__ == "__main__":
    main()
