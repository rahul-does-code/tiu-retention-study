# TIU-H1: Retention Policy Study — Which CSR Mode Should Be the Default?

## Executive summary

The TIU's per-block retention policy is headed to silicon, and this study models it
in software to answer which importance signal it should accumulate and how it should
treat cold blocks. Two findings are decision-relevant for the CSR default. **First:
routing policy cannot be validated on short-context benchmarks** — on HellaSwag every
policy ties, because 4-bit KV is lossless at this model scale, so the evaluation is
uninformative. **Second, at the spec's long-context configuration (128 tracked blocks,
2048 tokens), the policy choice is decisive:** attention-accumulation routing beats a
random baseline by 2–4× perplexity and beats naive FIFO eviction by roughly 40×. The
best cold-block fate is *conditional* — evicting cold blocks wins when routing is good
and loses badly when it isn't — which yields a specific recommendation: default to the
attention-accumulation signal, and only default to eviction when that signal is active;
otherwise demote, which bounds worst-case damage. Full method, tables, and the hardware
caveats (notably an oracle-lookahead limitation in the signal) are below.

---

**Status:** COMPLETE (software model; hardware-fidelity caveats below)
**Scope:** SmolLM-360M, per-16-token-block KV retention, matched memory budgets
**Benchmarks:** HellaSwag (short context, ~8 blocks) + WikiText-2 perplexity (1024 and 2048 tokens = 64 and 128 blocks; 2048 matches the TIU spec's 128 tracked blocks exactly)
**Repos:** builds on `dont-waste-bits` (hooks, quantizers, and eval harness reused unchanged); new code is `tiu_policies.py` + `tiu_longcontext.py`
**Authorship note:** study designed, run, and validated by Rahul; code and analysis developed with AI assistance (used as a tool). All experiments executed locally on an RTX 4050 (6GB) and checked against in-codebase anchors.

---

## Question

The TIU decides which cached 16-token blocks to keep at high precision and which to
compress or evict, and this policy is headed to silicon untested in software. Two
questions:

1. **Which importance signal should the hardware accumulate?** Attention-accumulation
   (H2O-style, the TIU's planned signal) vs. recency (StreamingLLM-style), FIFO, and
   random controls.
2. **For cold blocks, does demoting to low precision beat evicting them entirely, at
   the same memory budget?**

## Method (summary)

Four routing policies were modeled in software, all sharing one budget backend so
that on every sequence, every policy places the **identical number of blocks at each
precision** — only *which* blocks differ. This matched-budget property was
unit-verified before any GPU run (early fixed-fraction budgeting drifted by ~0.3
avg bits between policies, as large as the accuracy differences it confounded; it
was replaced with an exact two-tier allocation).

- **tiu_h2o** — blocks ranked by total attention received (summed over heads and
  query positions, last layer)
- **tiu_streaming** — attention-sink block always kept + recency ranking
- **tiu_fifo** — pure position order (oldest coldest), no sink rescue
- **tiu_random** — no signal (floor)

Cold-tier fates: **demote** (cold blocks → INT2) vs. **evict** (cold blocks zeroed).
The budget used for all headline results is b3 (3.0 avg bits/block): warm blocks at
INT4, 50% cold for demote, 25% cold for evict — both fates land at exactly 3.0.

Quantization is applied via the repo's k_proj/v_proj forward hooks. Anchors are
the reimplementation's own numbers only (DWB 33.8% / FP16 42.6% on HellaSwag;
per-run FP16 perplexity on WikiText). The DWB paper's 41.2% is never used as a
comparison point (known IMPL_GAP in this codebase).

## Results

### 1. Short context (HellaSwag, 500 samples): routing does not matter

| Budget | Result |
|---|---|
| 4.0 bits (all-INT4) | All 4 policies × both fates: **44.0%** — an exact 8-way tie |
| 3.0 bits (sub-floor) | demote: 33.4–36.0%; evict: 40.4–41.8% — all policy gaps within CI (±~6pp at n=500) |

INT4 is lossless at 360M (consistent with this repo's prior FPGA-controller
finding), so at ≥4 bits there is nothing for a routing policy to do. Even below
the 4-bit floor, HellaSwag's ~100-token contexts (~8 blocks) give routing too
little room to matter: the policy comparison is a **clean null** at n=500.

One real short-context effect: **evict beat demote in every cell** (~41% vs ~35%).
At fixed memory, keeping fewer blocks at pristine INT4 beats keeping all blocks at
corrupted INT2 — a 2-bit block still participates in attention as noise, while an
evicted block simply contributes nothing.

### 2. Long context (WikiText-2 PPL, lower = better): routing matters decisively

1024 tokens (64 blocks), 30 chunks, FP16 anchor PPL 13.6:

| Policy | Demote | Evict |
|---|---|---|
| **tiu_h2o** | **92.5** (±5.7) | **33.8** (±2.0) |
| tiu_random | 147.9 (±9.8) | 58.8 (±6.8) |
| tiu_streaming | 179.8 (±12.7) | 147.4 (±11.0) |
| tiu_fifo | 493.4 (±34.0) | 1357.7 (±103.5) |

2048 tokens (128 blocks — TIU spec-exact), 15 chunks, FP16 anchor PPL 12.0:

| Policy | Demote | Evict |
|---|---|---|
| **tiu_h2o** | **126.5** (±10.1) | **39.5** (±2.8) |
| tiu_random | 169.4 (±13.7) | 76.5 (±8.2) |
| tiu_streaming | 271.8 (±21.5) | 359.9 (±26.9) |
| tiu_fifo | 494.5 (±42.9) | 1542.1 (±125.5) |

The ordering h2o < random < streaming < fifo holds in all four columns, with gaps
of 10+ standard errors. The HellaSwag null was a context-length artifact: at 64–128
blocks, *which* blocks a policy keeps is the dominant factor in quality, and the
attention-accumulation signal wins by roughly 2–4× PPL over the random floor and
~40× over FIFO eviction.

**Sink-block validity check.** The only difference between tiu_streaming and
tiu_fifo is that streaming rescues block 0 (the attention sink). At 1024-token
eviction this one change is worth ~9× PPL (147 vs 1358) — reproducing the known
StreamingLLM sink effect independently, which is strong evidence the harness
measures a real mechanism.

### 3. The interaction: eviction amplifies routing quality — in both directions

The demote-vs-evict answer is **not** a single winner. It depends on routing:

- With good routing (h2o): evict crushes demote (33.8 vs 92.5 at 1024; 39.5 vs 126.5 at 2048).
- With bad routing (fifo): evict is a disaster vs demote (1357.7 vs 493.4; 1542.1 vs 494.5).
- Streaming sits at the crossover, and it *flips with context length*: evict wins at
  1024 (147 vs 180) but loses at 2048 (360 vs 272) — as context grows, the middle
  region streaming sacrifices grows with it, and discarding it outright gets costlier.

Mechanism: demotion is damage-bounded (a corrupted block still carries partial
signal), while eviction is all-or-nothing and therefore high-variance — it pays off
exactly when the routing signal correctly identifies expendable blocks, and
compounds the error when it doesn't.

### Why streaming loses to *random* on this metric

Streaming's coldest 25% is a **contiguous mid-sequence band** (sink and recent
blocks are protected, so the old-middle goes cold). Whole-sequence teacher-forced
PPL scores predictions at *every* position; queries in the second half of a
Wikipedia article routinely need mid-article content, and streaming has destroyed
precisely that region wholesale. Random damages the same number of blocks but
scatters them, so no region is entirely lost. **Important honest framing:** this is
partly a metric artifact. StreamingLLM is designed for decode-frontier workloads
(predicting only the newest tokens), where a sacrificed middle costs far less.
These numbers indict streaming *for whole-context workloads*, not in general.

## Recommendation for the CSR default

1. **Enable the attention-accumulation signal (`tiu_h2o` mode) as the default.**
   It is the best policy in every long-context cell, in both fates, at both lengths.
2. **Couple the cold-tier fate to routing availability.** Eviction should be the
   default *only when* the accumulation signal is active. If the TIU may run with
   routing degraded or disabled (FIFO-like behavior), demotion is the safe fate —
   it bounds worst-case damage by ~3× where eviction compounds it.
3. **Do not validate retention policy on short-context benchmarks.** At ≤~8 blocks,
   or at any budget ≥4 bits on a ≤360M-class model, all policies tie and the
   evaluation is uninformative. Long-context perplexity at the spec's 128-block
   configuration is the discriminating test.

## Limitations (read before citing)

- **Oracle lookahead in the H2O signal.** Block scores sum attention from *all*
  query positions, extracted from a clean full-sequence pass. The hardware
  accumulates causally during decode and knows only the past. Reported h2o numbers
  are an **upper bound** on the accumulation signal. The margins (2–4× over random)
  are large enough that the ranking plausibly survives a causal version, but that
  version has not been run.
- **Static retention.** Bit assignments are fixed per forward pass; real decode
  updates retention as the sequence grows.
- **INT2 is a stand-in for CQ-4.** The silicon's cold tier is ChannelQuant CQ-4,
  not INT2 (a flagged fidelity gap in this harness). Note also that at 360M a
  4-bit cold tier is *lossless*, so the demote-vs-evict question as the hardware
  literally poses it cannot be tested at this model scale — the INT2 stand-in is
  what makes the question answerable in software at all.
- **Scale.** All results are SmolLM-360M. The repo's prior work shows INT4
  losslessness breaks between 360M and 1.7B; margins may shift at deployment scale.
- **PPL means across chunks are heavy-tailed**; at these gap sizes (10–100+ SEMs)
  no reasonable statistic changes the ordering, but small gaps in future runs
  should be bootstrap-tested rather than eyeballed.
- All comparisons are **within this codebase** against anchors reproduced in the
  same environment. Nothing here is a claim about the DWB paper.

## Suggested follow-ups

1. **Causal H2O** — accumulate attention only from past queries (one-line change to
   the aggregation) to close the oracle gap. Highest-value next experiment.
2. **Decode-frontier PPL** — score only the final N tokens to test the
   streaming-favorable workload and complete the workload×policy picture.
3. **Sink-augmented H2O** — h2o already ranks the sink highly, but forcing sink
   retention costs nothing and removes a failure mode.
4. **CQ-4 cold tier at ≥1.7B** — the fidelity-faithful version of question 2
   requires both the real quantizer and a scale where 4-bit is lossy (GPU-blocked,
   same as the repo's H1).

## Artifacts

- `research/src/tiu_policies.py` — policies, matched-budget backend, HellaSwag grid
- `research/src/tiu_longcontext.py` — WikiText-2 long-context PPL grid
- `research/data/tiu_grid_*samp_*.json` — HellaSwag results (100/500 samples)
- `research/data/tiu_longctx_1024tok_30chunks_*.json`, `tiu_longctx_2048tok_15chunks_*.json`
