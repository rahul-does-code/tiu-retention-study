# TIU Retention Policy Study

A software model of a hardware KV-cache retention policy: which cached token blocks
to keep at high precision, and which to compress or evict, under a fixed memory budget.

**Headline result:** on short-context benchmarks (HellaSwag) the choice of retention
policy is undetectable — all policies tie, because 4-bit KV is lossless at this model
scale. On long context (WikiText-2, 1024–2048 tokens), the choice is decisive:
attention-accumulation routing beats a random baseline by 2–4× perplexity and beats
naive FIFO eviction by ~40×. The right cold-block fate (evict vs. demote) is not fixed
— it depends on routing quality, which yields a conditional hardware recommendation.

Full analysis, tables, and limitations are in [`tiu_h1_findings.md`](tiu_h1_findings.md).

## What's here

- `src/tiu_policies.py` — four routing policies (attention-accumulation, streaming,
  FIFO, random) sharing one matched-budget backend; HellaSwag grid
- `src/tiu_longcontext.py` — WikiText-2 long-context perplexity grid
- `data/*.json` — final result grids (HellaSwag 500-sample; long-context 1024 & 2048 tokens)
- `tiu_h1_findings.md` — the writeup

## Basis and attribution

Built on [`LonghornSilicon/dont-waste-bits`](https://github.com/LonghornSilicon/dont-waste-bits)
(MIT), an independent re-implementation of adaptive KV-cache quantization. This study
reuses that repo's quantization hooks, quantizers, and evaluation harness unchanged;
the new contribution is the block-level routing policies, the matched-budget allocator,
and the long-context evaluation. Anchored throughout against that codebase's own
reproduced numbers — not against any external paper's reported figures.

## Reproducing

Requires the `dont-waste-bits` environment (PyTorch, transformers, datasets) plus a
CUDA GPU. Place `src/tiu_policies.py` and `src/tiu_longcontext.py` in that repo's
`research/src/`, then:

```
python research/src/tiu_policies.py --limit 500 --budgets b3,b4,b6 --device cuda
python research/src/tiu_longcontext.py --seq_len 1024 --n_chunks 30 --device cuda
python research/src/tiu_longcontext.py --seq_len 2048 --n_chunks 15 --device cuda
```

All runs completed on a single 6GB GPU (RTX 4050).

## Authorship

Study designed, run, and validated by Rahul (`rahul-does-code`). Code and analysis
developed with AI assistance (used as a tool, as one would use any library or
framework); all experiments were executed locally and all results independently
checked against in-codebase anchors.
