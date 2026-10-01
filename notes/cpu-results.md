# CPU checks

Python 3.12.3, PyTorch 2.14.1+cpu, x86_64, four threads, seed 42.
Three warmup runs and 20 measured repetitions.

| Attention, shape 1 x 4 x 128 x 32 | Median ms |
|---|---:|
| Dense | 0.236 |
| Python tiled | 1.476 |
| SDPA | 0.126 |

| Decoder generation | Median end-to-end ms |
|---|---:|
| Full context | 7.312 |
| Cached | 7.585 |

The decoder used one layer, width 32, context 32, a ten-byte prompt, and 16 output
tokens. It trained for 20 CPU steps on training abstracts. Greedy outputs matched.
At this size, cache overhead outweighed saved computation in this run.

The biomedical adapter check used two optimizer steps, rank eight, budget 64,
and 297,219 trainable parameters. Checkpoint reload and all 12 context conditions ran.

Frozen-encoder retrieval used 12 seeded test questions and a 36-paper index:

| Retriever | Recall at 5 | MRR at 5 |
|---|---:|---:|
| BM25 | 1.000 | 0.958 |
| Dense | 0.833 | 0.674 |
| Fused | 1.000 | 0.903 |

The same 12 questions against the full source index gave agent accuracy 0.750 and
macro-F1 0.286 after the two-step adapter check. The small sample and short training
make these execution checks. GPU measurements remain pending.
