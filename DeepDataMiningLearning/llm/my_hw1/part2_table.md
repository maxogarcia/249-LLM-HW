**Table 1. Character-level Tiny Shakespeare, 16.4M training tokens per run, 2 seeds (1337, 42). Val bpb is the final value on the full validation set (mean ± std over seeds). Δ and paired p are vs. the GPT-2-style Transformer. Seed noise floor = 0.0042 bpb; differences smaller than 2× that (0.0084) are marked not resolved. Train bpb = mean of the last 5 logged training losses (last 250 steps) ÷ ln 2. Gap = final val − train. FLOPs ≈ 6 × params × tokens.**

| model | params (M) | train FLOPs (PFLOP) | final val bpb | Δ vs GPT-2-style | paired p | resolved? | best val bpb | best at (M tokens) | train bpb | val − train gap | train time (min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transformer 4×256, RMSNorm + RoPE + SwiGLU | 3.16 | 0.31 | 2.1082 ± 0.0079 | -0.0711 | 0.023 | yes | 2.1057 | 15.4 | 1.739 | +0.369 | 1.24 |
| LSTM 2×256 | 1.07 | 0.11 | 2.1384 ± 0.0008 | -0.0409 | 0.039 | yes | 2.1384 | 16.4 | 1.863 | +0.276 | 1.60 |
| GRU 2×256 | 0.81 | 0.08 | 2.1631 ± 0.0041 | -0.0162 | 0.007 | yes | 2.1631 | 16.4 | 1.943 | +0.220 | 1.29 |
| LSTM 2×512 | 4.24 | 0.42 | 2.1745 ± 0.0067 | -0.0049 | 0.645 | NOT resolved | 2.1276 | 9.2 | 1.488 | +0.687 | 2.03 |
| Transformer 4×256, LayerNorm + learned pos + GELU | 3.44 | 0.34 | 2.1793 ± 0.0043 | 0 (baseline) | — | — | 2.1793 | 16.4 | 1.961 | +0.219 | 1.03 |
| Vanilla RNN 2×256 | 0.28 | 0.03 | 2.4378 ± 0.0040 | +0.2584 | 0.014 | yes | 2.4378 | 16.4 | 2.349 | +0.089 | 0.18 |

**Table 2. Pairwise comparisons for the report questions (paired over the same seeds).**

| question | A | B | Δ (B − A) bpb | paired p | resolved? |
|---|---:|---:|---:|---:|---:|
| Is the larger LSTM better? | LSTM 2×256 | LSTM 2×512 | +0.0360 | 0.093 | yes |
| GRU vs LSTM (same width) | LSTM 2×256 | GRU 2×256 | +0.0247 | 0.060 | yes |
| Do gates matter? | LSTM 2×256 | Vanilla RNN 2×256 | +0.2993 | 0.007 | yes |
| Does the modern recipe matter? | Transformer 4×256, LayerNorm + learned pos + GELU | Transformer 4×256, RMSNorm + RoPE + SwiGLU | -0.0711 | 0.023 | yes |
| Best RNN vs best Transformer | LSTM 2×256 | Transformer 4×256, RMSNorm + RoPE + SwiGLU | -0.0302 | 0.106 | yes |

**Table 3. Per-seed detail.**

| run | seed | final val bpb | best val bpb | best at (M tokens) | train bpb | gap |
|---|---:|---:|---:|---:|---:|---:|
| gpt_modern | 42 | 2.1138 | 2.1138 | 16.4 | 1.745 | +0.369 |
| gpt_modern | 1337 | 2.1026 | 2.0976 | 14.3 | 1.734 | +0.369 |
| gpt_textbook | 42 | 2.1824 | 2.1824 | 16.4 | 1.966 | +0.217 |
| gpt_textbook | 1337 | 2.1763 | 2.1763 | 16.4 | 1.955 | +0.221 |
| gru | 42 | 2.1660 | 2.1660 | 16.4 | 1.943 | +0.223 |
| gru | 1337 | 2.1603 | 2.1603 | 16.4 | 1.942 | +0.218 |
| lstm | 42 | 2.1390 | 2.1390 | 16.4 | 1.865 | +0.274 |
| lstm | 1337 | 2.1379 | 2.1379 | 16.4 | 1.861 | +0.277 |
| lstm_512 | 42 | 2.1697 | 2.1303 | 10.2 | 1.489 | +0.681 |
| lstm_512 | 1337 | 2.1792 | 2.1249 | 8.2 | 1.487 | +0.692 |
| rnn | 42 | 2.4349 | 2.4349 | 16.4 | 2.348 | +0.087 |
| rnn | 1337 | 2.4406 | 2.4406 | 16.4 | 2.350 | +0.091 |
