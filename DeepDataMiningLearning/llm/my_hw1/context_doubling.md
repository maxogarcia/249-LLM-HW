**Table 4. Change in validation bpb per doubling of context length (negative = better), paired over seeds 1337 and 42. Seed noise floor = 0.0046 bpb (median within-config std); a gain smaller than 2× that (0.0093) is marked not resolved.**

| model | doubling | bpb before | bpb after | Δ bpb | paired p | |Δ| / noise | resolved? |
|---|---|---:|---:|---:|---:|---:|---:|
| Transformer (modern) 4×256 | 64 → 128 | 2.1537 | 2.1180 | -0.0358 | 0.227 | 7.7× | yes |
| Transformer (modern) 4×256 | 128 → 256 | 2.1180 | 2.1105 | -0.0074 | 0.722 | 1.6× | NOT resolved |
| Transformer (modern) 4×256 | 256 → 512 | 2.1105 | 2.1065 | -0.0040 | 0.434 | 0.9× | NOT resolved |
| LSTM 2×256 | 64 → 128 | 2.1889 | 2.1543 | -0.0346 | 0.049 | 7.5× | yes |
| LSTM 2×256 | 128 → 256 | 2.1543 | 2.1387 | -0.0157 | 0.130 | 3.4× | yes |
| LSTM 2×256 | 256 → 512 | 2.1387 | 2.1239 | -0.0148 | 0.169 | 3.2× | yes |
