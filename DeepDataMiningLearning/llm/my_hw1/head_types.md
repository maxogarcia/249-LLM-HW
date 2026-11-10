**Table 9. Attention-head types in the 4-layer character Transformer (RMSNorm + RoPE + SwiGLU), two training seeds. For each type, the strongest head in each seed and its average attention weight on the target position, per text; chance in parentheses = average of 1/(t+1) over the same query positions (uniform attention). Verdict: found if the weight is ≥ 3× chance on every text in both seeds; weak if the median over texts is ≥ 2× chance in both seeds; otherwise not found. Word-start and line-start skip queries where the target is t−1 (and word-start also t−2) so a previous-character or two-back head cannot score. n/a = the text has no newline. Texts: “ROMEO:⏎But soft! what light through”; “The bus was late again, so I walked to work.”; “def add(a, b):⏎    return a + b”; a 7-line Shakespeare dialogue (line-start only needs several lines).**

| head type | attends to | seed | head | Shakespeare | modern prose | code | multi-line Shakespeare | median × chance | lowest × chance | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| previous-character | t−1 | gpt_modern_s1337 | L0H1 | 0.56 (0.09) | 0.59 (0.08) | 0.50 (0.10) | 0.48 (0.06) | 6.8× | 5.0× | found |
| previous-character | t−1 | gpt_modern_s42 | L0H2 | 0.67 (0.09) | 0.61 (0.08) | 0.47 (0.10) | 0.50 (0.06) | 7.5× | 4.7× | found |
| two-back | t−2 | gpt_modern_s1337 | L0H3 | 0.39 (0.08) | 0.41 (0.07) | 0.41 (0.09) | 0.33 (0.05) | 5.4× | 4.7× | found |
| two-back | t−2 | gpt_modern_s42 | L0H1 | 0.34 (0.08) | 0.40 (0.07) | 0.40 (0.09) | 0.33 (0.05) | 5.2× | 4.2× | found |
| word-start | first letter of the current word (4th letter on) | gpt_modern_s1337 | L1H3 | 0.32 (0.08) | 0.03 (0.04) | 0.50 (0.04) | 0.26 (0.06) | 4.4× | 0.8× | weak |
| word-start | first letter of the current word (4th letter on) | gpt_modern_s42 | L0H0 | 0.12 (0.08) | 0.15 (0.04) | 0.15 (0.04) | 0.17 (0.06) | 3.3× | 1.5× | weak |
| line-start | most recent newline (2nd char of the line on) | gpt_modern_s1337 | L3H2 | 0.53 (0.05) | n/a | 0.31 (0.04) | 0.42 (0.04) | 10.1× | 7.3× | found |
| line-start | most recent newline (2nd char of the line on) | gpt_modern_s42 | L1H0 | 0.49 (0.05) | n/a | 0.53 (0.04) | 0.20 (0.04) | 9.3× | 5.6× | found |
| first-position | position 0 of the text | gpt_modern_s1337 | L3H2 | 0.11 (0.09) | 0.40 (0.08) | 0.08 (0.10) | 0.06 (0.06) | 1.1× | 0.8× | not found |
| first-position | position 0 of the text | gpt_modern_s42 | L1H2 | 0.11 (0.09) | 0.10 (0.08) | 0.10 (0.10) | 0.06 (0.06) | 1.1× | 1.0× | not found |
