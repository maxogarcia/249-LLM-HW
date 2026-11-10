**Table 6. Strongest head of each type (average attention weight on the relevant position) for each text, in the two seeds of the modern Transformer. Chance = average of 1/(t+1) over the same queries (uniform attention over the allowed positions). Word-start and line-start exclude the query right after the target, where it is also t−1.**

| head type: attends to | text | gpt_modern_s1337 | gpt_modern_s42 | chance | # queries |
|---|---|---|---|---|---|
| previous character | shakespeare | L0H1: 0.56 | L0H2: 0.67 | 0.093 | 34 |
| previous character | prose | L0H1: 0.59 | L0H2: 0.61 | 0.078 | 43 |
| previous character | code | L0H1: 0.50 | L3H2: 0.48 | 0.101 | 30 |
| first letter of current word | shakespeare | L1H1: 0.40 | L2H3: 0.37 | 0.087 | 16 |
| first letter of current word | prose | L3H3: 0.42 | L0H1: 0.46 | 0.072 | 14 |
| first letter of current word | code | L1H3: 0.58 | L0H1: 0.39 | 0.108 | 6 |
| first letter of current word (4th letter on) | shakespeare | L1H3: 0.32 | L2H3: 0.27 | 0.076 | 10 |
| first letter of current word (4th letter on) | prose | L1H1: 0.25 | L2H3: 0.23 | 0.038 | 7 |
| first letter of current word (4th letter on) | multiline | L1H3: 0.26 | L1H2: 0.24 | 0.056 | 15 |
| most recent newline | multiline | L3H2: 0.42 | L1H0: 0.20 | 0.036 | 53 |
| most recent newline | shakespeare | L3H2: 0.53 | L1H0: 0.49 | 0.053 | 27 |
| most recent newline | code | L3H2: 0.31 | L1H0: 0.53 | 0.043 | 15 |

**Table 7. The strongest head of each type in gpt_modern_s1337, measured on every text. n/a = the text has no such positions (e.g. no newline in the prose sentence).**

| head | attends to | shakespeare | prose | code | multiline |
|---|---|---|---|---|---|
| L0H1 | previous character | 0.56 (chance 0.09) | 0.59 (chance 0.08) | 0.50 (chance 0.10) | 0.48 (chance 0.06) |
| L1H1 | first letter of current word | 0.40 (chance 0.09) | 0.27 (chance 0.07) | 0.07 (chance 0.11) | 0.18 (chance 0.06) |
| L1H3 | first letter of current word (4th letter on) | 0.32 (chance 0.08) | 0.03 (chance 0.04) | 0.50 (chance 0.04) | 0.26 (chance 0.06) |
| L3H2 | most recent newline | 0.53 (chance 0.05) | n/a | 0.31 (chance 0.04) | 0.42 (chance 0.04) |

**Table 8 (appendix). All 16 heads of gpt_modern_s1337: average attention weight on each target. prev, two-back, word-start and position-0 measured on the Shakespeare text; word-start (4th letter on) and line-start on the multi-line text. Two-back is a control: a word-start head should score high on word start but not on two-back.**

| head | prev char (t−1) | two back (t−2) | word start | word start, 4th letter on | line start (newline) | position 0 |
|---|---|---|---|---|---|---|
| L0H0 | 0.50 | 0.12 | 0.06 | 0.01 | 0.03 | 0.03 |
| L0H1 | 0.56 | 0.30 | 0.16 | 0.00 | 0.00 | 0.04 |
| L0H2 | 0.45 | 0.15 | 0.09 | 0.05 | 0.04 | 0.10 |
| L0H3 | 0.22 | 0.39 | 0.20 | 0.22 | 0.15 | 0.10 |
| L1H0 | 0.18 | 0.15 | 0.08 | 0.04 | 0.11 | 0.04 |
| L1H1 | 0.18 | 0.18 | 0.40 | 0.13 | 0.06 | 0.08 |
| L1H2 | 0.28 | 0.22 | 0.13 | 0.06 | 0.01 | 0.09 |
| L1H3 | 0.21 | 0.21 | 0.39 | 0.26 | 0.05 | 0.13 |
| L2H0 | 0.17 | 0.23 | 0.15 | 0.05 | 0.03 | 0.05 |
| L2H1 | 0.42 | 0.17 | 0.09 | 0.06 | 0.02 | 0.08 |
| L2H2 | 0.47 | 0.18 | 0.10 | 0.17 | 0.06 | 0.04 |
| L2H3 | 0.15 | 0.21 | 0.21 | 0.04 | 0.06 | 0.03 |
| L3H0 | 0.21 | 0.21 | 0.14 | 0.15 | 0.03 | 0.09 |
| L3H1 | 0.11 | 0.14 | 0.13 | 0.13 | 0.15 | 0.07 |
| L3H2 | 0.13 | 0.06 | 0.13 | 0.15 | 0.42 | 0.11 |
| L3H3 | 0.22 | 0.22 | 0.32 | 0.02 | 0.03 | 0.04 |
| chance | 0.093 | 0.080 | 0.087 | 0.056 | 0.036 | 0.093 |
