metric: `val_bpb` (lower is better)

```
                 n_seeds  val_bpb_mean  val_bpb_std  params_M  train_min  delta_vs_gpt_modern_T256  paired_p
name                                                                                                        
gpt_modern_T512        2        2.1065       0.0002    3.1626     1.2996                   -0.0040    0.4340
gpt_modern_T256        2        2.1105       0.0049    3.1626     1.4644                    0.0000       NaN
gpt_modern_T128        2        2.1180       0.0176    3.1626     1.5593                    0.0074    0.7219
lstm_T512              2        2.1239       0.0059    1.0698     2.2759                    0.0133    0.0341
lstm_T256              2        2.1387       0.0002    1.0698     1.4218                    0.0281    0.0747
gpt_modern_T64         2        2.1537       0.0013    3.1626     1.5220                    0.0432    0.0378
lstm_T128              2        2.1543       0.0044    1.0698     0.9571                    0.0438    0.0945
lstm_T64               2        2.1889       0.0082    1.0698     0.6555                    0.0784    0.0746
```
