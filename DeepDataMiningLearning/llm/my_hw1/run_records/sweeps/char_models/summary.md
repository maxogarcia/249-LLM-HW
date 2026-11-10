metric: `val_bpb` (lower is better)

```
              n_seeds  val_bpb_mean  val_bpb_std  params_M  train_min  delta_vs_gpt_textbook  paired_p
name                                                                                                  
gpt_modern          2        2.1082       0.0079    3.1626     1.2406                -0.0711    0.0228
lstm                2        2.1384       0.0008    1.0698     1.6008                -0.0409    0.0391
gru                 2        2.1631       0.0041    0.8067     1.2890                -0.0162    0.0066
lstm_512            2        2.1745       0.0067    4.2368     2.0257                -0.0049    0.6450
gpt_textbook        2        2.1793       0.0043    3.4383     1.0293                 0.0000       NaN
rnn                 2        2.4378       0.0040    0.2803     0.1788                 0.2584    0.0144
```
