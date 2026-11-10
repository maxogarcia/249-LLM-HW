**Table 10. GRUScratch vs nn.GRU: max |difference| (D=8, H=16, batch 4, 10 steps). Loss = MSE(output, random target) + ‖h_T‖², back-propagated through both models. GPU: NVIDIA GeForce RTX 4080 Laptop GPU.**

| setting | output | h_T | x.grad | grad weight_ih_l0 | grad bias_ih_l0 | grad weight_hh_l0 | grad bias_hh_l0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU float64 | 2.2e-16 | 1.7e-16 | 5.6e-17 | 6.7e-16 | 2.6e-16 | 1.1e-16 | 1.1e-16 |
| GPU float32, TF32 on | 5.4e-05 | 5.4e-05 | 5.8e-05 | 4.3e-04 | 2.3e-04 | 1.5e-04 | 1.2e-04 |
| GPU float32, TF32 off | 4.2e-06 | 3.2e-06 | 9.0e-07 | 8.8e-06 | 1.1e-05 | 2.2e-06 | 6.3e-06 |
| CPU float32 (reference point) | 1.0e-07 | 1.0e-07 | 3.0e-08 | 2.4e-07 | 4.8e-07 | 6.0e-08 | 2.4e-07 |
