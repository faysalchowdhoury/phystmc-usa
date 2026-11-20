# NGSIM I-80 Ground-Truth Validation

78 real clean trajectories; ground truth = Savitzky-Golay differentiation of full-rate positions; detection noise σ=0.5 m. Lower MAE is better.

| method    |   hz |   speed_mae_mps |   accel_mae_mps2 |   unphysical_pct |   n_samples |
|:----------|-----:|----------------:|-----------------:|-----------------:|------------:|
| ema_clamp |   10 |           2.612 |            4.135 |            39.76 |       68528 |
| raw       |   10 |           2.82  |           24.519 |            91.27 |       68528 |
| smoother  |   10 |           0.511 |            2.119 |            17.13 |       68528 |
| ema_clamp |    5 |           1.827 |            3.768 |            32.13 |       34205 |
| raw       |    5 |           1.406 |            6.125 |            67.67 |       34205 |
| smoother  |    5 |           0.545 |            1.514 |             5.67 |       34205 |
| ema_clamp |    2 |           1.442 |            2.842 |             8.81 |       13611 |
| raw       |    2 |           0.71  |            1.606 |            15.5  |       13611 |
| smoother  |    2 |           0.901 |            1.832 |             1.28 |       13611 |
