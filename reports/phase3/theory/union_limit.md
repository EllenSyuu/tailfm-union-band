# Limiting coverage and width of the finite-mesh union band

`scripts/theory_union_limit.py`; Monte Carlo evaluation from the exact Gaussian limit at t_j = k_j/k_max for k = [64, 128, 256, 512, 1024], 20000 evaluation paths (seed 6461), calibration on 10000 independent paths (seed 6460). Nominal 0.90; Monte Carlo s.e. of a coverage near 0.9 ≈ 0.0021. λ is the local bias √k δ(U(n/k)) at the top grid point, in units of ξ, swept over [−48, 48] (fine on [−12, 12]).

| mesh | ρ range | worst coverage over λ | at (ρ, λ) | coverage at λ = 0 (min over ρ) | width/oracle at top, λ = 0 (max over ρ) | width/oracle at top, worst λ |
|---|---|---|---|---|---|---|
| 6 | ρ ∈ R = [−2, −0.5] | 0.945 | (-2.000, -6.0) | 0.947 | 2.00 | 7.76 |
| 6 | ρ ∉ R | 0.000 | (-3.000, -48.0) | 0.947 | 2.25 | 10.42 |
| 12 | ρ ∈ R = [−2, −0.5] | 0.945 | (-2.000, -6.0) | 0.947 | 2.00 | 7.76 |
| 12 | ρ ∉ R | 0.000 | (-3.000, -48.0) | 0.947 | 2.25 | 10.42 |
| 24 | ρ ∈ R = [−2, −0.5] | 0.945 | (-2.000, -6.0) | 0.947 | 2.00 | 7.76 |
| 24 | ρ ∉ R | 0.000 | (-3.000, -48.0) | 0.947 | 2.25 | 10.42 |

Per-ρ worst case, 6-point mesh (the evaluated method):

| ρ | worst coverage | λ at worst | coverage at λ = −48 / +48 |
|---|---|---|---|
| -0.500 | 0.947 | +1.5 | 0.950 / 0.949 |
| -0.574 | 0.947 | +0.0 | 0.966 / 0.966 |
| -0.660 | 0.947 | +0.0 | 0.974 / 0.973 |
| -0.758 | 0.947 | +0.0 | 0.976 / 0.976 |
| -0.871 | 0.947 | +0.0 | 0.977 / 0.976 |
| -1.000 | 0.947 | +0.0 | 0.976 / 0.975 |
| -1.149 | 0.947 | +0.0 | 0.974 / 0.974 |
| -1.319 | 0.947 | +0.0 | 0.972 / 0.972 |
| -1.516 | 0.947 | +0.0 | 0.968 / 0.968 |
| -1.741 | 0.947 | +0.0 | 0.963 / 0.964 |
| -2.000 | 0.945 | -6.0 | 0.946 / 0.947 |
| -0.250 | 0.739 | -48.0 | 0.739 / 0.741 |
| -0.350 | 0.832 | -48.0 | 0.832 / 0.834 |
| -0.450 | 0.924 | +48.0 | 0.924 / 0.924 |
| -2.500 | 0.168 | +48.0 | 0.169 / 0.168 |
| -3.000 | 0.000 | -48.0 | 0.000 / 0.000 |
