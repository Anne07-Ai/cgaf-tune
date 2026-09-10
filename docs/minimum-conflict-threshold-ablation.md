# Minimum-conflict threshold ablation

Layer grouping and temperature 0.10 were fixed while the minimum conflict magnitude was varied.
Each configuration used seed 42, 100 steps, Dolly-15k, and generation token F1.

| Threshold | Domain F1 | Retention F1 | Harmonic F1 | Projected groups |
|---:|---:|---:|---:|---:|
| 0.00 | 0.3272 | 0.2105 | 0.2562 | 1,172 |
| 0.01 | 0.3272 | 0.2067 | 0.2534 | 874 |
| 0.03 | 0.3244 | 0.2018 | 0.2488 | 441 |
| **0.05** | **0.3272** | **0.2154** | **0.2598** | **238** |
| 0.10 | 0.3272 | 0.1824 | 0.2343 | 50 |

Threshold 0.05 is the screening winner. Relative to 0.00, it retained the same domain F1,
increased retention F1 by 0.0050, increased harmonic F1 by 0.0036, and reduced projection
operations by 79.7%.

The relationship is not monotonic: threshold 0.10 filters too aggressively and has the weakest
retention result. This suggests that ignoring weak conflicts may help, while ignoring moderate
conflicts can remove useful protection.

This is a single-seed screen on 12 examples per objective. Threshold 0.05 must be confirmed on
additional seeds before changing the project default.

[Hugging Face job](https://huggingface.co/jobs/lakshmianne/6aa2ce1521047bf1b0372e4b)
