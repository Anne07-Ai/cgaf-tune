# Three-seed minimum-conflict threshold confirmation

Thresholds 0.00 and 0.05 were compared with layer grouping, temperature 0.10, 100 steps, and
generation token F1 across seeds 42, 123, and 456.

| Threshold | Mean domain F1 | Mean retention F1 | Mean harmonic F1 | Mean projected groups |
|---:|---:|---:|---:|---:|
| **0.00** | **0.31479** | **0.19133** | **0.23789** | 1,126.7 |
| 0.05 | 0.31478 | 0.19061 | 0.23722 | **186.3** |

Threshold 0.05 reduced projection events by 83.5%, but its seed-42 screening improvement did not
replicate. It lost clearly on seed 123 and won only slightly on seed 456. The three-seed harmonic
mean is 0.00067 below threshold 0.00.

## Decision

Retain `minimum_conflict: 0.00` as the default. There is no consistent quality evidence for
changing it to 0.05.

Threshold 0.05 remains interesting as a sparse-intervention variant, but fewer projection events
did not materially reduce wall-clock cost because both domain and anchor gradients still must be
computed. It should be treated as an ablation, not the recommended setting.

The earlier job `6aa2d3cd5527934177ec199c` failed during dependency parsing before training and is
excluded. Only the corrected completed job is evidence.

[Valid confirmation job](https://huggingface.co/jobs/lakshmianne/6aa2d4135527934177ec19d0)
