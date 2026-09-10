# Three-seed temperature confirmation

Layer-grouped CGAF was evaluated at temperatures 0.03, 0.10, and 0.30 across seeds 42, 123,
and 456. Each run used 100 steps and generation-based token F1 on Dolly-15k summarization and QA.

| Temperature | Mean domain F1 | Mean retention F1 | Mean harmonic F1 |
|---:|---:|---:|---:|
| 0.03 | 0.31339 | 0.19111 | 0.23719 |
| **0.10** | 0.31339 | **0.19179** | **0.23785** |
| 0.30 | **0.31626** | 0.18628 | 0.23428 |

The seed-42 screening winner (0.03) did not remain the winner after confirmation. Temperature 0.10
has the highest three-seed harmonic mean, but its advantage over 0.03 is only 0.00066. That
difference is too small to support a substantive superiority claim.

## Decision

Retain the original temperature **0.10** as the project default. It is marginally best on the
three-seed balanced metric and avoids selecting a new hyperparameter from a one-seed result.

Temperature 0.30 shifts slightly toward domain generation at the cost of retention. Overall,
temperature sensitivity is modest in this range; grouping granularity appears to have a larger
effect than temperature.

All six confirmation runs completed successfully. Exact match remained unsuitable for these
free-form outputs, so conclusions use token F1.

[Confirmation job](https://huggingface.co/jobs/lakshmianne/6aa2c5c921047bf1b0372d49)
