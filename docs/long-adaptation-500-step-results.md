# 500-step, three-seed long-adaptation experiment

All 12 matched runs completed: four methods, three seeds, and 500 updates per run on Dolly-15k.

| Method | Domain F1 | Retention F1 | Harmonic F1 | Held-out domain-loss gain | Held-out retention-loss gain | Train time |
|---|---:|---:|---:|---:|---:|---:|
| LoRA | **0.3386** | 0.2379 | 0.2795 ± 0.0083 | -0.9424 | -1.1652 | 236 s |
| **Rehearsal** | 0.3352 | **0.2613** | **0.2936 ± 0.0191** | **-0.5422** | **-0.8104** | 447 s |
| Hard projection | 0.3144 | 0.2407 | 0.2726 ± 0.0126 | -0.8968 | -0.9328 | 466 s |
| CGAF | 0.3229 | 0.2407 | 0.2756 ± 0.0166 | -0.9026 | -1.0036 | 469 s |

Positive loss gain would mean improvement over the frozen base; every loss gain is negative.

## Findings

Rehearsal is the strongest balanced method after 500 steps. It has the highest harmonic
generation F1 and best retention F1. Its mean harmonic F1 is 0.0141 above matched LoRA.

CGAF slightly outperforms hard projection in harmonic F1 (+0.0030) and domain F1 (+0.0085), with
essentially identical retention F1. However, CGAF trails LoRA by 0.0039 harmonic F1 and rehearsal
by 0.0180. It therefore does not establish a superior adaptation-retention trade-off.

CGAF detected conflicts on 491–497 of 500 steps and projected roughly 6,500–6,700 layer groups,
so the mechanism remained active throughout training.

## Overfitting and metric warning

All methods drove training losses low but made held-out causal-LM losses worse than the frozen
base. At the same time, generation token F1 improved. This is an important warning rather than a
simple contradiction:

- 500 repeated updates over only 96 examples likely overfit;
- causal-LM loss includes prompt tokens, while generation F1 scores output overlap;
- token F1 on 24 free-form examples is a small and imperfect task metric.

The evidence supports early stopping and larger held-out sets. It does not support extending the
same small dataset to still more steps.

## Decision

Rehearsal remains the recommended baseline. CGAF's current value is diagnostic selectivity and a
small advantage over hard projection at long duration, not overall state-of-the-art performance.

The next implementation priority should be validation checkpoints and early stopping, followed by
a larger natural-domain dataset and task-appropriate metrics such as ROUGE for summarization and
exact/F1 for QA.

[Hugging Face job](https://huggingface.co/jobs/lakshmianne/6aa2d9895527934177ec1ac9)
