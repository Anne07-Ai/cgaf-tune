# Natural-data three-seed matched experiment

## Outcome

Twelve matched QLoRA runs completed successfully on a Tesla T4: four methods across seeds 42,
123, and 456, with 100 updates per run.

- Dataset: [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- Domain adaptation: summarization
- Retention anchors: closed QA and general QA
- Train/evaluation examples per objective: 96/24
- Model: `Qwen/Qwen3-0.6B`
- [Hugging Face job](https://huggingface.co/jobs/lakshmianne/6aa29df75527934177ec0c53)

The frozen base losses were 2.5108 for the domain set and 2.9148 for the retention set.

## Three-seed results

Gain is base loss minus adapted-model loss; higher is better. Values are mean ± sample standard
deviation across three seeds.

| Method | Domain gain ↑ | Retention gain ↑ | Conflict steps | Projected groups | Train time |
|---|---:|---:|---:|---:|---:|
| LoRA | 0.3601 ± 0.0061 | 0.2685 ± 0.0221 | — | — | 60.0 s |
| **Rehearsal** | 0.3561 ± 0.0041 | **0.3735 ± 0.0044** | — | — | 103.0 s |
| Hard projection | **0.3609 ± 0.0048** | 0.2984 ± 0.0138 | 98.3/100 | 1,139.7 |
| CGAF | 0.3609 ± 0.0051 | 0.2908 ± 0.0143 | 98.3/100 | 1,132.0 |

## Paired differences versus LoRA

| Method | Domain-gain difference | Retention-gain difference |
|---|---:|---:|
| Rehearsal | -0.0039 | **+0.1049** |
| Hard projection | +0.0008 | +0.0299 |
| CGAF | +0.0008 | +0.0222 |

## Interpretation

Rehearsal is the strongest balanced method in this experiment. It sacrifices only 0.0039 mean
domain gain versus LoRA while adding 0.1049 retention gain, and its retention result is consistent
across seeds.

CGAF's mechanism is active and stable: conflicts occurred on an average of 98.3 of 100 steps.
However, CGAF is effectively tied with hard projection on domain gain and trails it by 0.0077 on
retention gain. The soft gate used slightly fewer total projections, but that did not translate
into a better held-out trade-off here.

The domain improvements of hard projection and CGAF over LoRA are only about 0.0008—far smaller
than the across-seed standard deviations—so they should not be presented as meaningful wins.

## Research decision

The current claim that CGAF is superior is **not supported** on this natural-data configuration.
That is a useful falsification result. The next iteration should focus on when soft gating is
beneficial rather than assuming universal superiority:

1. sweep gate temperature and minimum-conflict threshold;
2. compare layer, module, and global grouping;
3. measure generation-task quality, not loss alone;
4. test stronger distribution shifts and longer adaptation;
5. report bootstrap intervals across held-out examples.

Until those ablations show a repeatable advantage, rehearsal remains the recommended baseline.
