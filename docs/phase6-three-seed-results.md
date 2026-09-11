# Phase 6 three-seed generation confirmation

**Job:** [Hugging Face 6aa31f875527934177ec2c84](https://huggingface.co/jobs/lakshmianne/6aa31f875527934177ec2c84)  
**Status:** Completed  
**Hardware:** Tesla T4  
**Model:** Qwen/Qwen3-0.6B (QLoRA)  
**Seeds:** 42, 123, 456

## Protocol

All methods used the same Dolly-15k split: 512 domain-training examples, 512 anchor-training examples, 64 validation examples per objective, and 24 unseen final-test examples per objective. Checkpoints were selected only by the validation set. Generation token F1 was measured only on the final test set. Training used a 1,200-step ceiling with validation every 100 steps and early stopping.

## Aggregate generation results

| Method | Domain F1 | Anchor F1 | Harmonic F1 |
|---|---:|---:|---:|
| Rehearsal | 0.2803 ± 0.0177 | 0.2152 ± 0.0074 | **0.2432 ± 0.0074** |
| LoRA | 0.2725 ± 0.0255 | **0.2175 ± 0.0120** | 0.2418 ± 0.0168 |
| CGAF | **0.2813 ± 0.0254** | 0.2094 ± 0.0096 | 0.2399 ± 0.0142 |
| Hard projection | 0.2752 ± 0.0264 | 0.2128 ± 0.0094 | 0.2396 ± 0.0125 |

Values are mean ± sample standard deviation across three seeds.

## Paired differences against LoRA

| Method | Domain F1 | Anchor F1 | Harmonic F1 |
|---|---:|---:|---:|
| Rehearsal | +0.0078 | -0.0023 | +0.0014 |
| Hard projection | +0.0027 | -0.0047 | -0.0021 |
| CGAF | **+0.0088** | -0.0081 | -0.0019 |

CGAF versus hard projection had mean differences of +0.0061 domain F1, -0.0034 anchor F1, and +0.0003 harmonic F1. The harmonic advantage is negligible at this sample size.

## Interpretation

CGAF produced the highest mean domain-adaptation score, which supports the claim that its soft gate permits useful domain updates. However, its retention score was the lowest of the four methods, so the primary balanced objective did not improve. Rehearsal had the best mean harmonic F1, but its advantage over LoRA was only 0.0014 and reversed on two of three individual seeds. None of these small differences supports a superiority claim.

The result is therefore informative but negative for the current CGAF hypothesis: under this model, dataset, and configuration, conflict gating changes the adaptation-retention trade-off but does not outperform simple LoRA or rehearsal on balanced generation quality.

## Phase 6 decision

Phase 6 is complete.

- Larger natural-data protocol: **passed**.
- Leakage-controlled train/validation/test separation: **passed**.
- Validation checkpoints and early stopping: **passed**.
- Three-seed generation evaluation: **passed**.
- CGAF superiority hypothesis: **not supported in the current setting**.

Phase 7 should report uncertainty and trade-off visualizations, analyse seed sensitivity, and frame CGAF as a tested method rather than a state-of-the-art claim. Further method development should target retention strength before spending compute on larger models.
