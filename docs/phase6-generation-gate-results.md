# Phase 6 generation gate results

**Job:** [Hugging Face 6aa3107621047bf1b037372c](https://huggingface.co/jobs/lakshmianne/6aa3107621047bf1b037372c)  
**Status:** Completed  
**Hardware:** Tesla T4  
**Model:** Qwen/Qwen3-0.6B (QLoRA)  
**Seed:** 42

## Leakage-controlled setup

Each method trained on 512 Dolly-15k summarization examples and 512 closed/general QA anchor examples. Checkpoint selection used a separate mixed validation split of 64 examples per objective. Final generation scoring used another unseen test split of 24 examples per objective. The test split did not influence early stopping.

## Generation results

| Method | Domain F1 | Anchor F1 | Harmonic F1 | Harmonic gain vs base |
|---|---:|---:|---:|---:|
| Base model | 0.1816 | 0.2045 | 0.1924 | — |
| Rehearsal | **0.2956** | 0.2081 | **0.2443** | **+0.0519** |
| Hard projection | 0.2516 | **0.2184** | 0.233802 | +0.0414 |
| CGAF | 0.2622 | 0.2109 | 0.233795 | +0.0414 |
| LoRA | 0.2458 | 0.2104 | 0.2267 | +0.0344 |

## Early-stopping checkpoints

| Method | Best step | Completed steps | Best validation loss |
|---|---:|---:|---:|
| LoRA | 500 | 800 | 2.5229 |
| Rehearsal | 400 | 700 | **2.4671** |
| Hard projection | 500 | 800 | 2.5035 |
| CGAF | 500 | 800 | 2.5073 |

All methods stopped before the 1,200-step ceiling and restored their best validation checkpoint.

## Interpretation

Rehearsal is the single-seed winner on balanced generation quality. CGAF and hard projection are effectively tied in harmonic F1: their absolute difference is about 0.000007. Their trade-off differs: CGAF provides 0.0106 more domain F1, while hard projection provides 0.0074 more anchor F1. CGAF also improves harmonic F1 over plain LoRA by 0.0071.

This result clears the engineering and competitiveness gate, but it does not prove superiority. The final Phase 6 claim requires the same leakage-controlled protocol over seeds 42, 123, and 456 with paired uncertainty estimates.

## Decision

- Early stopping and best-checkpoint restoration: **pass**.
- Generation evaluation and split isolation: **pass**.
- CGAF competitiveness against hard projection: **pass / statistical tie at seed 42**.
- Overall winner at seed 42: **rehearsal**.
- Next experiment: three-seed confirmation; do not tune hyperparameters on the final test split.
