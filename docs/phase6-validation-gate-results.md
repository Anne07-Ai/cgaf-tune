# Phase 6 validation-controlled scaling: gate results

**Job:** [Hugging Face 6aa2f6b221047bf1b037328e](https://huggingface.co/jobs/lakshmianne/6aa2f6b221047bf1b037328e)  
**Status:** Completed  
**Hardware:** Tesla T4  
**Model:** Qwen/Qwen3-0.6B (QLoRA)  
**Seed:** 42

## Setup

The gate used 512 Dolly-15k summarization examples as domain training data, 512 closed/general QA examples as the retention anchor, and a disjoint 128-example mixed validation set. Every method received the same maximum 1,200-step budget. Validation ran every 100 steps with patience 3 and minimum improvement 0.002. The best trainable adapter state was restored before saving.

## Results

| Method | Completed steps | Best step | Best validation loss | Stop reason | Wall time |
|---|---:|---:|---:|---|---:|
| Rehearsal | 700 | 400 | **2.4669** | Early stopping | 687.9 s |
| Hard projection | 800 | 500 | 2.5042 | Early stopping | 817.0 s |
| CGAF | 800 | 500 | 2.5074 | Early stopping | 835.3 s |
| LoRA | 800 | 500 | 2.5224 | Early stopping | 463.8 s |

## Interpretation

The validation-controlled runner worked as intended: all methods stopped 400–500 steps before the maximum budget, avoiding the unnecessary continued adaptation seen in the prior tiny-data run. Rehearsal achieved the best mixed validation loss. CGAF improved on plain LoRA by 0.0150 loss, but trailed hard projection by 0.0032, which is too small to treat as a meaningful win or loss from one seed.

This gate measures mixed causal-LM validation loss, not final generated-answer quality. Therefore it does **not** establish the Phase 6 research claim. Before a three-seed scaling matrix, the restored best checkpoints need generation-based domain and retention evaluation. If that check is healthy, the same protocol should run across seeds 42, 123, and 456.

## Decision

- Validation and early stopping implementation: **pass**.
- Larger-data execution reliability: **pass**.
- CGAF superiority claim: **not established**.
- Next action: generation evaluation of the four restored best checkpoints, followed by a three-seed matrix only if the gate remains competitive.
