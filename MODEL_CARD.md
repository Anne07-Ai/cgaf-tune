---
language:
  - en
license: apache-2.0
library_name: transformers
tags:
  - qwen3
  - peft
  - lora
  - catastrophic-forgetting
  - research
  - negative-results
base_model: Qwen/Qwen3-0.6B
---

# CGAF-Tune Research Card

CGAF-Tune is an experimental framework for Conflict-Gated Adaptive Fine-Tuning of LoRA adapters. This card describes the method and evaluation package; **it is not a published trained model checkpoint**.

## Intended use

Use this repository to reproduce research on domain adaptation versus capability retention, compare CGAF with matched LoRA/rehearsal/hard-projection baselines, or extend the gate with an explicit retention constraint.

Do not treat the current method as state of the art or deploy it as a safety-preservation mechanism.

## Method

CGAF computes domain and anchor gradients for trainable adapter groups. Negative alignment triggers a smooth, temperature-controlled projection. The confirmed configuration uses layer grouping, temperature 0.1, and minimum conflict 0.0.

## Evaluation summary

Across Qwen3-0.6B QLoRA runs with seeds 42, 123, and 456:

| Method | Domain F1 | Anchor F1 | Harmonic F1 |
|---|---:|---:|---:|
| Rehearsal | 0.2803 | 0.2152 | **0.2432** |
| LoRA | 0.2725 | **0.2175** | 0.2418 |
| CGAF | **0.2813** | 0.2094 | 0.2399 |
| Hard projection | 0.2752 | 0.2128 | 0.2396 |

CGAF improved domain adaptation but did not improve the balanced primary metric. All paired seed-bootstrap intervals against LoRA included zero.

## Training data

The final experiment uses license-governed examples from `databricks/databricks-dolly-15k`: summarization for domain adaptation and closed/general QA as retention anchors. Exact split construction is recorded in the experiment reports.

## Limitations

One small model, one English dataset, three seeds, and small final-test splits limit generalization. Token F1 is incomplete. Current Hugging Face artifacts do not include adapter weights; reproduce them from the code and configuration.

## Environmental considerations

QLoRA reduces memory requirements, but gradient-conflict methods require two backward paths and are slower than plain LoRA. Early stopping reduced unnecessary updates.

## Citation

Until an archival paper identifier exists, cite the software using `CITATION.cff` and link the GitHub repository.
