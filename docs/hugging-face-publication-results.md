# Hugging Face adapter publication

**Status:** Completed  
**Published model:** [lakshmianne/cgaf-qwen3-0.6b-seed42](https://huggingface.co/lakshmianne/cgaf-qwen3-0.6b-seed42)  
**Visibility:** Private  
**GPU job:** [6aa38b1321047bf1b0374c80](https://huggingface.co/jobs/lakshmianne/6aa38b1321047bf1b0374c80)

## Published artifact

The repository contains the validation-selected CGAF PEFT adapter for Qwen3-0.6B, tokenizer files,
a research model card, and release evaluation. The adapter was uploaded before the ephemeral job
terminated.

## Release results

| Measure | Value |
|---|---:|
| Completed steps | 800 / 1,200 |
| Best step | 500 |
| Best validation loss | 2.5071 |
| Domain F1 | 0.2553 |
| Anchor F1 | 0.2025 |
| Harmonic F1 | 0.2259 |

The publication rerun's harmonic F1 is lower than the earlier seed-42 result (0.2338). This is
reported as reproduction variance rather than hidden. Quantized GPU kernels and the software
environment may contribute to nondeterminism. The three-seed Phase 6 aggregate remains the primary
research result; this checkpoint is a reference release, not a selected best-performing seed.

## Security

The Hugging Face write token was supplied through a masked GitHub Actions secret. It was not stored
in repository files or printed in logs.
