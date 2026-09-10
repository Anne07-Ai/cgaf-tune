# Hugging Face T4 GPU pilot — 2026-09-10

## Outcome

The first real GPU execution of CGAF-Tune completed successfully on a Tesla T4 using
`Qwen/Qwen3-0.6B` with 4-bit QLoRA. All four matched method paths completed with exit code 0.

This was an **execution smoke pilot**, not an effectiveness benchmark: one update step, one seed,
four domain examples, and four anchor examples.

- Hugging Face job: [6aa28fdc5527934177ec092c](https://huggingface.co/jobs/lakshmianne/6aa28fdc5527934177ec092c)
- Seed: 42
- QLoRA rank / alpha: 8 / 16
- Maximum sequence length: 128
- Batch size: 1
- Update steps per method: 1
- Runtime: Tesla T4, PyTorch 2.14.0+cu130, CUDA 13.0

## Matched results

| Method | Domain loss | Anchor loss | Raw domain grad norm | Final grad norm | Projected groups | Mean conflict cosine | Step time |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoRA | 4.2869 | — | 9.9906 | 9.9906 | 0 | — | 3.559 s |
| Rehearsal | 4.2869 | 2.6893 | 23.3717 | 23.3717 | 0 | — | 2.973 s |
| Hard projection | 4.2869 | 2.6893 | 9.9906 | 1.0000 | 0 | 0.2749 | 2.930 s |
| CGAF | 4.2869 | 2.6893 | 9.9906 | 1.0000 | 0 | 0.2749 | 3.008 s |

The 1.0 final gradient norm for hard projection and CGAF comes from configured gradient clipping,
not conflict projection.

## Interpretation

All 28 measured layer-local domain/anchor cosine similarities were positive, ranging from
0.0785 to 0.6809. Therefore neither hard projection nor CGAF intervened:

- projected groups: 0/28
- CGAF mean gate: 0
- removed gradient norm: 0 in every layer

This is correct behaviour: CGAF should leave aligned gradients untouched. Because the tiny pilot
did not contain a conflicting batch, it cannot yet distinguish CGAF from hard projection or
support any claim about adaptation/retention quality.

The measured CGAF step took 3.008 seconds versus 2.930 seconds for hard projection in this single
cached run (about 2.7% difference). This sample is too small for a reliable performance claim.

## Next research run

Use a larger, deliberately diverse domain/anchor dataset and:

1. run 100–500 steps per method;
2. use seeds 42, 123, and 456;
3. evaluate the frozen base and each adapter on held-out domain and retention sets;
4. report domain gain, forgetting, harmonic score, confidence intervals, and GPU memory;
5. confirm that negative layer cosines occur and compare CGAF soft gates with hard projection.

Only that run should be treated as the first effectiveness experiment.
