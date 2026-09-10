# 100-step controlled-conflict effectiveness run

## Outcome

All four matched QLoRA methods completed 100 update steps on a Tesla T4. The run used seed 42,
96 domain examples, 96 retention-anchor examples, and separate 16-example held-out sets.

This is a controlled synthetic stress test, not yet a natural-domain or multi-seed result.

- [Hugging Face job](https://huggingface.co/jobs/lakshmianne/6aa294ce5527934177ec0a18)
- Model: `Qwen/Qwen3-0.6B`
- QLoRA rank / alpha: 8 / 16
- Sequence length: 128
- Base domain loss: 6.2299
- Base retention loss: 5.1928

## Held-out results

Lower final loss is better; gain is base loss minus final loss, so larger positive gain is better.

| Method | Domain loss | Domain gain | Retention loss | Retention gain | Conflict steps | Projected groups | Train time |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoRA | 4.0056 | 2.2243 | 4.6907 | 0.5021 | 0 | 0 | 50.4 s |
| Rehearsal | 3.8243 | 2.4056 | **3.3929** | **1.7999** | 0 | 0 | 87.4 s |
| Hard projection | 3.3742 | 2.8557 | 4.1383 | 1.0545 | 90 | 1,250 | 91.1 s |
| **CGAF** | **3.0886** | **3.1412** | 4.2595 | 0.9333 | 90 | 1,203 | 90.6 s |

## Interpretation

CGAF produced the strongest domain adaptation in this run. Relative to hard projection, its
held-out domain loss was 8.5% lower, while its retention loss was 2.9% higher. This is consistent
with the intended soft-versus-hard intervention trade-off: CGAF preserved more of the domain
gradient, whereas hard projection protected the anchor objective more aggressively.

Rehearsal achieved the best retention result and the strongest simple balanced improvement across
the two losses. Therefore this run does **not** establish CGAF as the overall winner.

Gradient conflict was real and frequent: CGAF and hard projection detected conflicting groups on
90 of 100 steps. CGAF projected 1,203 layer groups versus 1,250 for hard projection. At step 100,
hard projection altered 17 of 28 layer groups, while CGAF altered only one group with a soft gate.

## Research conclusion

The implementation passed its first multi-step effectiveness test, and the adaptive mechanism
behaved differently from its baselines. The promising signal is CGAF's best domain score with
fewer projected groups. The counter-signal is that rehearsal and hard projection retained the
anchor set better.

No publication-level claim should be made from one synthetic seed. Next run must use seeds 42,
123, and 456, a natural-domain dataset, generation-based task metrics, confidence intervals, and
peak-memory reporting.
