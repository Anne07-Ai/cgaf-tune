# Conflict-Gated Adaptive Fine-Tuning: A Three-Seed Study of Adaptation–Retention Trade-offs

**Lakshmi Anne**  
Project repository: [Anne07-Ai/cgaf-tune](https://github.com/Anne07-Ai/cgaf-tune)

## Abstract

Parameter-efficient fine-tuning can specialize a language model while weakening capabilities that are not represented in the adaptation data. We study Conflict-Gated Adaptive Fine-Tuning (CGAF), a LoRA-compatible method that measures gradient conflict between domain and capability-retention batches and smoothly projects only conflicting adapter-gradient components. CGAF is compared with plain LoRA, rehearsal, and hard gradient projection using Qwen3-0.6B under matched data, parameter, optimization, stopping, and evaluation budgets. The study includes controlled-conflict tests, grouping and temperature ablations, threshold screening, validation-controlled checkpoint selection, and a final three-seed generation experiment. In the final setting, CGAF achieved the highest mean domain token F1 (0.2813) but the lowest mean retention F1 (0.2094); its harmonic F1 (0.2399) did not exceed rehearsal (0.2432) or LoRA (0.2418). Paired seed-bootstrap intervals included zero. These results do not support a superiority claim. They instead show that conflict gating changes the adaptation–retention operating point, and that stronger or dynamically budgeted retention control is required.

## 1. Introduction

Fine-tuning a pretrained language model on a narrow domain creates a multi-objective problem. The model should improve on the target distribution while retaining broadly useful instruction-following and factual behavior. Ordinary LoRA provides an efficient adaptation mechanism but no direct protection for retained capabilities. Rehearsal mixes retention examples into training, while gradient-projection methods modify updates when objectives conflict.

CGAF tests a more selective intervention: measure conflict locally within adapter groups and scale projection continuously by conflict severity. The motivating hypothesis is that soft local correction can preserve more useful adaptation than uniform rehearsal or hard projection. This work evaluates that hypothesis as a falsifiable research question rather than assuming the method is superior.

The contributions are:

1. a PEFT-compatible implementation of local, soft conflict projection;
2. matched LoRA, rehearsal, and hard-projection baselines;
3. reproducible controlled and natural-data experiments with explicit ablations;
4. validation checkpoints, early stopping, leakage-separated generation evaluation, and multi-seed reporting;
5. an honest negative result identifying the method's current limitation.

## 2. Method

Let \(g_l^D\) and \(g_l^A\) denote domain and anchor gradients for adapter group \(l\). CGAF computes cosine alignment \(c_l\) and applies

\[
g'_l = g_l^D - \gamma_l
\frac{\min(0,\langle g_l^D,g_l^A\rangle)}
{\lVert g_l^A\rVert^2+\epsilon}g_l^A,
\qquad
\gamma_l=\sigma(-c_l/T).
\]

Positive alignment is unchanged. When alignment is negative, the projection magnitude increases smoothly with conflict severity. Temperature \(T\) controls gate sharpness; \(\epsilon\) stabilizes the denominator. The selected configuration uses layer grouping, temperature 0.1, and minimum conflict 0.0.

### 2.1 Baselines

- **LoRA:** domain-only low-rank adaptation.
- **Rehearsal:** joint domain and anchor loss.
- **Hard projection:** full projection of conflicting domain-gradient components.
- **CGAF:** smoothly gated local projection.

Trainable modules, LoRA rank, data budgets, maximum optimizer steps, validation schedule, seeds, and decoding are matched.

## 3. Experimental Design

### 3.1 Model and data

Experiments use Qwen3-0.6B with 4-bit QLoRA on a Tesla T4. The final protocol draws from Databricks Dolly-15k:

- 512 summarization examples for domain training;
- 512 closed/general QA examples for anchor training;
- 64 held-out validation examples per objective;
- 24 unseen final-test examples per objective.

A fixed dataset shuffle determines split membership. Training seeds are 42, 123, and 456. The final test split does not select checkpoints or hyperparameters.

### 3.2 Validation and stopping

Validation occurs every 100 optimizer steps. Training stops after three validations without improvement of at least 0.002, subject to a 1,200-step ceiling. The best trainable adapter state is restored before generation evaluation.

### 3.3 Metrics

Generated responses are scored with normalized token F1. We report domain F1, anchor-retention F1, and their harmonic mean. Mean, sample standard deviation, paired differences against LoRA, and seed-bootstrap 95% intervals are reported. With only three seeds, intervals are descriptive and no confirmatory significance claim is made.

## 4. Development Experiments

A one-step GPU pilot first verified all methods. A controlled 100-step conflict experiment tested whether the projection mechanism activates. Natural Dolly experiments then tested multiple seeds. Temperature values 0.03, 0.1, and 0.3 and global, layer, and module grouping were screened with generation evaluation. Layer grouping and temperature 0.1 were retained for confirmation. Minimum-conflict threshold experiments selected 0.0.

A 500-step experiment over only 96 examples showed an important warning: generation F1 improved while held-out causal-LM loss worsened. This exposed repeated-data overfitting and objective mismatch. Phase 6 therefore introduced larger splits, validation checkpoints, and early stopping rather than increasing steps on the same tiny dataset.

## 5. Results

| Method | Domain F1 | Anchor F1 | Harmonic F1 |
|---|---:|---:|---:|
| Rehearsal | 0.2803 ± 0.0177 | 0.2152 ± 0.0074 | **0.2432 ± 0.0074** |
| LoRA | 0.2725 ± 0.0255 | **0.2175 ± 0.0120** | 0.2418 ± 0.0168 |
| CGAF | **0.2813 ± 0.0254** | 0.2094 ± 0.0096 | 0.2399 ± 0.0142 |
| Hard projection | 0.2752 ± 0.0264 | 0.2128 ± 0.0094 | 0.2396 ± 0.0125 |

CGAF improves mean domain F1 over LoRA by 0.0088 but lowers anchor F1 by 0.0081, producing a harmonic difference of -0.0019. Against hard projection, CGAF gains approximately 0.0061 domain F1, loses 0.0034 anchor F1, and changes harmonic F1 by only +0.0003.

The non-dominated mean operating points are LoRA, rehearsal, and CGAF. Hard projection is dominated by rehearsal. Rankings are seed-sensitive: rehearsal wins seed 42, while LoRA wins seeds 123 and 456. Every paired seed-bootstrap interval against LoRA includes zero.

![Adaptation-retention trade-off](assets/phase7-tradeoff.svg)

![Seed sensitivity](assets/phase7-seed-sensitivity.svg)

## 6. Discussion

The results support a mechanism-level observation: soft conflict gating permits stronger domain movement than the tested alternatives. They do not support the primary superiority hypothesis because the gain is purchased with reduced retention. CGAF currently behaves as an adaptation-favoring point on the Pareto frontier rather than a better balanced optimizer.

Three factors may explain the outcome. First, a fixed gate temperature does not enforce a retention budget. Second, local cosine conflict is an instantaneous proxy and may not predict long-horizon forgetting. Third, the anchor objective and token-level loss may not adequately represent retained behavior. A promising next method would adapt gate strength to an explicit validation-retention constraint.

## 7. Limitations

The evaluation uses one 0.6B model, one English instruction dataset, three seeds, and 24 final examples per objective. Token F1 does not measure factuality, style quality, or human preference. Hyperparameters received a limited rather than equal exhaustive search. Generation evaluation is deterministic but the training system and quantized kernels may still introduce nondeterminism. No claim is made for larger models, multilingual data, safety retention, or production deployment.

The project does not currently publish trained adapters. Hugging Face Jobs environments were ephemeral and the connected credential did not grant repository-write access. All reported numerical results and code are preserved, but model-weight reproducibility requires rerunning the documented protocol.

## 8. Conclusion

CGAF is implementable, auditable, and competitive as an adaptation-focused method, but it does not improve balanced adaptation–retention performance in the tested setting. Rehearsal has the best mean harmonic F1 and LoRA has the best mean retention. The negative result narrows the next research question: can a conflict gate be coupled to an explicit, dynamically enforced retention budget?

## Reproducibility and Ethics Statement

Code, configurations, experiment reports, raw summaries, and analysis scripts are public in the project repository. The project uses public model and dataset artifacts subject to their original licenses. No private or personal training data is introduced. Results are reported regardless of whether they support the initial hypothesis.

## References

1. Hu et al. “LoRA: Low-Rank Adaptation of Large Language Models.” 2021.
2. Yu et al. “Gradient Surgery for Multi-Task Learning.” 2020.
3. Farajtabar et al. “Orthogonal Gradient Descent for Continual Learning.” 2019.
4. Luo et al. “An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning.” 2023.
5. Yang et al. “Qwen3 Technical Report.” 2025.
6. Meng et al. “PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models.” 2024.
