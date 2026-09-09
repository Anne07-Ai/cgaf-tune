# Evaluation protocol

## Decision criterion

CGAF succeeds only if it improves the adaptation–retention Pareto frontier under matched model, data, trainable-parameter, optimizer-step, and evaluation budgets. A single best score is insufficient.

## Metrics

Let (D_0,R_0) denote domain and retention scores before tuning and (D_1,R_1) after tuning.

- domain gain: (D_1-D_0)
- forgetting: (R_0-R_1) (lower is better)
- normalized adaptation: ((D_1-D_0)/(100-D_0))
- retention ratio: (R_1/R_0)
- harmonic trade-off: harmonic mean of normalized adaptation and retention ratio
- systems: peak allocated GPU memory, examples/second, and wall-clock time
- mechanism: fraction of gated groups, gate mean, and conflict distribution by depth

Report the raw component scores as well as aggregates.

## Matched baselines

1. frozen Qwen3 base model
2. LoRA/QLoRA
3. LoRA plus anchor rehearsal loss
4. hard PCGrad-style adapter projection
5. AdaLoRA
6. CGAF

All baselines use identical splits, token budgets, target modules, seeds, stopping rules, and hyperparameter-search budgets.

## Suggested tasks

The first release should use one specialization domain and several retention categories:

- domain: public, license-compatible instruction or classification dataset
- retention: general instruction following, reasoning, factual QA, and multilingual prompts
- leakage audit: exact and near-duplicate detection across train and evaluation data

Dataset selection remains configuration-driven so the method is not tailored to a single benchmark.

## Ablations

| Factor | Values |
|---|---|
| grouping | global, layer, module |
| temperature | 0.03, 0.1, 0.3, hard |
| anchor ratio | 0, 1/16, 1/8, 1/4 |
| projection interval | 1, 4, 16 steps |
| rank | 8, 16, 32 |
| anchor composition | fixed, stratified, mismatched |
| quantization | bf16 LoRA, 4-bit QLoRA |

## Statistical reporting

Run at least three fixed seeds. Report mean, standard deviation, paired differences against LoRA, and bootstrap 95% confidence intervals over evaluation examples. Publish failures and all predeclared primary outcomes. Do not call an outcome state of the art without a broad, current baseline study.

## Stop/go threshold for the pilot

Proceed to larger models only if CGAF:

- matches LoRA domain performance within a predeclared practical margin;
- reduces mean forgetting across retention categories;
- is not dominated by rehearsal or hard projection;
- adds no more than 35% wall-clock overhead in the pilot.

These thresholds should be frozen before examining final test results.
