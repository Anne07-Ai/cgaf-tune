# Reproducibility checklist

## Artifacts

- [x] Public source repository and Apache-2.0 license
- [x] Fixed model identifier and QLoRA configuration
- [x] Dataset identifier and deterministic split seed
- [x] Train, validation, and final-test roles separated
- [x] Baseline implementations share model and adapter budgets
- [x] Training seeds 42, 123, and 456 reported
- [x] Early-stopping rule and best-checkpoint restoration documented
- [x] Raw per-seed result summaries preserved
- [x] Statistical-analysis script and SVG figures preserved
- [x] Failed and negative findings reported
- [ ] Adapter checkpoints uploaded to a persistent model repository
- [ ] Independent rerun on a second hardware/software environment
- [ ] Larger evaluation set and human/factuality assessment
- [ ] Archival paper identifier (arXiv or venue DOI)

## Final experiment

- Base model: `Qwen/Qwen3-0.6B`
- Hardware: Tesla T4
- Quantization: 4-bit QLoRA
- Dataset: `databricks/databricks-dolly-15k`
- Split shuffle seed: 2026
- Training seeds: 42, 123, 456
- Training examples: 512 per objective
- Validation examples: 64 per objective
- Final-test examples: 24 per objective
- Maximum steps: 1,200
- Validation interval: 100
- Early-stopping patience: 3
- Minimum validation improvement: 0.002
- Decoding: deterministic, 96 maximum new tokens
- Metrics: domain token F1, anchor token F1, harmonic F1

## Reproduction commands

Install and validate:

```bash
pip install -e ".[dev]"
pytest
python scripts/train.py --config configs/phase6_validation.yaml --dry-run
```

Run one method with separated validation data:

```bash
python scripts/train.py \
  --config configs/phase6_validation.yaml \
  --domain-data data/domain.jsonl \
  --anchor-data data/anchor.jsonl \
  --validation-data data/validation.jsonl \
  --method cgaf \
  --output-dir outputs/cgaf-seed42 \
  --max-steps 1200
```

Regenerate Phase 7 statistics and figures:

```bash
python scripts/analyze_phase7.py \
  --input results/phase6-three-seed-summary.json \
  --output-dir outputs/phase7
```

## Known reproducibility gap

The completed Hugging Face Jobs used ephemeral filesystems and no write-scoped Hub credential. Adapter weights were not persisted. Numerical outputs and job references are preserved, but exact checkpoint reuse is unavailable. A release-quality rerun must upload each best adapter immediately after evaluation.
