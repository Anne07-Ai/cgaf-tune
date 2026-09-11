# CGAF-Tune v0.1 research release

## Included

- CGAF gradient engine with global, layer, and module grouping
- LoRA, rehearsal, and hard-projection matched baselines
- Qwen3 QLoRA runner with validation checkpoints and early stopping
- controlled-conflict, natural-data, ablation, and three-seed experiments
- leakage-separated generation evaluation
- raw result summaries, bootstrap analysis, Pareto and seed-sensitivity figures
- research paper draft, research card, citation metadata, and reproducibility checklist

## Primary finding

CGAF reached the highest mean domain F1 but did not exceed LoRA or rehearsal on balanced harmonic F1. The release intentionally preserves this negative result.

## Publication status

The repository release package is complete. Hugging Face model publication is pending a write-scoped Hub connection and a persistence-enabled rerun of the selected adapter checkpoints. arXiv/Hugging Face Paper Pages publication requires an archival paper submission and identifier.
