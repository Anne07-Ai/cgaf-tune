# Phase 6: validation-controlled scaling

Phase 6 replaces fixed long adaptation with held-out checkpoint selection. Its purpose is to test
whether CGAF's adaptation-retention trade-off survives a larger natural-data regime without the
over-training observed in the 500-step, 96-example experiment.

## Protocol

- Use disjoint train, validation, and final-test splits.
- Increase the natural training pool; never select checkpoints on final-test examples.
- Evaluate validation loss every 50 optimizer steps.
- Stop after three validations without an improvement of at least 0.002.
- Restore the best adapter weights before saving and generation-based final evaluation.
- Match data, optimizer, maximum-step, seed, and evaluation budgets across all four methods.

The runner writes `training_summary.json` with the planned/completed steps, stop reason, best step,
best validation loss, and wall time. `metrics.jsonl` records every validation decision.

## Success criteria

CGAF advances only if its three-seed harmonic generation F1 improves over hard projection and its
retention score is competitive with rehearsal. Loss and generation metrics must be reported
together; disagreement is treated as evidence, not hidden.
