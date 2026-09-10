# CGAF-Tune Architecture

## System view

```mermaid
flowchart TD
    D["Domain stream"] --> BD["Domain batch"]
    A["Retention dataset"] --> BA["Anchor batch"]
    BD --> Q["Frozen 4-bit Qwen3"]
    BA --> Q
    Q --> L["Trainable LoRA adapters"]
    L --> C["Layer-local gradient capture"]
    C --> G["Smooth conflict gate"]
    G --> P["Conditional projection"]
    P --> O["Optimizer step"]
    G --> T["Conflict traces"]
```

Only adapter gradients are modified; base-model weights remain frozen. Domain and anchor gradients are grouped by Transformer layer or target projection. A negative cosine activates a temperature-controlled gate that removes part of the anchor-opposing component.

## Training event sequence

```mermaid
sequenceDiagram
    participant T as Trainer
    participant M as Model + LoRA
    participant C as CGAF controller
    participant O as Optimizer
    T->>M: Domain forward/backward
    M-->>C: Domain adapter gradients
    T->>M: Anchor forward/backward
    M-->>C: Anchor adapter gradients
    C->>C: Cosine and smooth gate per group
    C-->>M: Projected domain gradients
    T->>O: Step and clear gradients
```

## Experiment lifecycle

```mermaid
flowchart LR
    C["Pinned config"] --> R["Seeded runs"]
    R --> E["Domain evaluation"]
    R --> K["Retention evaluation"]
    E --> P["Pareto analysis"]
    K --> P
    P --> A["Ablations + report"]
```

## Current implementation boundary

Implemented and tested (Phase 1):

- stable per-group cosine measurement;
- smooth gated projection without input mutation;
- numerical tests for aligned, conflicting, and grouped gradients;
- deterministic PEFT parameter grouping and gradient capture;
- two-pass CGAF optimizer-step engine with structured diagnostics;
- configuration validation, dry-run, and offline smoke-test entry points.

Next milestone (Phase 2):

- load JSONL/Hugging Face domain and anchor datasets;
- build the Qwen3 QLoRA model and tokenizer path;
- log conflict, gate activity, removed norm, memory, and time;
- run matched LoRA, rehearsal, hard-projection, and AdaLoRA controls.

This boundary is intentional: the repository does not represent the end-to-end LLM trainer as complete before integration and GPU validation.
