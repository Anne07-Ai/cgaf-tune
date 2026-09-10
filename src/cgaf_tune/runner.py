"""End-to-end local-data training loop for CGAF-Tune."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import CausalLMCollator, JsonlTextDataset
from .hf import build_model_and_tokenizer
from .projection import CGAFConfig
from .training import build_step_engine


def _cycle(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(parameter.device for parameter in model.parameters())


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def run_training(
    config: dict[str, Any],
    domain_path: str | Path,
    anchor_path: str | Path,
    output_dir: str | Path,
    max_steps: int | None = None,
) -> Path:
    """Train CGAF adapters and return the output directory."""
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model, tokenizer = build_model_and_tokenizer(config)
    training = config["training"]
    collator = CausalLMCollator(tokenizer, int(training["max_length"]))
    generator = torch.Generator().manual_seed(seed)
    domain_loader = DataLoader(
        JsonlTextDataset(domain_path),
        batch_size=int(training["per_device_batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )
    anchor_batch_size = max(1, round(
        int(training["per_device_batch_size"]) * float(training.get("anchor_ratio", 1.0))
    ))
    anchor_loader = DataLoader(
        JsonlTextDataset(anchor_path),
        batch_size=anchor_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
    )
    cgaf = config["cgaf"]
    method = str(training.get("method", "cgaf"))
    engine = build_step_engine(
        method,
        model,
        optimizer,
        config=CGAFConfig(
            temperature=float(cgaf["temperature"]),
            epsilon=float(cgaf["epsilon"]),
            minimum_conflict=float(cgaf["minimum_conflict"]),
        ),
        grouping=str(cgaf.get("grouping", "layer")),
        max_gradient_norm=training.get("max_gradient_norm", 1.0),
        anchor_weight=float(training.get("anchor_weight", 1.0)),
    )
    device = _model_device(model)
    anchor_batches = _cycle(anchor_loader)
    configured_steps = len(domain_loader) * int(training["epochs"])
    step_limit = min(configured_steps, max_steps) if max_steps else configured_steps
    metrics_path = output / "metrics.jsonl"
    started = time.perf_counter()
    completed = 0
    model.train()
    with metrics_path.open("w", encoding="utf-8") as metrics:
        while completed < step_limit:
            for domain_batch in domain_loader:
                if completed >= step_limit:
                    break
                domain = _move(domain_batch, device)
                anchor = _move(next(anchor_batches), device)
                result = engine.step(
                    lambda domain=domain: model(**domain).loss,
                    lambda anchor=anchor: model(**anchor).loss,
                )
                completed += 1
                record = result.to_dict()
                record.update({"step": completed, "elapsed_seconds": time.perf_counter() - started})
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()

    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    with (output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return output
