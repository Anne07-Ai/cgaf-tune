"""End-to-end training with validation checkpoints and early stopping."""

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


def evaluate_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return example-weighted mean validation loss without changing training mode."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            moved = _move(batch, device)
            batch_size = int(moved["input_ids"].shape[0])
            total += float(model(**moved).loss) * batch_size
            count += batch_size
    model.train(was_training)
    if count == 0:
        raise ValueError("validation dataset is empty")
    return total / count


def _trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_trainable_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device))


def run_training(
    config: dict[str, Any],
    domain_path: str | Path,
    anchor_path: str | Path,
    output_dir: str | Path,
    max_steps: int | None = None,
    validation_path: str | Path | None = None,
) -> Path:
    """Train adapters, optionally selecting the best checkpoint on validation loss."""
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
    domain_loader = DataLoader(JsonlTextDataset(domain_path), batch_size=int(training["per_device_batch_size"]), shuffle=True, generator=generator, collate_fn=collator)
    anchor_batch_size = max(1, round(int(training["per_device_batch_size"]) * float(training.get("anchor_ratio", 1.0))))
    anchor_loader = DataLoader(JsonlTextDataset(anchor_path), batch_size=anchor_batch_size, shuffle=True, generator=generator, collate_fn=collator)
    validation_loader = None
    if validation_path is not None:
        validation_loader = DataLoader(JsonlTextDataset(validation_path), batch_size=int(training.get("evaluation_batch_size", training["per_device_batch_size"])), shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=float(training["learning_rate"]))
    cgaf = config["cgaf"]
    engine = build_step_engine(str(training.get("method", "cgaf")), model, optimizer, config=CGAFConfig(temperature=float(cgaf["temperature"]), epsilon=float(cgaf["epsilon"]), minimum_conflict=float(cgaf["minimum_conflict"])), grouping=str(cgaf.get("grouping", "layer")), max_gradient_norm=training.get("max_gradient_norm", 1.0), anchor_weight=float(training.get("anchor_weight", 1.0)))
    device = _model_device(model)
    anchor_batches = _cycle(anchor_loader)
    configured_steps = len(domain_loader) * int(training["epochs"])
    step_limit = min(configured_steps, max_steps) if max_steps else configured_steps
    evaluation = config.get("evaluation", {})
    interval = max(1, int(evaluation.get("validation_interval", 50)))
    patience = max(1, int(evaluation.get("early_stopping_patience", 3)))
    min_delta = float(evaluation.get("early_stopping_min_delta", 0.0))
    metrics_path = output / "metrics.jsonl"
    started, completed = time.perf_counter(), 0
    best_loss, best_step, stale = float("inf"), 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    stop_reason = "step_limit"
    model.train()
    with metrics_path.open("w", encoding="utf-8") as metrics:
        while completed < step_limit:
            for domain_batch in domain_loader:
                if completed >= step_limit:
                    break
                domain, anchor = _move(domain_batch, device), _move(next(anchor_batches), device)
                result = engine.step(lambda domain=domain: model(**domain).loss, lambda anchor=anchor: model(**anchor).loss)
                completed += 1
                record = result.to_dict()
                record.update({"step": completed, "elapsed_seconds": time.perf_counter() - started})
                should_validate = validation_loader is not None and (completed % interval == 0 or completed == step_limit)
                if should_validate:
                    validation_loss = evaluate_loss(model, validation_loader, device)
                    record["validation_loss"] = validation_loss
                    if validation_loss < best_loss - min_delta:
                        best_loss, best_step, stale = validation_loss, completed, 0
                        best_state = _trainable_state(model)
                    else:
                        stale += 1
                    record.update({"best_validation_loss": best_loss, "best_step": best_step, "stale_validations": stale})
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                if should_validate and stale >= patience:
                    stop_reason = "early_stopping"
                    break
            if stop_reason == "early_stopping":
                break

    if best_state is not None:
        _restore_trainable_state(model, best_state)
    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    summary = {"completed_steps": completed, "planned_steps": step_limit, "stop_reason": stop_reason, "best_step": best_step or None, "best_validation_loss": best_loss if best_state is not None else None, "elapsed_seconds": time.perf_counter() - started}
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return output
