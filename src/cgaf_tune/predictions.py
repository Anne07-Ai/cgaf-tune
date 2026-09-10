"""Batched generation and per-example evaluation records."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

import torch

from .eval_data import EvaluationExample, exact_match, token_f1
from .evaluation import bootstrap_mean_ci


def generate_predictions(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[EvaluationExample],
    batch_size: int = 4,
    max_new_tokens: int = 64,
) -> list[str]:
    """Generate answers while decoding only newly generated tokens."""
    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError("batch size and max new tokens must be positive")
    device = next(model.parameters()).device
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            prompts = [_format_prompt(tokenizer, example.prompt) for example in batch]
            encoded = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            prompt_width = encoded["input_ids"].shape[1]
            outputs.extend(tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True))
    return [output.strip() for output in outputs]


def score_predictions(
    examples: list[EvaluationExample], predictions: list[str],
    bootstrap_samples: int = 10_000, seed: int = 17,
) -> dict[str, object]:
    """Return auditable per-example scores, category means, and example-level CIs."""
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have equal length")
    records = []
    categories: dict[str, list[dict[str, float]]] = defaultdict(list)
    for example, prediction in zip(examples, predictions, strict=True):
        scores = {
            "exact_match": exact_match(prediction, example.reference),
            "token_f1": token_f1(prediction, example.reference),
        }
        records.append({
            "id": example.id, "category": example.category, "prompt": example.prompt,
            "reference": example.reference, "prediction": prediction, **scores,
        })
        categories[example.category].append(scores)
    return {
        "examples": records,
        "overall": _aggregate([{"exact_match": row["exact_match"], "token_f1": row["token_f1"]}
                               for row in records], bootstrap_samples, seed),
        "categories": {
            category: _aggregate(values, bootstrap_samples, seed)
            for category, values in sorted(categories.items())
        },
    }


def _aggregate(rows: list[dict[str, float]], samples: int, seed: int) -> dict[str, dict[str, float]]:
    result = {}
    for offset, metric in enumerate(("exact_match", "token_f1")):
        values = [row[metric] for row in rows]
        low, high = bootstrap_mean_ci(values, samples=samples, seed=seed + offset)
        result[metric] = {
            "mean": statistics.fmean(values), "ci_low": low, "ci_high": high,
            "examples": float(len(values)),
        }
    return result


def _format_prompt(tokenizer: Any, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    return f"User: {prompt}\nAssistant:"
