"""Adaptation-retention metrics and multi-seed comparison summaries."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TradeoffMetrics:
    domain_before: float
    domain_after: float
    domain_gain: float
    normalized_adaptation: float
    mean_retention_before: float
    mean_retention_after: float
    mean_forgetting: float
    retention_ratio: float
    harmonic_tradeoff: float
    forgetting_by_category: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def compute_tradeoff(
    domain_before: float,
    domain_after: float,
    retention_before: Mapping[str, float],
    retention_after: Mapping[str, float],
    score_ceiling: float = 1.0,
) -> TradeoffMetrics:
    """Compute preregistered adaptation-retention metrics on a common score scale."""
    if retention_before.keys() != retention_after.keys() or not retention_before:
        raise ValueError("retention categories must be non-empty and match")
    values = [domain_before, domain_after, *retention_before.values(), *retention_after.values()]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("scores must be finite")
    if score_ceiling <= domain_before:
        raise ValueError("score ceiling must exceed the before-domain score")
    forgetting = {
        name: retention_before[name] - retention_after[name] for name in retention_before
    }
    mean_before = statistics.fmean(retention_before.values())
    mean_after = statistics.fmean(retention_after.values())
    gain = domain_after - domain_before
    adaptation = gain / (score_ceiling - domain_before)
    retention_ratio = mean_after / mean_before if mean_before else 0.0
    harmonic = _harmonic(max(0.0, adaptation), max(0.0, retention_ratio))
    return TradeoffMetrics(
        domain_before, domain_after, gain, adaptation, mean_before, mean_after,
        statistics.fmean(forgetting.values()), retention_ratio, harmonic, forgetting,
    )


def bootstrap_mean_ci(
    values: Sequence[float], samples: int = 10_000, confidence: float = 0.95, seed: int = 17
) -> tuple[float, float]:
    """Return a deterministic percentile-bootstrap confidence interval for a mean."""
    if not values:
        raise ValueError("bootstrap values cannot be empty")
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    generator = random.Random(seed)
    estimates = sorted(
        statistics.fmean(generator.choice(values) for _ in values) for _ in range(samples)
    )
    tail = (1 - confidence) / 2
    low = estimates[max(0, int(tail * samples))]
    high = estimates[min(samples - 1, int((1 - tail) * samples) - 1)]
    return low, high


def summarize_methods(
    records: Sequence[Mapping[str, object]], bootstrap_samples: int = 10_000
) -> dict[str, dict[str, dict[str, float]]]:
    """Summarize numeric metrics by method across independent seeds."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for record in records:
        method = str(record["method"])
        metrics = record["metrics"]
        if not isinstance(metrics, Mapping):
            raise TypeError("record metrics must be a mapping")
        target = grouped.setdefault(method, {})
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                target.setdefault(str(name), []).append(float(value))
    return {
        method: {
            name: _summary(values, bootstrap_samples)
            for name, values in metrics.items()
        }
        for method, metrics in grouped.items()
    }


def paired_differences(
    records: Sequence[Mapping[str, object]], baseline: str = "lora",
    bootstrap_samples: int = 10_000,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare each method with a baseline using only exactly matched seeds."""
    indexed: dict[str, dict[int, Mapping[str, object]]] = {}
    for record in records:
        indexed.setdefault(str(record["method"]), {})[int(record["seed"])] = record
    if baseline not in indexed:
        raise ValueError(f"baseline method not found: {baseline}")
    output = {}
    for method, runs in indexed.items():
        if method == baseline:
            continue
        common = sorted(runs.keys() & indexed[baseline].keys())
        metrics: dict[str, list[float]] = {}
        for seed in common:
            current = runs[seed]["metrics"]
            control = indexed[baseline][seed]["metrics"]
            if not isinstance(current, Mapping) or not isinstance(control, Mapping):
                raise TypeError("record metrics must be mappings")
            for name in current.keys() & control.keys():
                left, right = current[name], control[name]
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    metrics.setdefault(str(name), []).append(float(left) - float(right))
        output[method] = {name: _summary(values, bootstrap_samples) for name, values in metrics.items()}
    return output


def pareto_methods(summary: Mapping[str, Mapping[str, Mapping[str, float]]]) -> list[str]:
    """Return non-dominated methods: maximize gain and minimize forgetting."""
    points = {
        method: (metrics["domain_gain"]["mean"], metrics["mean_forgetting"]["mean"])
        for method, metrics in summary.items()
    }
    frontier = []
    for method, (gain, forgetting) in points.items():
        dominated = any(
            other_gain >= gain and other_forgetting <= forgetting
            and (other_gain > gain or other_forgetting < forgetting)
            for other, (other_gain, other_forgetting) in points.items() if other != method
        )
        if not dominated:
            frontier.append(method)
    return sorted(frontier)


def _summary(values: list[float], samples: int) -> dict[str, float]:
    low, high = bootstrap_mean_ci(values, samples=samples)
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "ci_low": low,
        "ci_high": high,
        "runs": float(len(values)),
    }


def _harmonic(left: float, right: float) -> float:
    return 0.0 if left <= 0 or right <= 0 else 2 * left * right / (left + right)
