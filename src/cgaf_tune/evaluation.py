"""Adaptation-retention metrics and multi-seed comparison summaries."""

from __future__ import annotations

import math
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


def summarize_methods(records: Sequence[Mapping[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
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
            name: {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "runs": float(len(values)),
            }
            for name, values in metrics.items()
        }
        for method, metrics in grouped.items()
    }


def _harmonic(left: float, right: float) -> float:
    return 0.0 if left <= 0 or right <= 0 else 2 * left * right / (left + right)
