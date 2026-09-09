"""Conflict-gated projection for LoRA/QLoRA parameter gradients."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CGAFConfig:
    """Numerically stable gate configuration."""

    temperature: float = 0.1
    epsilon: float = 1e-12
    minimum_conflict: float = 0.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True)
class ProjectionStats:
    cosine: float
    gate: float
    projected: bool
    removed_norm: float


def _flatten(tensors: list[Tensor]) -> Tensor:
    if not tensors:
        raise ValueError("gradient group cannot be empty")
    return torch.cat([tensor.detach().reshape(-1).float() for tensor in tensors])


def apply_cgaf_projection(
    domain_gradients: Mapping[str, list[Tensor]],
    anchor_gradients: Mapping[str, list[Tensor]],
    config: CGAFConfig | None = None,
) -> tuple[dict[str, list[Tensor]], dict[str, ProjectionStats]]:
    """Project conflicting domain gradients group-by-group.

    Inputs use matching group names (normally Transformer layer or LoRA module).
    Returned tensors do not mutate the inputs.
    """

    config = config or CGAFConfig()
    if domain_gradients.keys() != anchor_gradients.keys():
        raise ValueError("domain and anchor groups must match")

    projected: dict[str, list[Tensor]] = {}
    statistics: dict[str, ProjectionStats] = {}

    for group, domain_group in domain_gradients.items():
        anchor_group = anchor_gradients[group]
        if len(domain_group) != len(anchor_group):
            raise ValueError(f"gradient count differs for group {group!r}")

        domain = _flatten(domain_group)
        anchor = _flatten(anchor_group)
        dot = torch.dot(domain, anchor)
        denom = domain.norm() * anchor.norm() + config.epsilon
        cosine = dot / denom
        conflict = cosine < -config.minimum_conflict
        gate = torch.sigmoid(-cosine / config.temperature) if conflict else cosine.new_zeros(())

        coefficient = gate * torch.minimum(dot, dot.new_zeros(()))
        coefficient = coefficient / (anchor.square().sum() + config.epsilon)
        flat_projected = domain - coefficient * anchor

        result_group: list[Tensor] = []
        cursor = 0
        for original in domain_group:
            size = original.numel()
            result_group.append(
                flat_projected[cursor : cursor + size].reshape_as(original).to(original.dtype)
            )
            cursor += size

        removed_norm = (domain - flat_projected).norm()
        projected[group] = result_group
        statistics[group] = ProjectionStats(
            cosine=float(cosine),
            gate=float(gate),
            projected=bool(conflict),
            removed_norm=float(removed_norm),
        )

    return projected, statistics
