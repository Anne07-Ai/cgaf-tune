"""One optimizer-step engine for conflict-gated adapter fine-tuning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .gradients import (
    capture_gradients,
    gradient_norm,
    parameter_groups,
    restore_gradients,
    trainable_named_parameters,
)
from .projection import CGAFConfig, ProjectionStats, apply_cgaf_projection


@dataclass(frozen=True)
class StepResult:
    domain_loss: float
    anchor_loss: float
    domain_gradient_norm: float
    projected_gradient_norm: float
    projected_groups: int
    mean_conflict_cosine: float
    mean_gate: float
    groups: dict[str, ProjectionStats]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["groups"] = {name: asdict(stats) for name, stats in self.groups.items()}
        return payload


class CGAFStepEngine:
    """Execute domain and anchor backward passes, projection, and optimizer update."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: CGAFConfig | None = None,
        grouping: str = "layer",
        max_gradient_norm: float | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config or CGAFConfig()
        self.max_gradient_norm = max_gradient_norm
        self.groups = parameter_groups(trainable_named_parameters(model), grouping)

    def step(self, domain_loss_fn: Callable[[], Tensor], anchor_loss_fn: Callable[[], Tensor]) -> StepResult:
        """Apply one CGAF optimizer step using fresh forward passes from two closures."""
        self.optimizer.zero_grad(set_to_none=True)
        domain_loss = domain_loss_fn()
        _validate_loss(domain_loss, "domain")
        domain_loss.backward()
        domain_gradients = capture_gradients(self.groups)

        self.optimizer.zero_grad(set_to_none=True)
        anchor_loss = anchor_loss_fn()
        _validate_loss(anchor_loss, "anchor")
        anchor_loss.backward()
        anchor_gradients = capture_gradients(self.groups)

        projected, statistics = apply_cgaf_projection(
            domain_gradients, anchor_gradients, self.config
        )
        self.optimizer.zero_grad(set_to_none=True)
        restore_gradients(self.groups, projected)
        if self.max_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [parameter for values in self.groups.values() for parameter in values],
                self.max_gradient_norm,
            )
        self.optimizer.step()

        cosines = [stats.cosine for stats in statistics.values()]
        gates = [stats.gate for stats in statistics.values()]
        return StepResult(
            domain_loss=float(domain_loss.detach()),
            anchor_loss=float(anchor_loss.detach()),
            domain_gradient_norm=gradient_norm(domain_gradients),
            projected_gradient_norm=gradient_norm(projected),
            projected_groups=sum(stats.projected for stats in statistics.values()),
            mean_conflict_cosine=sum(cosines) / len(cosines),
            mean_gate=sum(gates) / len(gates),
            groups=statistics,
        )


def _validate_loss(loss: Tensor, label: str) -> None:
    if loss.ndim != 0:
        raise ValueError(f"{label} loss must be a scalar")
    if not torch.isfinite(loss):
        raise FloatingPointError(f"{label} loss is not finite")
