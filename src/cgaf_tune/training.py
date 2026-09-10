"""One optimizer-step engine for conflict-gated adapter fine-tuning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

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
    method: str
    domain_loss: float
    anchor_loss: float | None
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
            method="cgaf",
            domain_loss=float(domain_loss.detach()),
            anchor_loss=float(anchor_loss.detach()),
            domain_gradient_norm=gradient_norm(domain_gradients),
            projected_gradient_norm=gradient_norm(projected),
            projected_groups=sum(stats.projected for stats in statistics.values()),
            mean_conflict_cosine=sum(cosines) / len(cosines),
            mean_gate=sum(gates) / len(gates),
            groups=statistics,
        )


class LoRAStepEngine:
    """Matched plain-LoRA optimizer step without anchor use."""

    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer,
        max_gradient_norm: float | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.max_gradient_norm = max_gradient_norm
        self.parameters = [parameter for _, parameter in trainable_named_parameters(model)]

    def step(self, domain_loss_fn: Callable[[], Tensor], anchor_loss_fn: Callable[[], Tensor]) -> StepResult:
        del anchor_loss_fn
        self.optimizer.zero_grad(set_to_none=True)
        loss = domain_loss_fn()
        _validate_loss(loss, "domain")
        loss.backward()
        gradients = {"all": [parameter.grad.detach().clone() for parameter in self.parameters]}
        if self.max_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters, self.max_gradient_norm)
        self.optimizer.step()
        norm = gradient_norm(gradients)
        return StepResult("lora", float(loss.detach()), None, norm, norm, 0, 0.0, 0.0, {})


class RehearsalStepEngine(LoRAStepEngine):
    """Joint domain and anchor loss baseline."""

    def __init__(self, *args, anchor_weight: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if anchor_weight < 0:
            raise ValueError("anchor_weight must be non-negative")
        self.anchor_weight = anchor_weight

    def step(self, domain_loss_fn: Callable[[], Tensor], anchor_loss_fn: Callable[[], Tensor]) -> StepResult:
        self.optimizer.zero_grad(set_to_none=True)
        domain_loss, anchor_loss = domain_loss_fn(), anchor_loss_fn()
        _validate_loss(domain_loss, "domain")
        _validate_loss(anchor_loss, "anchor")
        (domain_loss + self.anchor_weight * anchor_loss).backward()
        gradients = {"all": [parameter.grad.detach().clone() for parameter in self.parameters]}
        if self.max_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters, self.max_gradient_norm)
        self.optimizer.step()
        norm = gradient_norm(gradients)
        return StepResult(
            "rehearsal", float(domain_loss.detach()), float(anchor_loss.detach()),
            norm, norm, 0, 0.0, 0.0, {},
        )


class HardProjectionStepEngine(CGAFStepEngine):
    """Hard adapter-space projection baseline (CGAF gate fixed to one)."""

    def __init__(self, *args, epsilon: float = 1e-12, **kwargs) -> None:
        super().__init__(*args, config=CGAFConfig(temperature=1e-6, epsilon=epsilon), **kwargs)

    def step(self, domain_loss_fn: Callable[[], Tensor], anchor_loss_fn: Callable[[], Tensor]) -> StepResult:
        result = super().step(domain_loss_fn, anchor_loss_fn)
        return replace(result, method="hard_projection")


def build_step_engine(
    method: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: CGAFConfig,
    grouping: str,
    max_gradient_norm: float | None,
    anchor_weight: float = 1.0,
):
    """Create a matched training method from configuration."""
    common = {"model": model, "optimizer": optimizer, "max_gradient_norm": max_gradient_norm}
    if method == "cgaf":
        return CGAFStepEngine(config=config, grouping=grouping, **common)
    if method == "lora":
        return LoRAStepEngine(**common)
    if method == "rehearsal":
        return RehearsalStepEngine(anchor_weight=anchor_weight, **common)
    if method == "hard_projection":
        return HardProjectionStepEngine(grouping=grouping, epsilon=config.epsilon, **common)
    raise ValueError(f"unsupported training method: {method}")


def _validate_loss(loss: Tensor, label: str) -> None:
    if loss.ndim != 0:
        raise ValueError(f"{label} loss must be a scalar")
    if not torch.isfinite(loss):
        raise FloatingPointError(f"{label} loss is not finite")
