"""Gradient capture and restoration utilities for trainable adapter parameters."""

from __future__ import annotations

import re
from collections.abc import Iterable

from torch import Tensor, nn

_LAYER_PATTERN = re.compile(r"(?:layers|layer|h|block|blocks)\.(\d+)")


def trainable_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Return trainable parameters in deterministic model order."""
    parameters = [(name, parameter) for name, parameter in model.named_parameters()
                  if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return parameters


def group_name(parameter_name: str, grouping: str = "layer") -> str:
    """Map a PEFT parameter name to a stable conflict-projection group."""
    if grouping == "global":
        return "global"
    if grouping == "module":
        parts = parameter_name.split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else parameter_name
    if grouping != "layer":
        raise ValueError("grouping must be one of: global, layer, module")
    match = _LAYER_PATTERN.search(parameter_name)
    return f"layer.{match.group(1)}" if match else "unscoped"


def parameter_groups(
    named_parameters: Iterable[tuple[str, nn.Parameter]], grouping: str = "layer"
) -> dict[str, list[nn.Parameter]]:
    """Group parameters while preserving their original order."""
    groups: dict[str, list[nn.Parameter]] = {}
    for name, parameter in named_parameters:
        groups.setdefault(group_name(name, grouping), []).append(parameter)
    return groups


def capture_gradients(groups: dict[str, list[nn.Parameter]]) -> dict[str, list[Tensor]]:
    """Clone current gradients, failing if a trainable parameter received none."""
    captured: dict[str, list[Tensor]] = {}
    for name, parameters in groups.items():
        gradients = []
        for parameter in parameters:
            if parameter.grad is None:
                raise RuntimeError(f"missing gradient in group {name!r}")
            gradients.append(parameter.grad.detach().clone())
        captured[name] = gradients
    return captured


def restore_gradients(
    groups: dict[str, list[nn.Parameter]], gradients: dict[str, list[Tensor]]
) -> None:
    """Restore captured gradients to their matching parameters."""
    if groups.keys() != gradients.keys():
        raise ValueError("parameter and gradient groups must match")
    for name, parameters in groups.items():
        values = gradients[name]
        if len(parameters) != len(values):
            raise ValueError(f"gradient count differs for group {name!r}")
        for parameter, gradient in zip(parameters, values, strict=True):
            if parameter.shape != gradient.shape:
                raise ValueError(f"gradient shape differs in group {name!r}")
            parameter.grad = gradient.detach().to(device=parameter.device, dtype=parameter.dtype)


def gradient_norm(gradients: dict[str, list[Tensor]]) -> float:
    """Compute the global L2 norm without concatenating all tensors."""
    squared = sum(float(tensor.detach().float().square().sum())
                  for group in gradients.values() for tensor in group)
    return squared ** 0.5
