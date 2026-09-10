"""CGAF-Tune research prototype."""

from .projection import CGAFConfig, ProjectionStats, apply_cgaf_projection, apply_hard_projection
from .training import (
    CGAFStepEngine,
    HardProjectionStepEngine,
    LoRAStepEngine,
    RehearsalStepEngine,
    StepResult,
    build_step_engine,
)

__all__ = [
    "CGAFConfig",
    "CGAFStepEngine",
    "HardProjectionStepEngine",
    "LoRAStepEngine",
    "ProjectionStats",
    "RehearsalStepEngine",
    "StepResult",
    "apply_cgaf_projection",
    "apply_hard_projection",
    "build_step_engine",
]
__version__ = "0.1.0"
