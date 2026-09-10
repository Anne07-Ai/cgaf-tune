"""CGAF-Tune research prototype."""

from .projection import CGAFConfig, ProjectionStats, apply_cgaf_projection
from .training import CGAFStepEngine, StepResult

__all__ = [
    "CGAFConfig",
    "CGAFStepEngine",
    "ProjectionStats",
    "StepResult",
    "apply_cgaf_projection",
]
__version__ = "0.1.0"
