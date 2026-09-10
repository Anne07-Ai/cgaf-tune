import pytest

from cgaf_tune.eval_data import EvaluationExample
from cgaf_tune.leakage import audit_leakage


def test_exact_and_near_leakage_are_detected():
    examples = [EvaluationExample("e1", "public bucket", "restrict access", "domain")]
    findings = audit_leakage(
        ["Public bucket restrict access", "public bucket restrict access immediately"],
        examples,
        threshold=0.7,
    )
    assert [finding.kind for finding in findings] == ["exact", "near"]


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError, match="threshold"):
        audit_leakage([], [], threshold=0)
