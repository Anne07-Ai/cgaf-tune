import pytest

from cgaf_tune.evaluation import compute_tradeoff, summarize_methods


def test_tradeoff_metrics():
    result = compute_tradeoff(
        0.4, 0.7,
        {"reasoning": 0.8, "facts": 0.6},
        {"reasoning": 0.7, "facts": 0.55},
    )
    assert result.domain_gain == pytest.approx(0.3)
    assert result.normalized_adaptation == pytest.approx(0.5)
    assert result.mean_forgetting == pytest.approx(0.075)
    assert result.harmonic_tradeoff > 0


def test_categories_must_match():
    with pytest.raises(ValueError, match="match"):
        compute_tradeoff(0.2, 0.3, {"a": 0.5}, {"b": 0.5})


def test_multi_seed_summary():
    summary = summarize_methods([
        {"method": "cgaf", "seed": 1, "metrics": {"domain_gain": 0.2}},
        {"method": "cgaf", "seed": 2, "metrics": {"domain_gain": 0.4}},
    ])
    assert summary["cgaf"]["domain_gain"]["mean"] == pytest.approx(0.3)
    assert summary["cgaf"]["domain_gain"]["runs"] == 2
