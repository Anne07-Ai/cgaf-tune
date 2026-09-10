import pytest

from cgaf_tune.evaluation import bootstrap_mean_ci, paired_differences, pareto_methods


def test_bootstrap_is_deterministic_and_contains_mean():
    first = bootstrap_mean_ci([0.1, 0.2, 0.3], samples=500, seed=9)
    second = bootstrap_mean_ci([0.1, 0.2, 0.3], samples=500, seed=9)
    assert first == second
    assert first[0] <= 0.2 <= first[1]


def test_paired_differences_use_common_seeds_only():
    records = [
        {"method": "lora", "seed": 1, "metrics": {"domain_gain": 0.2}},
        {"method": "lora", "seed": 2, "metrics": {"domain_gain": 0.3}},
        {"method": "cgaf", "seed": 1, "metrics": {"domain_gain": 0.25}},
        {"method": "cgaf", "seed": 3, "metrics": {"domain_gain": 0.9}},
    ]
    result = paired_differences(records, bootstrap_samples=100)
    assert result["cgaf"]["domain_gain"]["mean"] == pytest.approx(0.05)
    assert result["cgaf"]["domain_gain"]["runs"] == 1


def test_pareto_excludes_dominated_method():
    summary = {
        "a": {"domain_gain": {"mean": 0.3}, "mean_forgetting": {"mean": 0.1}},
        "b": {"domain_gain": {"mean": 0.2}, "mean_forgetting": {"mean": 0.2}},
        "c": {"domain_gain": {"mean": 0.4}, "mean_forgetting": {"mean": 0.3}},
    }
    assert pareto_methods(summary) == ["a", "c"]
