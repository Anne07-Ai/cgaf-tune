import pytest

from cgaf_tune.eval_data import EvaluationExample
from cgaf_tune.predictions import score_predictions


def test_per_example_and_category_scores():
    examples = [
        EvaluationExample("1", "p1", "alpha", "facts"),
        EvaluationExample("2", "p2", "red blue", "reasoning"),
    ]
    result = score_predictions(examples, ["Alpha.", "red green"], bootstrap_samples=200)
    assert len(result["examples"]) == 2
    assert result["overall"]["exact_match"]["mean"] == pytest.approx(0.5)
    assert result["categories"]["facts"]["exact_match"]["mean"] == 1.0
    assert result["categories"]["reasoning"]["token_f1"]["mean"] == pytest.approx(0.5)


def test_prediction_count_must_match():
    with pytest.raises(ValueError, match="equal length"):
        score_predictions([EvaluationExample("1", "p", "r", "c")], [])
