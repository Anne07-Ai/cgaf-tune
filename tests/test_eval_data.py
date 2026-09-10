import json

import pytest

from cgaf_tune.eval_data import (
    exact_match,
    load_evaluation_jsonl,
    normalize_answer,
    token_f1,
)


def test_normalized_answer_metrics():
    assert normalize_answer("The, Quick Fox!") == "quick fox"
    assert exact_match("An answer.", "answer") == 1.0
    assert token_f1("red blue", "red green") == pytest.approx(0.5)


def test_load_evaluation_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "eval.jsonl"
    row = {"id": "one", "prompt": "p", "reference": "r", "category": "c"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate id"):
        load_evaluation_jsonl(path)
