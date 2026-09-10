import json

import pytest

from cgaf_tune.experiments import build_matrix, run_matrix


def test_matrix_is_deterministic():
    matrix = build_matrix(["lora", "cgaf"], [2, 1])
    assert [run["run_id"] for run in matrix] == [
        "lora-seed2", "lora-seed1", "cgaf-seed2", "cgaf-seed1"
    ]


def test_matrix_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicates"):
        build_matrix(["cgaf", "cgaf"], [1])


def test_run_matrix_records_success_and_failure(tmp_path):
    calls = []

    def fake_train(config, domain, anchor, output, max_steps):
        del domain, anchor, max_steps
        calls.append((config["training"]["method"], config["experiment"]["seed"]))
        if config["training"]["method"] == "rehearsal":
            raise RuntimeError("expected failure")
        return output

    config = {"experiment": {"name": "pilot", "seed": 0}, "training": {"method": "cgaf"}}
    manifest_path = run_matrix(
        config, "domain.jsonl", "anchor.jsonl", tmp_path,
        ["lora", "rehearsal"], [7], train_fn=fake_train,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert calls == [("lora", 7), ("rehearsal", 7)]
    assert [run["status"] for run in manifest["runs"]] == ["completed", "failed"]
    assert config["training"]["method"] == "cgaf"
