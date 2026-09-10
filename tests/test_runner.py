import json
from types import SimpleNamespace

import torch
from torch import nn

from cgaf_tune import runner


class TinyTokenizer:
    def __call__(self, texts, **kwargs):
        del kwargs
        ids = torch.tensor([[1, 2] for _ in texts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, input_ids, attention_mask, labels):
        del attention_mask
        target = labels[labels != -100].float().mean()
        prediction = self.weight * input_ids.float().mean()
        return SimpleNamespace(loss=(prediction - target).square())

    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "adapter.bin")


def test_training_runner_writes_metrics_and_adapter(tmp_path, monkeypatch):
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"text": "hello world"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(runner, "build_model_and_tokenizer", lambda config: (TinyLM(), TinyTokenizer()))
    config = {
        "experiment": {"seed": 3},
        "model": {"name": "tiny"},
        "lora": {},
        "training": {
            "max_length": 8,
            "per_device_batch_size": 1,
            "anchor_ratio": 1.0,
            "learning_rate": 0.1,
            "epochs": 1,
            "max_gradient_norm": 1.0,
        },
        "cgaf": {
            "temperature": 0.1,
            "epsilon": 1e-12,
            "minimum_conflict": 0.0,
            "grouping": "global",
        },
    }
    output = runner.run_training(config, data, data, tmp_path / "output", max_steps=1)
    records = (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["step"] == 1
    assert (output / "adapter" / "adapter.bin").is_file()
    assert (output / "run_config.json").is_file()
