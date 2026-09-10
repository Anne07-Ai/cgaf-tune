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


def config():
    return {"experiment": {"seed": 3}, "model": {"name": "tiny"}, "lora": {}, "training": {"max_length": 8, "per_device_batch_size": 1, "anchor_ratio": 1.0, "learning_rate": 0.1, "epochs": 10, "max_gradient_norm": 1.0}, "cgaf": {"temperature": 0.1, "epsilon": 1e-12, "minimum_conflict": 0.0, "grouping": "global"}, "evaluation": {"validation_interval": 1, "early_stopping_patience": 1, "early_stopping_min_delta": 100.0}}


def test_training_runner_early_stops_and_writes_summary(tmp_path, monkeypatch):
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"text": "hello world"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(runner, "build_model_and_tokenizer", lambda unused: (TinyLM(), TinyTokenizer()))
    output = runner.run_training(config(), data, data, tmp_path / "output", max_steps=10, validation_path=data)
    summary = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["stop_reason"] == "early_stopping"
    assert summary["completed_steps"] == 2
    assert summary["best_step"] == 1
    assert (output / "adapter" / "adapter.bin").is_file()


def test_evaluate_loss_restores_training_mode():
    model = TinyLM()
    model.train()
    batch = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long), "labels": torch.tensor([[1, 2]])}
    assert runner.evaluate_loss(model, [batch], torch.device("cpu")) >= 0
    assert model.training
