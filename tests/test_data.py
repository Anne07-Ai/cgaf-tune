import json

import pytest
import torch

from cgaf_tune.data import CausalLMCollator, JsonlTextDataset, record_to_text


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        del kwargs
        width = max(len(text.split()) for text in texts)
        input_ids = []
        attention = []
        for text in texts:
            size = len(text.split())
            input_ids.append(list(range(1, size + 1)) + [0] * (width - size))
            attention.append([1] * size + [0] * (width - size))
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention),
        }


def test_supported_record_formats():
    assert record_to_text({"text": "hello"}) == "hello"
    assert "Assistant: answer" in record_to_text({"prompt": "question", "response": "answer"})
    assert record_to_text({"messages": [{"role": "user", "content": "hello"}]}) == "User: hello"


def test_jsonl_reports_invalid_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"text": "valid"}) + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        JsonlTextDataset(path)


def test_collator_masks_padding():
    batch = CausalLMCollator(FakeTokenizer(), max_length=16)(["one two", "one"])
    assert batch["labels"].tolist() == [[1, 2], [1, -100]]
