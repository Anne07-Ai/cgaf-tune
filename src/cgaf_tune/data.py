"""Local JSONL ingestion and causal-language-model collation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def record_to_text(record: dict[str, Any]) -> str:
    """Normalize supported research-data records into one training string."""
    if isinstance(record.get("text"), str):
        text = record["text"]
    elif isinstance(record.get("messages"), list):
        parts = []
        for message in record["messages"]:
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise TypeError("every message requires string content")
            role = str(message.get("role", "user")).strip().title()
            parts.append(f"{role}: {message['content'].strip()}")
        text = "\n".join(parts)
    elif isinstance(record.get("prompt"), str) and isinstance(record.get("response"), str):
        text = f"User: {record['prompt'].strip()}\nAssistant: {record['response'].strip()}"
    else:
        raise TypeError("record requires text, messages, or prompt and response")
    text = text.strip()
    if not text:
        raise ValueError("training text cannot be empty")
    return text


class JsonlTextDataset(Dataset[str]):
    """Eagerly validate a small or medium local JSONL research dataset."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.texts: list[str] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise TypeError("JSON value must be an object")
                    self.texts.append(record_to_text(record))
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise ValueError(f"{self.path}:{line_number}: {error}") from error
        if not self.texts:
            raise ValueError(f"dataset is empty: {self.path}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> str:
        return self.texts[index]


class CausalLMCollator:
    """Tokenize text and mask padding tokens from the causal-LM loss."""

    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, texts: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        encoded["labels"] = labels
        return encoded
