"""Validated per-example evaluation data and answer scoring."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

_ARTICLES = re.compile(r"\b(a|an|the)\b")


@dataclass(frozen=True)
class EvaluationExample:
    id: str
    prompt: str
    reference: str
    category: str


def load_evaluation_jsonl(path: str | Path) -> list[EvaluationExample]:
    """Load evaluation examples, requiring stable unique IDs and categories."""
    source = Path(path)
    examples = []
    seen = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                example = EvaluationExample(**{key: str(row[key]).strip() for key in asdict(
                    EvaluationExample("", "", "", "")
                )})
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"{source}:{line_number}: invalid evaluation record") from error
            if not all(asdict(example).values()):
                raise ValueError(f"{source}:{line_number}: fields cannot be empty")
            if example.id in seen:
                raise ValueError(f"{source}:{line_number}: duplicate id {example.id!r}")
            seen.add(example.id)
            examples.append(example)
    if not examples:
        raise ValueError(f"evaluation dataset is empty: {source}")
    return examples


def normalize_answer(text: str) -> str:
    """Lowercase and remove punctuation, English articles, and extra whitespace."""
    lowered = text.lower()
    without_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted, expected = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)
