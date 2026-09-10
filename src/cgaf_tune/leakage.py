"""Exact and token-Jaccard near-duplicate leakage detection."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from .eval_data import EvaluationExample, normalize_answer


@dataclass(frozen=True)
class LeakageFinding:
    evaluation_id: str
    training_index: int
    kind: str
    similarity: float


def audit_leakage(
    training_texts: list[str], examples: list[EvaluationExample], threshold: float = 0.85
) -> list[LeakageFinding]:
    """Compare normalized evaluation prompt/reference text with every training record."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    normalized_training = [normalize_answer(text) for text in training_texts]
    hashes: dict[str, list[int]] = {}
    for index, text in enumerate(normalized_training):
        hashes.setdefault(_digest(text), []).append(index)
    findings = []
    for example in examples:
        evaluation_text = normalize_answer(f"{example.prompt} {example.reference}")
        exact_indices = set(hashes.get(_digest(evaluation_text), []))
        for index in exact_indices:
            findings.append(LeakageFinding(example.id, index, "exact", 1.0))
        evaluation_tokens = set(evaluation_text.split())
        for index, training_text in enumerate(normalized_training):
            if index in exact_indices:
                continue
            similarity = _jaccard(evaluation_tokens, set(training_text.split()))
            if similarity >= threshold:
                findings.append(LeakageFinding(example.id, index, "near", similarity))
    return sorted(findings, key=lambda finding: (finding.evaluation_id, -finding.similarity))


def findings_to_dict(findings: list[LeakageFinding]) -> list[dict]:
    return [asdict(finding) for finding in findings]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0
