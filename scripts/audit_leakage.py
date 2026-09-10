"""Audit exact and near-duplicate overlap between training and evaluation JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cgaf_tune.data import JsonlTextDataset
from cgaf_tune.eval_data import load_evaluation_jsonl
from cgaf_tune.leakage import audit_leakage, findings_to_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--evaluation-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--fail-on-leakage", action="store_true")
    args = parser.parse_args()
    training = JsonlTextDataset(args.training_data).texts
    evaluation = load_evaluation_jsonl(args.evaluation_data)
    findings = audit_leakage(training, evaluation, args.threshold)
    payload = {
        "training_examples": len(training), "evaluation_examples": len(evaluation),
        "threshold": args.threshold, "findings": findings_to_dict(findings),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"leakage findings: {len(findings)}")
    if findings and args.fail_on_leakage:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
