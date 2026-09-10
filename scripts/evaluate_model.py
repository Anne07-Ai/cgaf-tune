"""Generate and score per-example predictions for a base model or PEFT adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from cgaf_tune.eval_data import load_evaluation_jsonl
from cgaf_tune.hf import load_evaluation_model
from cgaf_tune.predictions import generate_predictions, score_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    examples = load_evaluation_jsonl(args.data)
    model, tokenizer = load_evaluation_model(
        config, str(args.adapter) if args.adapter else None
    )
    predictions = generate_predictions(
        model, tokenizer, examples, args.batch_size, args.max_new_tokens
    )
    result = score_predictions(examples, predictions, args.bootstrap_samples)
    result["metadata"] = {
        "model": config["model"]["name"],
        "adapter": str(args.adapter) if args.adapter else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"per-example evaluation: {args.output}")


if __name__ == "__main__":
    main()
