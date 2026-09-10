"""Validated command-line entry point for CGAF-Tune."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch import nn

from cgaf_tune import CGAFConfig, CGAFStepEngine
from cgaf_tune.runner import run_training

REQUIRED = {"experiment", "model", "lora", "training", "cgaf", "evaluation"}


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = REQUIRED.difference(config)
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--domain-data", type=Path)
    parser.add_argument("--anchor-data", type=Path)
    parser.add_argument("--validation-data", type=Path, help="held-out JSONL used only for checkpoint selection")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cgaf-pilot"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--method", choices=["cgaf", "lora", "rehearsal", "hard_projection"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.method:
        config["training"]["method"] = args.method
    if args.dry_run:
        print(f"configuration valid: {config['experiment']['name']}")
        return
    if args.smoke_test:
        run_smoke_test(config)
        return
    if args.domain_data is None or args.anchor_data is None:
        parser.error("real training requires --domain-data and --anchor-data")
    destination = run_training(config, args.domain_data, args.anchor_data, args.output_dir, args.max_steps, args.validation_data)
    print(f"training complete: {destination}")


def run_smoke_test(config: dict) -> None:
    torch.manual_seed(config["experiment"]["seed"])
    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    engine = CGAFStepEngine(model, optimizer, CGAFConfig(temperature=config["cgaf"]["temperature"], epsilon=config["cgaf"]["epsilon"], minimum_conflict=config["cgaf"]["minimum_conflict"]), grouping="global")
    result = engine.step(lambda: torch.nn.functional.mse_loss(model(torch.randn(4, 4)), torch.ones(4, 2)), lambda: torch.nn.functional.mse_loss(model(torch.randn(4, 4)), -torch.ones(4, 2)))
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
