"""Validated entry point; full Trainer integration is tracked in the roadmap."""

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
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    missing = REQUIRED.difference(config)
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke-test", action="store_true", help="exercise a real CGAF optimizer step"
    )
    parser.add_argument("--domain-data", type=Path, help="domain JSONL training data")
    parser.add_argument("--anchor-data", type=Path, help="capability-retention JSONL data")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cgaf-pilot"))
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)

    if args.dry_run:
        print(f"configuration valid: {config['experiment']['name']}")
        print(f"model: {config['model']['name']}")
        print(f"method: {config['training']['method']}")
        return

    if args.smoke_test:
        run_smoke_test(config)
        return

    if args.domain_data is None or args.anchor_data is None:
        parser.error("real training requires --domain-data and --anchor-data")
    destination = run_training(
        config, args.domain_data, args.anchor_data, args.output_dir, args.max_steps
    )
    print(f"training complete: {destination}")


def run_smoke_test(config: dict) -> None:
    """Run the production gradient path on a deterministic tiny network."""
    torch.manual_seed(config["experiment"]["seed"])
    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    engine = CGAFStepEngine(
        model,
        optimizer,
        CGAFConfig(
            temperature=config["cgaf"]["temperature"],
            epsilon=config["cgaf"]["epsilon"],
            minimum_conflict=config["cgaf"]["minimum_conflict"],
        ),
        grouping="global",
    )
    domain_inputs = torch.randn(4, 4)
    anchor_inputs = torch.randn(4, 4)
    domain_targets = torch.ones(4, 2)
    anchor_targets = -torch.ones(4, 2)
    result = engine.step(
        lambda: torch.nn.functional.mse_loss(model(domain_inputs), domain_targets),
        lambda: torch.nn.functional.mse_loss(model(anchor_inputs), anchor_targets),
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
