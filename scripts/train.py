#!/usr/bin/env python
"""Validated entry point; full Trainer integration is tracked in the roadmap."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

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
    args = parser.parse_args()
    config = load_config(args.config)

    if args.dry_run:
        print(f"configuration valid: {config['experiment']['name']}")
        print(f"model: {config['model']['name']}")
        print(f"method: {config['training']['method']}")
        return

    raise NotImplementedError(
        "End-to-end Trainer integration is the next milestone. "
        "Use --dry-run to validate the experiment configuration."
    )


if __name__ == "__main__":
    main()
