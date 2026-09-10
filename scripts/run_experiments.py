"""Run the matched CGAF method-by-seed experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from cgaf_tune.experiments import SUPPORTED_METHODS, build_matrix, run_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--domain-data", type=Path, required=True)
    parser.add_argument("--anchor-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(SUPPORTED_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.dry_run:
        print(json.dumps({"runs": build_matrix(args.methods, args.seeds)}, indent=2))
        return
    manifest = run_matrix(
        config,
        args.domain_data,
        args.anchor_data,
        args.output_root,
        args.methods,
        args.seeds,
        args.max_steps,
        args.fail_fast,
    )
    print(f"experiment manifest: {manifest}")


if __name__ == "__main__":
    main()
