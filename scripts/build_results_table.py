"""Combine model-evaluation JSON files into a publication-ready Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cgaf_tune.reporting import markdown_results_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", action="append", required=True, metavar="LABEL=PATH",
        help="repeat for each model or method",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for specification in args.result:
        if "=" not in specification:
            parser.error("--result must use LABEL=PATH")
        label, path = specification.split("=", 1)
        results[label] = json.loads(Path(path).read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown_results_table(results), encoding="utf-8")
    print(f"results table: {args.output}")


if __name__ == "__main__":
    main()
