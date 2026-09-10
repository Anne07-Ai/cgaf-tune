"""Compute CGAF adaptation-retention metrics from a JSON result file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cgaf_tune.charts import write_pareto_svg
from cgaf_tune.evaluation import (
    compute_tradeoff,
    paired_differences,
    pareto_methods,
    summarize_methods,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chart", type=Path, help="optional Pareto SVG output")
    parser.add_argument("--baseline", default="lora")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    records = []
    for run in payload["runs"]:
        metrics = compute_tradeoff(
            run["domain"]["before"],
            run["domain"]["after"],
            run["retention"]["before"],
            run["retention"]["after"],
            payload.get("score_ceiling", 1.0),
        ).to_dict()
        records.append({"method": run["method"], "seed": run["seed"], "metrics": metrics})
    summary = summarize_methods(records, args.bootstrap_samples)
    result = {
        "runs": records,
        "summary": summary,
        "paired_vs_baseline": paired_differences(
            records, args.baseline, args.bootstrap_samples
        ),
        "pareto_methods": pareto_methods(summary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.chart:
        write_pareto_svg(summary, args.chart)
    print(f"evaluation report: {args.output}")


if __name__ == "__main__":
    main()
