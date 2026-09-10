"""Publication-ready Markdown result tables."""

from __future__ import annotations

from collections.abc import Mapping


def markdown_results_table(results: Mapping[str, Mapping[str, object]]) -> str:
    """Render overall exact match and token F1 with example-bootstrap intervals."""
    if not results:
        raise ValueError("results cannot be empty")
    lines = [
        "| Model / method | Exact match (95% CI) | Token F1 (95% CI) | Examples |",
        "|---|---:|---:|---:|",
    ]
    for label, result in results.items():
        overall = result["overall"]
        exact, f1 = overall["exact_match"], overall["token_f1"]
        lines.append(
            f"| {label} | {_interval(exact)} | {_interval(f1)} | {int(exact['examples'])} |"
        )
    return "\n".join(lines) + "\n"


def _interval(metric: Mapping[str, float]) -> str:
    return f"{metric['mean']:.3f} [{metric['ci_low']:.3f}, {metric['ci_high']:.3f}]"
