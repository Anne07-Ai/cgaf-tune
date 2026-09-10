"""Dependency-free SVG chart generation for research artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from pathlib import Path


def write_pareto_svg(
    summary: Mapping[str, Mapping[str, Mapping[str, float]]], path: str | Path
) -> Path:
    """Write domain gain versus forgetting as an accessible SVG scatter chart."""
    points = [(method, values["mean_forgetting"]["mean"], values["domain_gain"]["mean"])
              for method, values in summary.items()]
    if not points:
        raise ValueError("summary has no methods")
    width, height, left, right, top, bottom = 800, 520, 90, 40, 60, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    xs, ys = [point[1] for point in points], [point[2] for point in points]
    x_min, x_max = _bounds(xs)
    y_min, y_max = _bounds(ys)

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="400" y="30" text-anchor="middle" font-family="sans-serif" font-size="21">Adaptation–retention Pareto comparison</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="{left + plot_w / 2}" y="500" text-anchor="middle" font-family="sans-serif">Mean forgetting (lower is better)</text>',
        f'<text x="20" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 20 {top + plot_h / 2})" font-family="sans-serif">Domain gain (higher is better)</text>',
    ]
    for index in range(5):
        xv = x_min + index * (x_max - x_min) / 4
        yv = y_min + index * (y_max - y_min) / 4
        lines.extend([
            f'<text x="{x(xv):.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{xv:.3f}</text>',
            f'<text x="{left - 12}" y="{y(yv) + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{yv:.3f}</text>',
        ])
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2"]
    for index, (method, forgetting, gain) in enumerate(sorted(points)):
        px, py, color = x(forgetting), y(gain), colors[index % len(colors)]
        lines.extend([
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="{color}"/>',
            f'<text x="{px + 10:.1f}" y="{py - 10:.1f}" font-family="sans-serif" font-size="13">{escape(method)}</text>',
        ])
    lines.append("</svg>")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def _bounds(values: list[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    if low == high:
        padding = max(0.01, abs(low) * 0.1)
    else:
        padding = (high - low) * 0.15
    return low - padding, high + padding
