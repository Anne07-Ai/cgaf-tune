# ruff: noqa\n"""Analyse Phase 6 generation runs and emit reproducible Phase 7 artifacts."""

from __future__ import annotations

import argparse
import html
import json
import random
import statistics
from pathlib import Path

METHODS = ["lora", "rehearsal", "hard_projection", "cgaf"]
COLORS = {"lora": "#3b82f6", "rehearsal": "#a855f7", "hard_projection": "#f59e0b", "cgaf": "#ec4899"}


def bootstrap_ci(values: list[float], samples: int = 20000, seed: int = 2026) -> list[float]:
    rng = random.Random(seed)
    draws = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples))
    return [draws[int(samples * 0.025)], draws[int(samples * 0.975) - 1]]


def analyse(payload: dict) -> dict:
    runs = payload["runs"]
    summary, paired = {}, {}
    for method in METHODS:
        rows = [r for r in runs if r["method"] == method]
        summary[method] = {}
        for metric in ["domain_f1", "anchor_f1", "harmonic_f1"]:
            values = [r[metric] for r in rows]
            summary[method][metric] = {"mean": statistics.fmean(values), "sd": statistics.stdev(values), "seed_bootstrap_95_ci": bootstrap_ci(values)}
    baseline = {r["seed"]: r for r in runs if r["method"] == "lora"}
    for method in METHODS[1:]:
        current = {r["seed"]: r for r in runs if r["method"] == method}
        paired[method] = {}
        for metric in ["domain_f1", "anchor_f1", "harmonic_f1"]:
            differences = [current[s][metric] - baseline[s][metric] for s in sorted(baseline)]
            paired[method][metric] = {"mean_difference": statistics.fmean(differences), "differences": differences, "seed_bootstrap_95_ci": bootstrap_ci(differences)}
    points = {m: (summary[m]["domain_f1"]["mean"], summary[m]["anchor_f1"]["mean"]) for m in METHODS}
    pareto = [m for m,(d,a) in points.items() if not any((od >= d and oa >= a and (od > d or oa > a)) for o,(od,oa) in points.items() if o != m)]
    return {"summary": summary, "paired_vs_lora": paired, "pareto_methods": pareto, "limitations": ["Only three seeds; seed-bootstrap intervals are descriptive and discrete.", "Token F1 on 24 examples per objective does not capture human preference or factuality."]}


def tradeoff_svg(result: dict, path: Path) -> None:
    width,height=820,560; left,top,right,bottom=90,55,35,80
    xmin,xmax,ymin,ymax=.205,.220,.265,.287
    def x(v): return left+(v-xmin)/(xmax-xmin)*(width-left-right)
    def y(v): return height-bottom-(v-ymin)/(ymax-ymin)*(height-top-bottom)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="#0f172a"/>','<text x="410" y="30" text-anchor="middle" fill="#f8fafc" font-size="22" font-family="sans-serif">Adaptation–Retention Trade-off (3 Seeds)</text>']
    for i in range(6):
        xv=xmin+(xmax-xmin)*i/5; px=x(xv); parts += [f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{height-bottom}" stroke="#334155"/>',f'<text x="{px:.1f}" y="{height-bottom+25}" text-anchor="middle" fill="#cbd5e1" font-size="12" font-family="sans-serif">{xv:.3f}</text>']
    for i in range(6):
        yv=ymin+(ymax-ymin)*i/5; py=y(yv); parts += [f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="#334155"/>',f'<text x="{left-12}" y="{py+4:.1f}" text-anchor="end" fill="#cbd5e1" font-size="12" font-family="sans-serif">{yv:.3f}</text>']
    for m in METHODS:
        s=result["summary"][m]; px=x(s["anchor_f1"]["mean"]); py=y(s["domain_f1"]["mean"]); label=m.replace("_"," ").title()
        parts += [f'<circle cx="{px:.1f}" cy="{py:.1f}" r="9" fill="{COLORS[m]}" stroke="#fff" stroke-width="2"/>',f'<text x="{px+13:.1f}" y="{py-11:.1f}" fill="#f8fafc" font-size="14" font-family="sans-serif">{html.escape(label)}</text>']
    parts += [f'<text x="{(left+width-right)/2}" y="535" text-anchor="middle" fill="#e2e8f0" font-size="15" font-family="sans-serif">Retention / Anchor F1 →</text>',f'<text transform="translate(24 {(top+height-bottom)/2}) rotate(-90)" text-anchor="middle" fill="#e2e8f0" font-size="15" font-family="sans-serif">Domain F1 →</text>','</svg>']
    path.write_text("".join(parts),encoding="utf-8")


def seed_svg(payload: dict, path: Path) -> None:
    width,height=820,520; left,top,right,bottom=80,55,35,75; seeds=[42,123,456]
    vals=[r["harmonic_f1"] for r in payload["runs"]]; ymin,ymax=min(vals)-.006,max(vals)+.006
    def x(i): return left+i*(width-left-right)/2
    def y(v): return height-bottom-(v-ymin)/(ymax-ymin)*(height-top-bottom)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="#0f172a"/>','<text x="410" y="30" text-anchor="middle" fill="#f8fafc" font-size="22" font-family="sans-serif">Harmonic F1 by Training Seed</text>']
    for i,s in enumerate(seeds): parts += [f'<line x1="{x(i):.1f}" y1="{top}" x2="{x(i):.1f}" y2="{height-bottom}" stroke="#334155"/>',f'<text x="{x(i):.1f}" y="{height-bottom+27}" text-anchor="middle" fill="#cbd5e1" font-size="13" font-family="sans-serif">Seed {s}</text>']
    for m in METHODS:
        rows=sorted([r for r in payload["runs"] if r["method"]==m],key=lambda r:r["seed"]); pts=" ".join(f'{x(i):.1f},{y(r["harmonic_f1"]):.1f}' for i,r in enumerate(rows)); label=m.replace("_"," ").title()
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{COLORS[m]}" stroke-width="3"/>')
        for i,r in enumerate(rows): parts.append(f'<circle cx="{x(i):.1f}" cy="{y(r["harmonic_f1"]):.1f}" r="6" fill="{COLORS[m]}"/>')
        parts.append(f'<text x="{width-right-5}" y="{y(rows[-1]["harmonic_f1"])-8:.1f}" text-anchor="end" fill="{COLORS[m]}" font-size="13" font-family="sans-serif">{html.escape(label)}</text>')
    parts += [f'<text transform="translate(23 {(top+height-bottom)/2}) rotate(-90)" text-anchor="middle" fill="#e2e8f0" font-size="15" font-family="sans-serif">Harmonic F1</text>','</svg>']
    path.write_text("".join(parts),encoding="utf-8")


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
    payload=json.loads(a.input.read_text());a.output_dir.mkdir(parents=True,exist_ok=True);result=analyse(payload)
    (a.output_dir/"phase7-statistics.json").write_text(json.dumps(result,indent=2)+"\n")
    tradeoff_svg(result,a.output_dir/"phase7-tradeoff.svg");seed_svg(payload,a.output_dir/"phase7-seed-sensitivity.svg")

if __name__ == "__main__": main()
