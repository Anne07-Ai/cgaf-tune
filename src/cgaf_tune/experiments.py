"""Reproducible method-by-seed experiment orchestration."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .runner import run_training

SUPPORTED_METHODS = ("lora", "rehearsal", "hard_projection", "cgaf")


def build_matrix(methods: Sequence[str], seeds: Sequence[int]) -> list[dict[str, object]]:
    """Create deterministic method-major experiment specifications."""
    unknown = set(methods) - set(SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"unsupported methods: {sorted(unknown)}")
    if not methods or not seeds:
        raise ValueError("methods and seeds must be non-empty")
    if len(set(methods)) != len(methods) or len(set(seeds)) != len(seeds):
        raise ValueError("methods and seeds must not contain duplicates")
    return [
        {"method": method, "seed": int(seed), "run_id": f"{method}-seed{seed}"}
        for method in methods
        for seed in seeds
    ]


def run_matrix(
    base_config: dict,
    domain_path: str | Path,
    anchor_path: str | Path,
    output_root: str | Path,
    methods: Sequence[str],
    seeds: Sequence[int],
    max_steps: int | None = None,
    fail_fast: bool = False,
    train_fn: Callable = run_training,
) -> Path:
    """Execute the matrix and continuously checkpoint a machine-readable manifest."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "experiment-manifest.json"
    manifest = {"runs": build_matrix(methods, seeds)}
    _write_manifest(manifest_path, manifest)
    for run in manifest["runs"]:
        config = copy.deepcopy(base_config)
        method, seed = str(run["method"]), int(run["seed"])
        config["experiment"]["seed"] = seed
        config["experiment"]["name"] = (
            f"{base_config['experiment']['name']}-{method}-seed{seed}"
        )
        config["training"]["method"] = method
        run["status"] = "running"
        run["started_at_unix"] = time.time()
        _write_manifest(manifest_path, manifest)
        try:
            destination = train_fn(
                config, domain_path, anchor_path, root / str(run["run_id"]), max_steps
            )
            run.update({"status": "completed", "output_dir": str(destination)})
        except Exception as error:
            run.update(
                {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
            )
            if fail_fast:
                _write_manifest(manifest_path, manifest)
                raise
        finally:
            run["finished_at_unix"] = time.time()
            _write_manifest(manifest_path, manifest)
    return manifest_path


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)
