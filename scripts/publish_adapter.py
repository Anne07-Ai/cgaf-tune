"""Publish a trained adapter and research metadata to Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--visibility", choices=["public", "private"], required=True)
    parser.add_argument("--model-card", type=Path, default=Path("MODEL_CARD.md"))
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required; pass it as a secret, never as a CLI argument")
    if not args.adapter.is_dir():
        raise SystemExit(f"adapter directory not found: {args.adapter}")
    if not args.model_card.is_file():
        raise SystemExit(f"model card not found: {args.model_card}")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.visibility == "private",
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=args.adapter,
        path_in_repo=".",
        commit_message="Upload validation-selected CGAF adapter",
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=args.model_card,
        path_in_repo="README.md",
        commit_message="Add CGAF research model card",
    )
    if args.results:
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=args.results,
            path_in_repo="results/phase6-three-seed-summary.json",
            commit_message="Add matched three-seed evaluation summary",
        )
    print(f"published: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
