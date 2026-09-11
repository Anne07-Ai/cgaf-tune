"""Submit the persistent CGAF publication job from GitHub Actions."""

import argparse
import os
from pathlib import Path

from huggingface_hub import run_uv_job

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", required=True)
parser.add_argument("--visibility", choices=["private", "public"], required=True)
args = parser.parse_args()
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("GitHub Actions secret HF_TOKEN is missing")

job = run_uv_job(
    "jobs/publish_cgaf.py",
    script_args=[
        "--repo-id", args.repo_id,
        "--visibility", args.visibility,
    ],
    flavor="t4-small",
    timeout="2h",
    secrets={"HF_TOKEN": token},
)
print(f"Submitted Hugging Face publication job: {job.url}")
