# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch>=2.2", "transformers>=4.51,<5", "peft>=0.15",
#   "datasets>=3.0", "accelerate>=1.2", "bitsandbytes>=0.45",
#   "pyyaml>=6.0", "huggingface_hub>=0.26"
# ]
# ///
"""Train, evaluate, and persist the validation-selected CGAF reference adapter."""

import argparse
import gc
import json
import os
import pathlib
import subprocess
import sys
import yaml
import torch
from datasets import load_dataset
from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", required=True)
parser.add_argument("--visibility", choices=["private", "public"], required=True)
args = parser.parse_args()
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN secret is required")

root = pathlib.Path("/tmp/cgaf-tune")
subprocess.run(
    ["git", "clone", "--depth", "1", "https://github.com/Anne07-Ai/cgaf-tune.git", str(root)],
    check=True,
)
sys.path.insert(0, str(root / "src"))
from cgaf_tune.eval_data import token_f1
from cgaf_tune.hf import load_evaluation_model

config = yaml.safe_load((root / "configs/phase6_validation.yaml").read_text())
config["experiment"]["seed"] = 42
config["training"]["method"] = "cgaf"
config["training"]["epochs"] = 3
config["evaluation"].update(
    {
        "validation_interval": 100,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.002,
    }
)
config_path = root / "configs/publish-cgaf-seed42.yaml"
config_path.write_text(yaml.safe_dump(config, sort_keys=False))

dataset = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=2026)
def convert(row):
    prompt = row["instruction"].strip()
    context = (row.get("context") or "").strip()
    if context:
        prompt += "\n\nContext:\n" + context
    return {"prompt": prompt, "response": row["response"].strip()}

domain = [convert(x) for x in dataset if x["category"] == "summarization"]
anchor = [
    convert(x) for x in dataset
    if x["category"] in {"closed_qa", "general_qa"}
]
data = root / "data"
data.mkdir()
for name, rows in [
    ("domain.jsonl", domain[:512]),
    ("anchor.jsonl", anchor[:512]),
    ("validation.jsonl", domain[512:576] + anchor[512:576]),
]:
    (data / name).write_text("\n".join(json.dumps(x) for x in rows) + "\n")

output = root / "outputs/cgaf-seed42-release"
env = os.environ.copy()
env["PYTHONPATH"] = str(root / "src")
command = [
    "python", str(root / "scripts/train.py"),
    "--config", str(config_path),
    "--domain-data", str(data / "domain.jsonl"),
    "--anchor-data", str(data / "anchor.jsonl"),
    "--validation-data", str(data / "validation.jsonl"),
    "--method", "cgaf",
    "--output-dir", str(output),
    "--max-steps", "1200",
]
subprocess.run(command, cwd=root, env=env, check=True)

test = [
    dict(x, category="domain") for x in domain[576:600]
] + [
    dict(x, category="anchor") for x in anchor[576:600]
]
model, tokenizer = load_evaluation_model(config, str(output / "adapter"))
model.eval()
tokenizer.padding_side = "left"
predictions = []
with torch.inference_mode():
    for start in range(0, len(test), 4):
        batch = test[start:start + 4]
        prompts = [f"User: {x['prompt']}\nAssistant:" for x in batch]
        encoded = tokenizer(
            prompts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        predictions.extend(
            text.strip()
            for text in tokenizer.batch_decode(
                generated[:, encoded["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
        )
scores = {}
for category in ["domain", "anchor"]:
    values = [
        token_f1(prediction, example["response"])
        for example, prediction in zip(test, predictions, strict=True)
        if example["category"] == category
    ]
    scores[category + "_f1"] = sum(values) / len(values)
domain_f1, anchor_f1 = scores["domain_f1"], scores["anchor_f1"]
scores["harmonic_f1"] = 2 * domain_f1 * anchor_f1 / (domain_f1 + anchor_f1 + 1e-12)
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()

summary = {
    "release": "cgaf-qwen3-0.6b-seed42",
    "base_model": config["model"]["name"],
    "seed": 42,
    "scores": scores,
    "training": json.loads((output / "training_summary.json").read_text()),
    "source": "https://github.com/Anne07-Ai/cgaf-tune",
}
results_path = output / "release-results.json"
results_path.write_text(json.dumps(summary, indent=2) + "\n")
card = (root / "MODEL_CARD.md").read_text()
card += (
    "\n## Published checkpoint\n\n"
    "This repository contains the validation-selected seed-42 CGAF adapter. "
    "Release evaluation: "
    f"domain F1 {domain_f1:.4f}, anchor F1 {anchor_f1:.4f}, "
    f"harmonic F1 {scores['harmonic_f1']:.4f}.\n"
)
(output / "README.md").write_text(card)

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
    folder_path=output / "adapter",
    path_in_repo=".",
    commit_message="Upload validation-selected CGAF adapter",
)
api.upload_file(
    repo_id=args.repo_id,
    repo_type="model",
    path_or_fileobj=output / "README.md",
    path_in_repo="README.md",
    commit_message="Add CGAF model card",
)
api.upload_file(
    repo_id=args.repo_id,
    repo_type="model",
    path_or_fileobj=results_path,
    path_in_repo="results/release-results.json",
    commit_message="Add release evaluation",
)
print("CGAF_PUBLISHED=" + json.dumps({"url": f"https://huggingface.co/{args.repo_id}", **summary}))
