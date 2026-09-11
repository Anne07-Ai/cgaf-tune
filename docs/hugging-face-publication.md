# Hugging Face publication procedure

Publishing requires a Hugging Face token or OAuth connection with model-repository write access.
Do not paste tokens into chat or command arguments.

1. Reconnect Hugging Face with write-repository permission.
2. Rerun CGAF training with validation and best-checkpoint restoration.
3. In the same persistent job, publish the adapter before the job exits.
4. Use an explicit repository visibility.
5. Verify the Hub files, model card, base-model link, license, and result summary.
6. Only link a Hugging Face Paper Page after an arXiv identifier exists.

Local publication command:

```bash
export HF_TOKEN=...  # set securely outside shell history where possible
python scripts/publish_adapter.py \
  --adapter outputs/cgaf-seed42/adapter \
  --repo-id lakshmianne/cgaf-qwen3-0.6b-seed42 \
  --visibility private \
  --model-card MODEL_CARD.md \
  --results results/phase6-three-seed-summary.json
```

For Hugging Face Jobs, pass `HF_TOKEN` through the job's secrets facility and call the same script
before the ephemeral job terminates. Start private for verification; change visibility only after
the model card and artifacts have been checked.
