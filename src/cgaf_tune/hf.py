"""Hugging Face and PEFT construction for the single-GPU QLoRA pilot."""

from __future__ import annotations

from typing import Any

import torch


def resolve_dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as error:
        raise ValueError(f"unsupported torch dtype: {name}") from error
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"unsupported torch dtype: {name}")
    return dtype


def build_model_and_tokenizer(config: dict[str, Any]):
    """Load a causal LM, prepare quantized training, and attach LoRA adapters."""
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError("install the project training dependencies before a real run") from error

    model_config = config["model"]
    lora_config = config["lora"]
    dtype = resolve_dtype(model_config.get("torch_dtype", "bfloat16"))
    quantization = None
    if model_config.get("load_in_4bit", True):
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit QLoRA requires a CUDA GPU")
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_config["name"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        torch_dtype=dtype,
        quantization_config=quantization,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if quantization is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config["training"].get("gradient_checkpointing", True)
        )
    adapter = LoraConfig(
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["targets"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, adapter)
    model.config.use_cache = False
    return model, tokenizer
