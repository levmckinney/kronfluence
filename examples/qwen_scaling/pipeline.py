from typing import List, Optional

import torch
from datasets import load_dataset
from torch import nn
from torch.utils import data
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from kronfluence.utils.model import apply_fsdp

# Supported Qwen 2.5 models
QWEN_MODELS = {
    "0.5B": "Qwen/Qwen2.5-0.5B",
    "1.5B": "Qwen/Qwen2.5-1.5B",
    "3B": "Qwen/Qwen2.5-3B",
    "7B": "Qwen/Qwen2.5-7B",
    "14B": "Qwen/Qwen2.5-14B",
    "32B": "Qwen/Qwen2.5-32B",
}

MAX_LENGTH = 512


def get_model_name(model_size: str) -> str:
    if model_size not in QWEN_MODELS:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(QWEN_MODELS.keys())}")
    return QWEN_MODELS[model_size]


def construct_model(model_size: str) -> nn.Module:
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = get_model_name(model_size)
    config = AutoConfig.from_pretrained(
        model_name,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map=device_map,
        dtype=torch.bfloat16,
        ignore_mismatched_sizes=False,
    )
    return model


def apply_fsdp_to_model(model: nn.Module) -> nn.Module:
    return apply_fsdp(model, sequential=model.model.layers, cpu_offload=False)


def get_openwebtext_dataset(
    model_size: str,
    num_samples: Optional[int] = None,
    indices: Optional[List[int]] = None,
) -> data.Dataset:
    model_name = get_model_name(model_size)
    raw_datasets = load_dataset("Elriggs/openwebtext-100k")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        results = tokenizer(examples[text_column_name], truncation=True, padding=True, max_length=MAX_LENGTH)
        results["labels"] = results["input_ids"].copy()
        results["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in label] for label in results["labels"]
        ]
        return results

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    ds = tokenized_datasets["train"]

    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    if indices is not None:
        ds = ds.select(indices)

    return ds


def get_eval_dataset(
    model_size: str,
    num_samples: Optional[int] = None,
    indices: Optional[List[int]] = None,
) -> data.Dataset:
    model_name = get_model_name(model_size)
    raw_datasets = load_dataset("Elriggs/openwebtext-100k")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        results = tokenizer(examples[text_column_name], truncation=True, padding=True, max_length=MAX_LENGTH)
        results["labels"] = results["input_ids"].copy()
        results["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in label] for label in results["labels"]
        ]
        return results

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    # Use a different slice for eval (from the end of the dataset)
    ds = tokenized_datasets["train"]
    total_len = len(ds)

    if num_samples is not None:
        # Take samples from the end to avoid overlap with training
        start_idx = max(0, total_len - num_samples)
        ds = ds.select(range(start_idx, total_len))

    if indices is not None:
        ds = ds.select(indices)

    return ds


if __name__ == "__main__":
    from kronfluence import Analyzer

    for size in ["0.5B", "1.5B"]:
        print(f"\n=== {size} Model ===")
        model = construct_model(size)
        print(f"Num layers: {model.config.num_hidden_layers}")
        print(Analyzer.get_module_summary(model))
