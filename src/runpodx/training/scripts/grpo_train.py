#!/usr/bin/env python3
"""Generic GRPO training script with pluggable task profiles."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from trl import GRPOConfig, GRPOTrainer
except ImportError:
    GRPOConfig = None
    GRPOTrainer = None

try:
    from sft_shared import resolve_wandb_run_id
except ImportError:
    from runpodx.training.scripts.sft_shared import resolve_wandb_run_id


BUILTIN_GRPO_PROFILES = {
    "ascii_art": "grpo_profile_ascii_art",
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_dataset_split(config: dict) -> tuple[Dataset, Dataset | None]:
    local_path = config.get("dataset_path")
    hf_dataset = config.get("hf_dataset")

    if local_path and os.path.exists(local_path):
        ds = load_dataset("json", data_files=local_path, split="train")
    elif hf_dataset:
        split = config.get("hf_split", "train")
        ds = load_dataset(hf_dataset, split=split)
    else:
        print("[runpodx] ERROR: No dataset source found in config.", file=sys.stderr)
        sys.exit(1)

    max_samples = config.get("max_samples")
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    validation_ratio = config.get("validation_ratio")
    if validation_ratio:
        split = ds.train_test_split(
            test_size=validation_ratio,
            seed=config.get("validation_seed", 42),
            shuffle=True,
        )
        return split["train"], split["test"]

    return ds, None


def build_training_args(config_cls, common_args: dict, eval_enabled: bool):
    base_args = dict(common_args)
    if eval_enabled:
        base_args["eval_strategy"] = "epoch"

    candidates = [base_args]
    if eval_enabled:
        legacy_args = dict(base_args)
        legacy_args["evaluation_strategy"] = legacy_args.pop("eval_strategy")
        candidates.append(legacy_args)

    last_exc = None
    seen: set[tuple[str, ...]] = set()
    queue = list(candidates)
    while queue:
        candidate = queue.pop(0)
        key = tuple(sorted(candidate.keys()))
        if key in seen:
            continue
        seen.add(key)
        try:
            return config_cls(**candidate)
        except TypeError as exc:
            last_exc = exc

    raise last_exc


def patch_rope_compat() -> None:
    """Patch transformers rope helpers for older custom remote-code models."""
    try:
        from transformers import modeling_rope_utils as rope_utils
    except ImportError:
        return

    rope_init_functions = getattr(rope_utils, "ROPE_INIT_FUNCTIONS", None)
    if not isinstance(rope_init_functions, dict) or "default" in rope_init_functions:
        return

    default_fn = getattr(rope_utils, "_compute_default_rope_parameters", None)
    if default_fn is not None:
        rope_init_functions["default"] = default_fn
        return

    if rope_init_functions:
        rope_init_functions["default"] = next(iter(rope_init_functions.values()))


def load_grpo_profile(profile_name: str):
    candidates = []
    if profile_name in BUILTIN_GRPO_PROFILES:
        candidates.append(BUILTIN_GRPO_PROFILES[profile_name])
    candidates.append(profile_name)
    if "." not in profile_name:
        candidates.append(f"runpodx.training.scripts.{profile_name}")

    seen: set[str] = set()
    errors: list[str] = []
    for module_name in candidates:
        if module_name in seen:
            continue
        seen.add(module_name)
        try:
            module = importlib.import_module(module_name)
            break
        except ModuleNotFoundError as exc:
            errors.append(f"{module_name}: {exc}")
    else:
        joined = "; ".join(errors)
        raise RuntimeError(
            f"Unable to import GRPO profile {profile_name!r}. Tried: {joined}"
        )

    required_attrs = ("prepare_datasets", "build_reward_functions")
    missing = [attr for attr in required_attrs if not callable(getattr(module, attr, None))]
    if missing:
        raise RuntimeError(
            f"GRPO profile {profile_name!r} is missing required callables: {missing}"
        )
    return module


def reward_function_name(reward_func) -> str:
    name = getattr(reward_func, "__name__", None)
    if name:
        return name
    wrapped = getattr(reward_func, "func", None)
    return getattr(wrapped, "__name__", reward_func.__class__.__name__)


def main():
    if GRPOTrainer is None or GRPOConfig is None:
        print("[runpodx] ERROR: TRL with GRPO support is required.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[runpodx] Config loaded: {json.dumps(config, indent=2)}")
    profile_name = config.get("grpo_profile")
    if not profile_name:
        print("[runpodx] ERROR: grpo_profile is required for GRPO runs.", file=sys.stderr)
        sys.exit(1)

    profile = load_grpo_profile(profile_name)
    print(f"[runpodx] Loaded GRPO profile: {profile_name}")

    wandb_project = config.get("wandb_project")
    if wandb_project:
        import wandb

        os.environ["WANDB_PROJECT"] = wandb_project
        if config.get("wandb_entity"):
            os.environ["WANDB_ENTITY"] = config["wandb_entity"]
        run_id, resume_mode = resolve_wandb_run_id(
            config.get("output_dir", "/workspace/output"),
            resume_from_checkpoint=config.get("resume_from_checkpoint"),
            generate_id=wandb.util.generate_id,
        )
        os.environ["WANDB_RUN_ID"] = run_id
        if resume_mode:
            os.environ["WANDB_RESUME"] = resume_mode
            print(f"[runpodx] Resuming W&B run id: {run_id}")
        else:
            os.environ.pop("WANDB_RESUME", None)
            print(f"[runpodx] Using W&B run id: {run_id}")
        report_to = "wandb"
        run_name = config.get("wandb_run_name") or None
    else:
        report_to = "none"
        run_name = None

    model_name = config["model_name"]
    print(f"[runpodx] Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    patch_rope_compat()

    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if getattr(model_config, "pad_token_id", None) is None:
        model_config.pad_token_id = tokenizer.pad_token_id
    if getattr(model_config, "eos_token_id", None) is None:
        model_config.eos_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_config,
        torch_dtype=torch.bfloat16 if config.get("bf16", True) else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if config.get("adapter") == "lora":
        adapter_path = config.get("adapter_path")
        if adapter_path:
            print(f"[runpodx] Loading existing LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
            model.print_trainable_parameters()
        else:
            print("[runpodx] Applying fresh LoRA adapter")
            lora_config = LoraConfig(
                r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                lora_dropout=config.get("lora_dropout", 0.05),
                target_modules=config.get("lora_target_modules"),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

    print("[runpodx] Loading dataset...")
    train_dataset, eval_dataset = load_dataset_split(config)
    print(f"[runpodx] Train dataset size: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"[runpodx] Validation dataset size: {len(eval_dataset)}")

    train_dataset, eval_dataset = profile.prepare_datasets(
        train_dataset,
        eval_dataset,
        dataset_format=config.get("dataset_format", "alpaca"),
        tokenizer=tokenizer,
        config=config,
    )
    print("[runpodx] Dataset preparation complete for GRPO profile")

    reward_funcs = profile.build_reward_functions(config)
    reward_names = [reward_function_name(func) for func in reward_funcs]
    if not reward_funcs:
        raise ValueError(f"GRPO profile {profile_name!r} did not enable any reward functions.")
    print(f"[runpodx] Enabled GRPO rewards: {', '.join(reward_names)}")
    reward_weights = config.get("grpo_reward_weights")
    if reward_weights is not None and len(reward_weights) != len(reward_funcs):
        raise ValueError(
            "grpo_reward_weights length must match reward function count "
            f"({len(reward_funcs)}: {', '.join(reward_names)})."
        )
    common_args = dict(
        output_dir=config.get("output_dir", "/workspace/output"),
        num_train_epochs=config.get("num_train_epochs", 1),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 1e-6),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        weight_decay=config.get("weight_decay", 0.0),
        bf16=config.get("bf16", True),
        fp16=config.get("fp16", False),
        optim=config.get("optimizer", "adamw_8bit"),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 100),
        save_total_limit=config.get("save_total_limit", 3),
        report_to=report_to,
        run_name=run_name,
        push_to_hub=config.get("push_to_hub", False),
        hub_model_id=config.get("hub_model_id") or None,
        hub_token=config.get("hub_token") or None,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_completion_length=config.get("grpo_max_completion_length", 192),
        num_generations=config.get("grpo_num_generations", 4),
        beta=config.get("grpo_beta", 0.0),
        temperature=config.get("grpo_temperature", 1.0),
        top_p=config.get("grpo_top_p", 1.0),
        reward_weights=reward_weights,
    )
    training_args = build_training_args(
        GRPOConfig,
        common_args,
        eval_enabled=eval_dataset is not None,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("[runpodx] Starting GRPO training...")
    resume_from = config.get("resume_from_checkpoint")
    if resume_from == "latest":
        output_dir = Path(config.get("output_dir", "/workspace/output"))
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=os.path.getmtime)
        if checkpoints:
            resume_from = str(checkpoints[-1])
            print(f"[runpodx] Resuming from latest checkpoint: {resume_from}")
        else:
            print("[runpodx] No checkpoints found, starting from scratch.")
            resume_from = None
    elif resume_from:
        print(f"[runpodx] Resuming from checkpoint: {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)

    benchmark_runner = getattr(profile, "run_generation_benchmark", None)
    if callable(benchmark_runner):
        benchmark_runner(trainer.model, tokenizer, config, report_to)

    print("[runpodx] Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    print("[runpodx] Done.")


if __name__ == "__main__":
    main()
