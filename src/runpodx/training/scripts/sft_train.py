#!/usr/bin/env python3
"""SFT training script for runpodx.

Uploaded to the RunPod pod and executed remotely.
Reads a JSON config and runs supervised fine-tuning using
transformers + peft + trl.

Usage:
    python sft_train.py --config /workspace/runpodx_run/config.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_sft_dataset(config: dict) -> tuple[Dataset, Dataset | None]:
    """Load dataset and optionally split off a validation set."""
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
    """Build TrainingArguments/SFTConfig with version-compatible eval kwargs."""
    eval_key = "eval_strategy"
    if eval_enabled:
        common_args[eval_key] = "epoch"

    try:
        return config_cls(**common_args)
    except TypeError as exc:
        if eval_enabled and "eval_strategy" in common_args:
            legacy_args = dict(common_args)
            legacy_args["evaluation_strategy"] = legacy_args.pop("eval_strategy")
            return config_cls(**legacy_args)
        raise exc


def format_alpaca(example: dict) -> str:
    """Format an alpaca-style example (instruction/input/output)."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")

    if inp:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def format_sharegpt(example: dict) -> str:
    """Format a sharegpt/messages-style example into a prompt string."""
    messages = example.get("messages") or example.get("conversations") or []
    parts = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        parts.append(f"### {role}:\n{content}")
    return "\n\n".join(parts)


FORMATTERS = {
    "alpaca": format_alpaca,
    "sharegpt": format_sharegpt,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[runpodx] Config loaded: {json.dumps(config, indent=2)}")

    # --- W&B setup ---
    wandb_project = config.get("wandb_project")
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        if config.get("wandb_entity"):
            os.environ["WANDB_ENTITY"] = config["wandb_entity"]
        report_to = "wandb"
        run_name = config.get("wandb_run_name") or None
    else:
        report_to = "none"
        run_name = None

    # --- Load model + tokenizer ---
    model_name = config["model_name"]
    print(f"[runpodx] Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if config.get("bf16", True) else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # --- LoRA ---
    if config.get("adapter") == "lora":
        adapter_path = config.get("adapter_path")
        if adapter_path:
            # Load existing LoRA adapter weights (continue fine-tuning)
            print(f"[runpodx] Loading existing LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
            model.print_trainable_parameters()
        else:
            # Create fresh LoRA adapter
            print("[runpodx] Applying fresh LoRA adapter")
            lora_config = LoraConfig(
                r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                lora_dropout=config.get("lora_dropout", 0.05),
                target_modules=config.get("lora_target_modules"),
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

    # --- Dataset ---
    print("[runpodx] Loading dataset...")
    train_dataset, eval_dataset = load_sft_dataset(config)
    print(f"[runpodx] Train dataset size: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"[runpodx] Validation dataset size: {len(eval_dataset)}")

    # --- Training args ---
    common_args = dict(
        output_dir=config.get("output_dir", "/workspace/output"),
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 2e-5),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        weight_decay=config.get("weight_decay", 0.01),
        bf16=config.get("bf16", True),
        fp16=config.get("fp16", False),
        optim=config.get("optimizer", "adamw_8bit"),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 3),
        report_to=report_to,
        run_name=run_name,
        push_to_hub=config.get("push_to_hub", False),
        hub_model_id=config.get("hub_model_id") or None,
        hub_token=config.get("hub_token") or None,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    # trl >= 0.15: SFTConfig has max_seq_length; older: use TrainingArguments
    max_seq = config.get("max_seq_length", 2048)
    if SFTConfig is not None:
        try:
            training_args = build_training_args(
                SFTConfig,
                {"max_seq_length": max_seq, **common_args},
                eval_enabled=eval_dataset is not None,
            )
            max_seq = None  # already in config, don't pass to SFTTrainer
        except TypeError:
            training_args = build_training_args(
                TrainingArguments,
                common_args,
                eval_enabled=eval_dataset is not None,
            )
    else:
        training_args = build_training_args(
            TrainingArguments,
            common_args,
            eval_enabled=eval_dataset is not None,
        )

    # --- Train ---
    dataset_format = config.get("dataset_format", "alpaca")
    formatting_func = FORMATTERS.get(dataset_format)
    if formatting_func is None:
        print(f"[runpodx] ERROR: Unknown dataset_format '{dataset_format}'. "
              f"Available: {list(FORMATTERS.keys())}", file=sys.stderr)
        sys.exit(1)
    print(f"[runpodx] Using formatter: {dataset_format}")

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    if max_seq is not None:
        # Try passing to SFTTrainer; if rejected, set on tokenizer as fallback
        try:
            trainer = SFTTrainer(max_seq_length=max_seq, **trainer_kwargs)
        except TypeError:
            tokenizer.model_max_length = max_seq
            trainer = SFTTrainer(**trainer_kwargs)
    else:
        trainer = SFTTrainer(**trainer_kwargs)

    print("[runpodx] Starting training...")
    resume_from = config.get("resume_from_checkpoint")
    if resume_from == "latest":
        # Auto-detect latest checkpoint in output dir
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

    if eval_dataset is not None:
        print("[runpodx] Running final validation...")
        metrics = trainer.evaluate()
        print(f"[runpodx] Validation metrics: {metrics}")

    # --- Save ---
    print("[runpodx] Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub:
        print("[runpodx] Pushing to Hub...")
        trainer.push_to_hub()

    print("[runpodx] Training complete!")


if __name__ == "__main__":
    main()
