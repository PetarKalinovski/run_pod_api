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
from collections.abc import Mapping
from functools import partial
from pathlib import Path

import torch
if torch.cuda.is_available():
    torch.cuda.init()
    print(f"[runpodx] CUDA initialized: {torch.cuda.get_device_name(0)}")
else:
    print("[runpodx] WARNING: CUDA not available, forcing bf16 skip")
    os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "no")
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

try:
    from sft_shared import (
        DEFAULT_ASCII_BENCHMARK_PROMPTS,
        analyze_ascii_response,
        build_generation_prompt,
        convert_dataset_to_prompt_completion,
        format_alpaca_example,
        format_sharegpt_example,
        resolve_wandb_run_id,
        summarize_generation_benchmark,
        write_generation_benchmark_report,
    )
except ImportError:
    from runpodx.training.scripts.sft_shared import (
        DEFAULT_ASCII_BENCHMARK_PROMPTS,
        analyze_ascii_response,
        build_generation_prompt,
        convert_dataset_to_prompt_completion,
        format_alpaca_example,
        format_sharegpt_example,
        resolve_wandb_run_id,
        summarize_generation_benchmark,
        write_generation_benchmark_report,
    )

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
    base_args = dict(common_args)
    if eval_enabled:
        base_args["eval_strategy"] = "epoch"

    optional_keys = ("completion_only_loss", "assistant_only_loss")
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
            for optional_key in optional_keys:
                if optional_key in candidate:
                    trimmed = dict(candidate)
                    trimmed.pop(optional_key)
                    queue.append(trimmed)

    raise last_exc


def build_formatting_func(dataset_format: str, tokenizer, config: dict):
    formatter_kwargs = {
        "eos_token": tokenizer.eos_token or "",
        "append_eos": config.get("append_eos_token", True),
        "sanitize_ascii": config.get("sanitize_ascii_outputs", False),
        "max_tail_line_repeats": config.get("ascii_max_tail_line_repeats"),
    }
    formatters = {
        "alpaca": partial(format_alpaca_example, **formatter_kwargs),
        "sharegpt": partial(format_sharegpt_example, **formatter_kwargs),
    }
    return formatters.get(dataset_format), list(formatters.keys())


def maybe_convert_datasets_to_prompt_completion(
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    dataset_format: str,
    tokenizer,
    config: dict,
) -> tuple[Dataset, Dataset | None, str]:
    if not config.get("convert_to_prompt_completion"):
        return train_dataset, eval_dataset, dataset_format

    converter_kwargs = {
        "source_format": dataset_format,
        "eos_token": tokenizer.eos_token or "",
        "append_eos": config.get("append_eos_token", True),
        "sanitize_ascii": config.get("sanitize_ascii_outputs", False),
        "max_tail_line_repeats": config.get("ascii_max_tail_line_repeats"),
    }
    original_train_size = len(train_dataset)
    train_dataset = convert_dataset_to_prompt_completion(train_dataset, **converter_kwargs)
    print(
        "[runpodx] Converted train dataset to prompt/completion: "
        f"{original_train_size} -> {len(train_dataset)}"
    )

    if eval_dataset is not None:
        original_eval_size = len(eval_dataset)
        eval_dataset = convert_dataset_to_prompt_completion(eval_dataset, **converter_kwargs)
        print(
            "[runpodx] Converted validation dataset to prompt/completion: "
            f"{original_eval_size} -> {len(eval_dataset)}"
        )

    return train_dataset, eval_dataset, "prompt_completion"


def run_generation_benchmark(model, tokenizer, dataset_format: str, config: dict, report_to: str):
    prompts = config.get("generation_eval_prompts") or DEFAULT_ASCII_BENCHMARK_PROMPTS
    if not config.get("generation_eval_enabled") and not config.get("generation_eval_prompts"):
        return None

    print("[runpodx] Running generation benchmark...")
    benchmark_rows = []
    prompt_device = next(model.parameters()).device
    max_new_tokens = config.get("generation_eval_max_new_tokens", 384)

    for prompt in prompts:
        rendered_prompt = build_generation_prompt(prompt, dataset_format)
        tokenized = tokenizer(rendered_prompt, return_tensors="pt")
        tokenized = {key: value.to(prompt_device) for key, value in tokenized.items()}
        with torch.no_grad():
            generated = model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = generated[0][tokenized["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        row = {
            "prompt": prompt,
            "response": response,
            "terminated_on_eos": int(tokenizer.eos_token_id in generated_ids.tolist()),
        }
        row.update(analyze_ascii_response(response))
        benchmark_rows.append(row)

    aggregate = summarize_generation_benchmark(benchmark_rows)
    print(f"[runpodx] Generation benchmark aggregate: {json.dumps(aggregate, indent=2)}")
    json_report_path, md_report_path = write_generation_benchmark_report(
        config.get("output_dir", "/workspace/output"),
        aggregate,
        benchmark_rows,
    )
    print(f"[runpodx] Wrote benchmark reports: {json_report_path} | {md_report_path}")

    if report_to == "wandb":
        try:
            import wandb

            wandb.log({f"benchmark/{key}": value for key, value in aggregate.items()})
            columns = [
                "prompt",
                "response",
                "line_count",
                "non_empty_line_count",
                "repeat_line_fraction",
                "max_consecutive_repeat",
                "tail_repeat_count",
                "char_count",
                "width_std",
                "occupancy_std",
                "alnum_fraction",
                "is_too_short",
                "is_looping",
                "is_punctuation_blob",
                "contains_non_art_text",
                "failure_tags",
                "terminated_on_eos",
            ]
            table = wandb.Table(columns=columns)
            for row in benchmark_rows:
                table.add_data(*[row.get(column) for column in columns])
            wandb.log({"benchmark/samples": table})
        except Exception as exc:
            print(f"[runpodx] W&B benchmark logging skipped: {exc}")

    return {"aggregate": aggregate, "rows": benchmark_rows, "json_report_path": json_report_path, "md_report_path": md_report_path}


def patch_rope_compat() -> None:
    """Patch transformers rope helpers for older custom remote-code models."""
    try:
        from transformers import modeling_rope_utils as rope_utils
    except ImportError:
        return

    rope_init_functions = getattr(rope_utils, "ROPE_INIT_FUNCTIONS", None)
    if not isinstance(rope_init_functions, dict):
        return

    if "default" not in rope_init_functions:
        default_fn = getattr(rope_utils, "_compute_default_rope_parameters", None)
        if callable(default_fn):
            rope_init_functions["default"] = default_fn
        elif rope_init_functions:
            # Last-resort fallback for custom model code that expects a "default" entry.
            rope_init_functions["default"] = next(iter(rope_init_functions.values()))

    linear_fn = getattr(rope_utils, "_compute_linear_scaling_rope_parameters", None)
    if not callable(linear_fn):
        return

    def _patched_linear_scaling_rope_parameters(config, device, seq_len=None, **rope_kwargs):
        for attr in ("rope_parameters", "rope_scaling"):
            rope_cfg = getattr(config, attr, None)
            if rope_cfg is None:
                continue
            if isinstance(rope_cfg, Mapping):
                patched = dict(rope_cfg)
            elif hasattr(rope_cfg, "items"):
                patched = dict(rope_cfg.items())
            else:
                continue
            if "factor" not in patched:
                patched["factor"] = 1.0
                setattr(config, attr, patched)
        return linear_fn(config, device, seq_len=seq_len, **rope_kwargs)

    rope_utils._compute_linear_scaling_rope_parameters = _patched_linear_scaling_rope_parameters
    # Trinity's remote code looks up the callable from ROPE_INIT_FUNCTIONS,
    # so replace that entry directly regardless of what it currently points to.
    rope_init_functions["linear"] = _patched_linear_scaling_rope_parameters


def patch_model_config_compat(model_config, tokenizer) -> None:
    """Patch incomplete config fields used by Trinity's custom remote code."""
    if getattr(model_config, "pad_token_id", None) is None:
        model_config.pad_token_id = tokenizer.pad_token_id
    if getattr(model_config, "eos_token_id", None) is None:
        model_config.eos_token_id = tokenizer.eos_token_id

    for attr in ("rope_parameters", "rope_scaling"):
        rope_cfg = getattr(model_config, attr, None)
        if rope_cfg is None:
            continue
        if isinstance(rope_cfg, Mapping):
            patched = dict(rope_cfg)
        elif hasattr(rope_cfg, "items"):
            patched = dict(rope_cfg.items())
        else:
            continue
        if "rope_type" not in patched and "type" in patched:
            patched["rope_type"] = patched["type"]
        if "factor" not in patched:
            # A factor of 1.0 preserves base behavior while satisfying newer helpers.
            patched["factor"] = 1.0

        setattr(model_config, attr, patched)


def patch_model_loading_compat() -> None:
    """Patch newer transformers init hooks for Trinity's custom rotary modules."""
    try:
        from transformers import modeling_utils
    except ImportError:
        return

    if getattr(modeling_utils.PreTrainedModel, "_runpodx_trinity_init_patch", False):
        return

    original_init_weights = modeling_utils.PreTrainedModel._init_weights

    def _patched_init_weights(self, module):
        try:
            return original_init_weights(self, module)
        except AttributeError as exc:
            if "compute_default_rope_parameters" not in str(exc):
                raise

            # Trinity's custom rotary embedding already initializes itself in its own
            # constructor, but newer transformers tries to re-run a code path that
            # assumes a newer rotary API. Skip that extra init for this module only.
            if module.__class__.__name__.endswith("RotaryEmbedding"):
                return None
            raise

    modeling_utils.PreTrainedModel._init_weights = _patched_init_weights
    modeling_utils.PreTrainedModel._runpodx_trinity_init_patch = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[runpodx] Config loaded: {json.dumps(config, indent=2)}")

    # --- W&B setup ---
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

    # --- Load model + tokenizer ---
    model_name = config["model_name"]
    print(f"[runpodx] Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    patch_model_loading_compat()
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    patch_model_config_compat(model_config, tokenizer)
    patch_rope_compat()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_config,
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
    dataset_format = config.get("dataset_format", "alpaca")
    train_dataset, eval_dataset, training_dataset_format = maybe_convert_datasets_to_prompt_completion(
        train_dataset,
        eval_dataset,
        dataset_format,
        tokenizer,
        config,
    )

    # --- Training args ---
    common_args = dict(
        output_dir=config.get("output_dir", "/workspace/output"),
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 2e-5),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=config.get("max_grad_norm", 1.0),
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
    # NEFTune: add noise to embeddings during training (regularization)
    neftune_alpha = config.get("neftune_noise_alpha")
    if neftune_alpha is not None:
        common_args["neftune_noise_alpha"] = float(neftune_alpha)
        print(f"[runpodx] NEFTune enabled: noise_alpha={neftune_alpha}")

    if config.get("convert_to_prompt_completion"):
        common_args["completion_only_loss"] = True

    # Sequence length: trl >= 0.29 renamed max_seq_length -> max_length in SFTConfig.
    # Try SFTConfig first (gets max_length right), fall back to TrainingArguments.
    max_seq = config.get("max_seq_length", 2048)
    print(f"[runpodx] Configured max sequence length: {max_seq}")
    training_args = None
    if SFTConfig is not None:
        for seq_key in ("max_length", "max_seq_length"):
            try:
                training_args = build_training_args(
                    SFTConfig,
                    {seq_key: max_seq, **common_args},
                    eval_enabled=eval_dataset is not None,
                )
                print(f"[runpodx] Built SFTConfig with {seq_key}={max_seq}")
                max_seq = None  # already in config
                break
            except (TypeError, ValueError):
                continue
    if training_args is None:
        print("[runpodx] Falling back to TrainingArguments")
        training_args = build_training_args(
            TrainingArguments,
            common_args,
            eval_enabled=eval_dataset is not None,
        )

    # --- Train ---
    formatting_func = None
    if training_dataset_format != "prompt_completion":
        formatting_func, available_formatters = build_formatting_func(
            training_dataset_format, tokenizer, config
        )
        if formatting_func is None:
            print(
                f"[runpodx] ERROR: Unknown dataset_format '{training_dataset_format}'. "
                f"Available: {available_formatters}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[runpodx] Using formatter: {training_dataset_format}")
    else:
        print("[runpodx] Using prompt/completion dataset representation")

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    if formatting_func is not None:
        trainer_kwargs["formatting_func"] = formatting_func
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

    run_generation_benchmark(model, tokenizer, training_dataset_format, config, report_to)

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
