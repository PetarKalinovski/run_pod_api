from __future__ import annotations

from pathlib import Path

from runpodx.training.configs import RunConfig
from runpodx.training.methods import BaseMethod, register_method

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


@register_method("grpo")
class GRPOMethod(BaseMethod):
    """Group Relative Policy Optimization training method."""

    def script_path(self, run_config: RunConfig | None = None) -> Path:
        return _SCRIPTS_DIR / "grpo_train.py"

    def requirements_path(self, run_config: RunConfig | None = None) -> Path:
        return _SCRIPTS_DIR / "requirements.txt"

    def generate_remote_config(self, run_config: RunConfig) -> dict:
        m = run_config.model
        d = run_config.dataset
        t = run_config.training
        o = run_config.output
        mon = run_config.monitoring

        config: dict = {
            "model_name": m.name,
            "adapter": m.adapter,
            "dataset_format": d.format,
            "dataset_path": "/workspace/runpodx_run/data/dataset.jsonl",
            "max_seq_length": t.max_seq_length,
            "num_train_epochs": t.epochs,
            "learning_rate": t.lr,
            "per_device_train_batch_size": t.batch_size,
            "gradient_accumulation_steps": t.gradient_accumulation,
            "warmup_ratio": t.warmup_ratio,
            "weight_decay": t.weight_decay,
            "bf16": t.bf16,
            "fp16": t.fp16,
            "optimizer": t.optimizer,
            "lr_scheduler_type": t.lr_scheduler_type,
            "logging_steps": t.logging_steps,
            "save_steps": t.save_steps,
            "save_total_limit": t.save_total_limit,
            "sanitize_ascii_outputs": t.sanitize_ascii_outputs,
            "ascii_max_tail_line_repeats": t.ascii_max_tail_line_repeats,
            "generation_eval_enabled": t.generation_eval_enabled,
            "generation_eval_max_new_tokens": t.generation_eval_max_new_tokens,
            "generation_eval_prompts": t.generation_eval_prompts,
            "grpo_num_generations": t.grpo_num_generations,
            "grpo_max_completion_length": t.grpo_max_completion_length,
            "grpo_soft_overlong_cache": t.grpo_soft_overlong_cache,
            "grpo_beta": t.grpo_beta,
            "grpo_temperature": t.grpo_temperature,
            "grpo_top_p": t.grpo_top_p,
            "grpo_reward_weights": t.grpo_reward_weights,
            "grpo_profile": t.grpo_profile,
            "output_dir": o.save_path,
            "push_to_hub": o.push_to_hub,
            "hub_model_id": o.hub_model_id,
            "hub_token": o.hub_token,
            "wandb_project": mon.wandb_project,
            "wandb_run_name": mon.wandb_run_name,
            "wandb_entity": mon.wandb_entity,
        }

        if m.adapter == "lora":
            config.update(
                {
                    "lora_r": m.lora_r,
                    "lora_alpha": m.lora_alpha,
                    "lora_dropout": m.lora_dropout,
                    "lora_target_modules": m.lora_target_modules,
                }
            )

        if m.adapter_path:
            config["adapter_path"] = m.adapter_path

        if m.resume_from_checkpoint:
            config["resume_from_checkpoint"] = m.resume_from_checkpoint

        if d.source == "huggingface":
            config["hf_dataset"] = d.path
            config["hf_split"] = d.split

        if d.max_samples is not None:
            config["max_samples"] = d.max_samples

        if d.validation_ratio is not None:
            config["validation_ratio"] = d.validation_ratio
            config["validation_seed"] = d.validation_seed

        return config

    def canonical_format(self) -> str:
        # GRPO uses the same source data shapes as SFT before conversion to prompt/target pairs.
        return "sft"
