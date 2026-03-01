from __future__ import annotations

from pathlib import Path

from runpodx.training.configs import RunConfig
from runpodx.training.methods import BaseMethod, register_method

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


@register_method("sft")
class SFTMethod(BaseMethod):
    """Supervised Fine-Tuning method."""

    def script_path(self) -> Path:
        return _SCRIPTS_DIR / "sft_train.py"

    def requirements_path(self) -> Path:
        return _SCRIPTS_DIR / "requirements.txt"

    def generate_remote_config(self, run_config: RunConfig) -> dict:
        """Flatten RunConfig into the JSON format the SFT script expects."""
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

        return config

    def canonical_format(self) -> str:
        return "sft"
