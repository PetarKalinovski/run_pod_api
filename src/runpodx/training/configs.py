from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from runpodx.configs.base import BaseConfig
from runpodx.configs.pod import GpuConfig


class RunMetaConfig(BaseConfig):
    """Top-level run metadata."""

    method: Annotated[
        Literal["sft", "grpo", "unsloth_grpo"],
        Field(description="Training method to use."),
    ]
    pod_id: Annotated[
        str | None,
        Field(description="Existing pod ID. Omit to create a new pod."),
    ] = None
    terminate_on_complete: Annotated[
        bool,
        Field(description="Delete the pod after training completes successfully."),
    ] = False
    download_on_complete: Annotated[
        str | None,
        Field(
            description=(
                "Local directory to download the output to after training. "
                "Downloads from [output].save_path. None = skip download."
            ),
        ),
    ] = None


class PodSpecConfig(BaseConfig):
    """Pod specification for provisioning a new pod."""

    name: Annotated[str, Field(description="Pod name.")] = "runpodx-training"
    image: Annotated[str, Field(description="Container image.")] = (
        "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
    )
    cloud_type: Annotated[
        Literal["SECURE", "COMMUNITY"],
        Field(description="Cloud type."),
    ] = "SECURE"
    gpu: Annotated[GpuConfig, Field(description="GPU configuration.")] = GpuConfig()
    container_disk_gb: Annotated[int, Field(ge=1, description="Container disk in GB.")] = 50
    volume_gb: Annotated[int, Field(ge=0, description="Persistent volume in GB.")] = 50
    volume_mount_path: Annotated[str, Field(description="Volume mount path.")] = "/workspace"


class ModelConfig(BaseConfig):
    """Model and adapter configuration."""

    name: Annotated[str, Field(description="HuggingFace model ID.")]
    adapter: Annotated[
        Literal["lora", "full"],
        Field(description="Fine-tuning strategy."),
    ] = "lora"
    adapter_path: Annotated[
        str | None,
        Field(
            description=(
                "Path to existing LoRA adapter weights to load before training. "
                "Can be a HuggingFace ID (e.g., 'user/my-lora-adapter') or "
                "a local path that gets uploaded to the pod."
            ),
        ),
    ] = None
    resume_from_checkpoint: Annotated[
        str | None,
        Field(
            description=(
                "Path to a training checkpoint to resume from (on the pod). "
                "Use 'latest' to auto-detect the latest checkpoint in output dir."
            ),
        ),
    ] = None
    lora_r: Annotated[int, Field(ge=1, description="LoRA rank.")] = 16
    lora_alpha: Annotated[int, Field(ge=1, description="LoRA alpha.")] = 32
    lora_dropout: Annotated[float, Field(ge=0.0, le=1.0, description="LoRA dropout.")] = 0.05
    lora_target_modules: Annotated[
        list[str] | None,
        Field(description="Target modules for LoRA. None = auto-detect."),
    ] = None


class DatasetConfig(BaseConfig):
    """Dataset source and format configuration."""

    source: Annotated[
        Literal["huggingface", "local"],
        Field(description="Where the dataset comes from."),
    ]
    path: Annotated[str, Field(description="HF dataset ID or local file path.")]
    split: Annotated[str, Field(description="Dataset split.")] = "train"
    format: Annotated[
        Literal["alpaca", "sharegpt", "completion", "custom"],
        Field(description="Input data format for normalization."),
    ] = "alpaca"
    max_samples: Annotated[int | None, Field(ge=1, description="Limit dataset size.")] = None
    validation_ratio: Annotated[
        float | None,
        Field(
            gt=0.0,
            lt=1.0,
            description="Fraction of samples reserved for validation. None disables eval.",
        ),
    ] = None
    validation_seed: Annotated[
        int,
        Field(description="Random seed used when splitting train/validation."),
    ] = 42
    field_mapping: Annotated[
        dict[str, str] | None,
        Field(description="Field mapping for custom format."),
    ] = None

    @model_validator(mode="after")
    def validate_custom_mapping(self):
        if self.format == "custom" and not self.field_mapping:
            raise ValueError("field_mapping is required when format='custom'")
        return self


class TrainingConfig(BaseConfig):
    """Training hyperparameters."""

    epochs: Annotated[int, Field(ge=1, description="Number of training epochs.")] = 3
    lr: Annotated[float, Field(gt=0, description="Learning rate.")] = 2e-5
    batch_size: Annotated[int, Field(ge=1, description="Per-device batch size.")] = 4
    gradient_accumulation: Annotated[
        int, Field(ge=1, description="Gradient accumulation steps.")
    ] = 8
    max_seq_length: Annotated[int, Field(ge=1, description="Maximum sequence length.")] = 2048
    warmup_ratio: Annotated[float, Field(ge=0.0, le=1.0, description="Warmup ratio.")] = 0.03
    weight_decay: Annotated[float, Field(ge=0.0, description="Weight decay.")] = 0.01
    bf16: Annotated[bool, Field(description="Use bfloat16.")] = True
    fp16: Annotated[bool, Field(description="Use float16.")] = False
    optimizer: Annotated[str, Field(description="Optimizer name.")] = "adamw_8bit"
    lr_scheduler_type: Annotated[str, Field(description="LR scheduler.")] = "cosine"
    logging_steps: Annotated[int, Field(ge=1, description="Log every N steps.")] = 10
    save_steps: Annotated[int, Field(ge=1, description="Save checkpoint every N steps.")] = 500
    save_total_limit: Annotated[int | None, Field(ge=1, description="Max checkpoints.")] = 3


class OutputConfig(BaseConfig):
    """Output and Hub push configuration."""

    save_path: Annotated[str, Field(description="Save path on the pod.")] = "/workspace/output"
    push_to_hub: Annotated[bool, Field(description="Push to HuggingFace Hub.")] = False
    hub_model_id: Annotated[str, Field(description="Hub model ID.")] = ""
    hub_token: Annotated[str | None, Field(description="Hub token.")] = None


class MonitoringConfig(BaseConfig):
    """Weights & Biases monitoring configuration."""

    wandb_project: Annotated[str, Field(description="W&B project name.")] = ""
    wandb_run_name: Annotated[str, Field(description="W&B run name.")] = ""
    wandb_entity: Annotated[str | None, Field(description="W&B entity/team.")] = None
    log_interval: Annotated[int, Field(ge=1, description="Log every N steps.")] = 10


class SSHConfig(BaseConfig):
    """SSH connection settings."""

    key_path: Annotated[
        str | None,
        Field(description="Path to SSH private key. Default: ~/.ssh/id_ed25519"),
    ] = None
    port: Annotated[int, Field(description="SSH port.")] = 22
    user: Annotated[str, Field(description="SSH user.")] = "root"
    connect_timeout: Annotated[int, Field(ge=1, description="SSH connect timeout.")] = 30


class RunConfig(BaseConfig):
    """Top-level config for `runpodx run`. One TOML file = one RunConfig."""

    run: Annotated[RunMetaConfig, Field(description="Run metadata.")]
    pod: Annotated[
        PodSpecConfig | None,
        Field(description="Pod spec. Ignored if run.pod_id is set."),
    ] = None
    model: Annotated[ModelConfig, Field(description="Model configuration.")]
    dataset: Annotated[DatasetConfig, Field(description="Dataset configuration.")]
    training: Annotated[
        TrainingConfig, Field(description="Training hyperparameters.")
    ] = TrainingConfig()
    output: Annotated[OutputConfig, Field(description="Output configuration.")] = OutputConfig()
    monitoring: Annotated[
        MonitoringConfig, Field(description="W&B monitoring.")
    ] = MonitoringConfig()
    ssh: Annotated[SSHConfig, Field(description="SSH settings.")] = SSHConfig()

    @model_validator(mode="after")
    def validate_pod_config(self):
        if self.run.pod_id is None and self.pod is None:
            raise ValueError(
                "Either run.pod_id must be set (existing pod) "
                "or [pod] section must be provided (create new pod)."
            )
        return self
