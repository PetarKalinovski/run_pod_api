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
    terminate_after_hub_push: Annotated[
        bool,
        Field(
            description=(
                "Delete the pod after a successful training run that pushes the "
                "model to HuggingFace Hub."
            ),
        ),
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
    validation_split: Annotated[
        str | None,
        Field(description="HF split name for a pre-built validation set (e.g. 'validation'). Takes precedence over validation_ratio."),
    ] = None
    validation_ratio: Annotated[
        float | None,
        Field(
            gt=0.0,
            lt=1.0,
            description="Fraction of samples reserved for validation. None disables eval. Ignored if validation_split is set.",
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
    eval_batch_size: Annotated[int, Field(ge=1, description="Per-device eval batch size.")] = 1
    gradient_accumulation: Annotated[
        int, Field(ge=1, description="Gradient accumulation steps.")
    ] = 8
    max_seq_length: Annotated[int, Field(ge=1, description="Maximum sequence length.")] = 2048
    warmup_ratio: Annotated[float, Field(ge=0.0, le=1.0, description="Warmup ratio.")] = 0.03
    weight_decay: Annotated[float, Field(ge=0.0, description="Weight decay.")] = 0.01
    max_grad_norm: Annotated[
        float,
        Field(gt=0.0, description="Gradient clipping norm."),
    ] = 1.0
    bf16: Annotated[bool, Field(description="Use bfloat16.")] = True
    fp16: Annotated[bool, Field(description="Use float16.")] = False
    optimizer: Annotated[str, Field(description="Optimizer name.")] = "adamw_8bit"
    lr_scheduler_type: Annotated[str, Field(description="LR scheduler.")] = "cosine"
    logging_steps: Annotated[int, Field(ge=1, description="Log every N steps.")] = 10
    save_steps: Annotated[int, Field(ge=1, description="Save checkpoint every N steps.")] = 500
    save_total_limit: Annotated[int | None, Field(ge=1, description="Max checkpoints.")] = 3
    eval_steps: Annotated[int | None, Field(ge=1, description="Eval every N steps. If set, overrides per-epoch eval.")] = None
    append_eos_token: Annotated[
        bool,
        Field(description="Append EOS to each assistant/response target during SFT."),
    ] = True
    sanitize_ascii_outputs: Annotated[
        bool,
        Field(description="Trim pathological repeated tail lines in ASCII-art targets."),
    ] = False
    ascii_max_tail_line_repeats: Annotated[
        int | None,
        Field(ge=1, description="Max repeated copies allowed for the final output line."),
    ] = None
    neftune_noise_alpha: Annotated[
        float | None,
        Field(ge=0, description="NEFTune noise alpha for embedding regularization. Typical: 5-15."),
    ] = None
    generation_eval_enabled: Annotated[
        bool,
        Field(description="Run a prompt-based generation benchmark after training."),
    ] = False
    generation_eval_max_new_tokens: Annotated[
        int,
        Field(ge=1, description="Max new tokens to generate for the benchmark prompts."),
    ] = 384
    generation_eval_prompts: Annotated[
        list[str] | None,
        Field(description="Optional prompt list for the post-training generation benchmark."),
    ] = None
    convert_to_prompt_completion: Annotated[
        bool,
        Field(
            description=(
                "Convert the loaded dataset into prompt/completion pairs and train "
                "only on the completion tokens when supported by TRL."
            ),
        ),
    ] = False
    grpo_num_generations: Annotated[
        int,
        Field(ge=2, description="Number of sampled completions per prompt for GRPO."),
    ] = 4
    grpo_max_completion_length: Annotated[
        int,
        Field(ge=1, description="Maximum generated completion length for GRPO rollouts."),
    ] = 192
    grpo_soft_overlong_cache: Annotated[
        int,
        Field(
            ge=0,
            description=(
                "Token window near the GRPO completion cap where a soft overlong "
                "penalty starts to apply."
            ),
        ),
    ] = 32
    grpo_beta: Annotated[
        float,
        Field(ge=0.0, description="KL regularization coefficient for GRPO."),
    ] = 0.0
    grpo_temperature: Annotated[
        float,
        Field(gt=0.0, description="Sampling temperature for GRPO rollouts."),
    ] = 1.0
    grpo_top_p: Annotated[
        float,
        Field(gt=0.0, le=1.0, description="Top-p sampling value for GRPO rollouts."),
    ] = 1.0
    grpo_reward_weights: Annotated[
        list[float] | None,
        Field(description="Optional reward weights for the GRPO ASCII reward functions."),
    ] = None
    grpo_profile: Annotated[
        str | None,
        Field(
            description=(
                "Named GRPO profile or import path that supplies reward functions "
                "and dataset preparation for a GRPO run."
            ),
        ),
    ] = None
    grpo_clip_reward: Annotated[
        bool,
        Field(description="Enable CLIP visual reward for ASCII art GRPO. Adds a 6th reward function."),
    ] = False


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
        if self.run.terminate_after_hub_push and not self.output.push_to_hub:
            raise ValueError(
                "run.terminate_after_hub_push requires output.push_to_hub = true."
            )
        if self.run.method == "grpo" and not self.training.grpo_profile:
            raise ValueError("training.grpo_profile is required when run.method = 'grpo'.")
        return self
