# AGENTS.md — runpodx

Instructions for AI coding agents (Codex, Cursor, OpenCode, Windsurf, etc.).

## Project Summary

runpodx is a config-driven Python client and CLI for RunPod's REST API (`rest.runpod.io/v1`).
Two layers: (1) typed SDK for pod/template/volume/endpoint management, (2) training orchestrator
that provisions a GPU pod, SSHs in, uploads scripts + data, and runs fine-tuning end-to-end.

## Setup

```bash
uv sync                          # Install deps
uv run pytest tests/ -v          # Run all tests (~1s)
uv run ruff check src/ tests/    # Lint
uv run runpodx --help            # CLI
```

## Source Layout

```
src/runpodx/
├── client.py                    # RunPodClient — top-level entry point
├── cli/                         # Typer CLI (main.py, pods.py, run.py, templates.py, etc.)
├── configs/                     # Pydantic config schemas (pod.py, template.py, endpoint.py, network_volume.py, loader.py)
├── models/                      # API response models (pod.py, template.py, etc.)
├── resources/                   # Resource classes with actions (pods.py, templates.py, etc.)
├── http/                        # HttpTransport + error hierarchy
└── training/
    ├── orchestrator.py          # TrainingOrchestrator — pipeline controller
    ├── executor.py              # SSHConnection + SSHExecutor (subprocess ssh/scp)
    ├── configs.py               # RunConfig + all sub-configs for training
    ├── methods/                 # Registry: BaseMethod, SFTMethod, GRPOMethod
    ├── normalizers/             # Registry: dataset format normalizers (alpaca, sharegpt, completion, custom)
    └── scripts/                 # Uploaded to pod: sft_train.py, grpo_train.py, requirements.txt
```

## Architecture Rules

- **REST API only** — no GraphQL. REST returns `portMappings` dict + `publicIp`.
- **SSH via subprocess** — not paramiko. Uses user's SSH keys.
- **File transfer via scp** — not rsync (unavailable on Windows).
- **PUBLIC_KEY env var** — must be injected when creating pods via REST API for SSH to work.
- **Registry pattern** — `@register_method("sft")` / `@register_normalizer("sft", "alpaca")`.
  Adding a new training method = drop a new file in `methods/` and register it.
- **Training scripts are standalone .py** — uploaded to pod. Heavy deps (torch, transformers, trl)
  are only installed on the pod, never locally.
- **Config resolution** — `resolve_config(ConfigClass, source)` accepts TOML path, dict, or
  existing config instance.
- **Pydantic v2** with `ConfigDict(extra="allow", populate_by_name=True)` for API models.
  Field aliases map camelCase API fields to snake_case Python.

## Environment Variables

- `RUNPOD_API_KEY` — required for all API calls
- `RUNPOD_PRIVATE_KEY` — SSH key passphrase (optional)
- `WANDB_API_KEY` — forwarded to pod for W&B logging
- `HF_TOKEN` — forwarded to pod for HuggingFace Hub

## Training Methods

| Method | Script | Description |
|--------|--------|-------------|
| `sft` | `sft_train.py` | Supervised fine-tuning. Supports LoRA and full fine-tune |
| `grpo` | `grpo_train.py` | Group Relative Policy Optimization. Requires `training.grpo_profile` |
| `unsloth_grpo` | `grpo_train.py` | GRPO via Unsloth runtime |

Models starting with `arcee-ai/trinity-` auto-select the `sft_train_trinity.py` variant.

## Training Config TOML Sections

All training is driven by a single TOML file passed to `runpodx run <config.toml>`.

### `[run]` — Run Metadata
- `method` (required): `"sft"`, `"grpo"`, or `"unsloth_grpo"`
- `pod_id`: existing pod ID (omit to create new pod from `[pod]`)
- `terminate_on_complete`: delete pod after training succeeds
- `terminate_after_hub_push`: delete pod after successful Hub push (requires `output.push_to_hub`)
- `download_on_complete`: local dir to download output to

### `[pod]` + `[pod.gpu]` — Pod Provisioning
- `name`, `image`, `cloud_type` (`"SECURE"` / `"COMMUNITY"`), `container_disk_gb`, `volume_gb`, `volume_mount_path`
- GPU: `type_ids`, `count`, `type_priority`, `min_ram_per_gpu`, `min_vcpu_per_gpu`, `allowed_cuda_versions`

### `[model]` — Model & Adapter
- `name` (required): HuggingFace model ID
- `adapter`: `"lora"` or `"full"`
- `adapter_path`: existing LoRA weights (HF ID or local path)
- `resume_from_checkpoint`: checkpoint path or `"latest"`
- LoRA params: `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules`

### `[dataset]` — Dataset Source & Format
- `source` (required): `"huggingface"` or `"local"`
- `path` (required): HF dataset ID or local file path
- `split`, `format` (`"alpaca"`, `"sharegpt"`, `"completion"`, `"custom"`), `max_samples`
- `validation_ratio`, `validation_seed`
- `field_mapping`: required when `format = "custom"`

### `[training]` — Hyperparameters
- Core: `epochs`, `lr`, `batch_size`, `gradient_accumulation`, `max_seq_length`
- Precision: `bf16`, `fp16`
- Optimizer: `optimizer`, `lr_scheduler_type`, `warmup_ratio`, `weight_decay`
- Checkpointing: `logging_steps`, `save_steps`, `save_total_limit`
- SFT-specific: `append_eos_token`, `convert_to_prompt_completion`
- ASCII-art: `sanitize_ascii_outputs`, `ascii_max_tail_line_repeats`
- Generation eval: `generation_eval_enabled`, `generation_eval_max_new_tokens`, `generation_eval_prompts`
- GRPO-specific: `grpo_profile` (required for GRPO), `grpo_num_generations`, `grpo_max_completion_length`, `grpo_soft_overlong_cache`, `grpo_beta`, `grpo_temperature`, `grpo_top_p`, `grpo_reward_weights`

### `[output]` — Model Output
- `save_path`, `push_to_hub`, `hub_model_id`, `hub_token`

### `[monitoring]` — Weights & Biases
- `wandb_project`, `wandb_run_name`, `wandb_entity`, `log_interval`

### `[ssh]` — SSH Connection
- `key_path`, `port`, `user`, `connect_timeout`

## Config Validation Rules

- Either `run.pod_id` OR `[pod]` section must be provided
- `run.terminate_after_hub_push` requires `output.push_to_hub = true`
- `run.method = "grpo"` requires `training.grpo_profile` to be set
- `dataset.format = "custom"` requires `dataset.field_mapping` to be set

## Testing

- All tests use mock transport — no real API calls
- `conftest.py` provides `sample_pod_data` with `portMappings` for SSH tests
- Training orchestrator tests mock SSHExecutor and SSHConnection
- Run: `uv run pytest tests/ -v`
- Lint: `uv run ruff check src/ tests/`

## Gotchas

- RunPod REST API returns `runtime: null` — do NOT rely on `runtime.ports`
- `wait_until_ready` checks truthiness of `public_ip` (empty string = not ready)
- `supportPublicIp: true` required in pod creation for TCP SSH to work
- trl library API varies by version — training scripts handle SFTConfig/TrainingArguments compat
- Windows: no rsync, use scp. SSH encoding must be `utf-8` with `errors="replace"`
- `terminate_on_complete` only fires on success — failed training leaves the pod alive for debugging
- Models starting with `arcee-ai/trinity-` auto-select the Trinity training script variant
