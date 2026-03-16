# runpodx

Config-driven Python client and CLI for [RunPod's](https://runpod.io) REST API. Manage GPU pods, templates, and network volumes programmatically — or run end-to-end LLM fine-tuning with a single TOML file.

## Features

- **Pod Management** — create, start, stop, restart, delete GPU pods
- **Template & Volume Management** — manage pod templates and network storage
- **Training Orchestrator** — provision a pod, SSH in, upload scripts, install deps, train, stream logs
- **Config-Driven** — everything controlled via TOML files
- **Dataset Formats** — Alpaca, ShareGPT, completion, custom field mapping
- **LoRA & Full Fine-Tuning** — with automatic PEFT setup
- **W&B Integration** — stream metrics to Weights & Biases
- **HuggingFace Hub** — push trained models directly

## Installation

```bash
pip install runpodx
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add runpodx
```

## Setup

Set your RunPod API key:

```bash
export RUNPOD_API_KEY="your-api-key"
```

Or create a `.env` file:

```env
RUNPOD_API_KEY=your-api-key
WANDB_API_KEY=your-wandb-key       # optional, for W&B logging
HF_TOKEN=your-hf-token             # optional, for HuggingFace Hub push
RUNPOD_PRIVATE_KEY=your-ssh-pass   # optional, SSH key passphrase
```

## Quick Start

### CLI

```bash
# List your pods
runpodx pods list

# Create a pod from config
runpodx pods create pod.toml

# Run a training job
runpodx run training.toml

# Dry run — preview what would happen
runpodx run training.toml --dry-run

# Use an existing pod instead of creating a new one
runpodx run training.toml --pod pod_abc123

# Download checkpoints from a pod
runpodx download pod_abc123 --local ./checkpoints
```

### Python API

```python
from runpodx import RunPodClient, PodCreateConfig, GpuConfig

client = RunPodClient()

# List pods
for pod in client.pods.list():
    print(f"{pod.name} ({pod.id}) — {pod.desired_status}")

# Create a pod
config = PodCreateConfig(
    name="my-gpu-pod",
    image="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    gpu=GpuConfig(type_ids=["NVIDIA GeForce RTX 4090"]),
)
pod = client.pods.create(config)
pod.wait_until_ready()

# Or create from TOML
pod = client.pods.create("examples/basic_pod.toml")

# Stop when done
pod.stop()
client.close()
```

---

## Training

runpodx automates the full fine-tuning pipeline:

1. **Provision** — create a GPU pod or connect to an existing one
2. **Wait** — poll until pod is running with SSH access
3. **Connect** — establish SSH connection (handles key passphrase automatically)
4. **Upload** — transfer training script, requirements, and config
5. **Install** — `pip install` dependencies on the pod (streamed to your terminal)
6. **Train** — run training with live log streaming back to your terminal

### Full Example

```toml
# ascii_art_sft.toml

[run]
method = "sft"                      # Training method: "sft", "grpo", or ..
# pod_id = "pod_abc123"             # Use an existing pod instead of creating a new one
# terminate_on_complete = true      # Delete the pod after training finishes (saves money)
# download_on_complete = "./output" # Download trained model to this local dir after training

[pod]
name = "qwen3-ascii-art-sft"       # Display name for the pod
image = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"  # Docker image
cloud_type = "SECURE"               # "SECURE" (dedicated) or "COMMUNITY" (shared, cheaper)
container_disk_gb = 80              # Ephemeral disk (destroyed on pod delete)
volume_gb = 100                     # Persistent volume (survives restarts)

[pod.gpu]
type_ids = ["NVIDIA A100-SXM4-80GB"]  # GPU models to try, in preference order
count = 1                              # Number of GPUs

[model]
name = "Qwen/Qwen3-8B"             # HuggingFace model ID to fine-tune
adapter = "full"                    # "lora" (parameter-efficient) or "full" (all weights)

[dataset]
source = "huggingface"              # "huggingface" (loaded on pod) or "local" (uploaded)
path = "AvaLovelace/ASCII-Art"      # HF dataset ID or local file path
split = "train"                     # Dataset split to use
format = "sharegpt"                 # "alpaca", "sharegpt", "completion", or "custom"

[training]
epochs = 3                          # Number of full passes over the dataset
lr = 5e-6                           # Learning rate (lower for full fine-tune: 1e-6 to 5e-6)
batch_size = 2                      # Per-GPU batch size (lower if OOM)
gradient_accumulation = 16          # Steps before weight update (effective batch = 2 * 16 = 32)
max_seq_length = 4096               # Max tokens per sample (longer = more VRAM)
bf16 = true                         # bfloat16 mixed precision (use fp16 if no A100/H100)
optimizer = "adamw_8bit"            # "adamw_8bit", "adamw_torch", "sgd", "adafactor"

[output]
save_path = "/workspace/output"     # Where the trained model is saved on the pod
push_to_hub = true                  # Push final model to HuggingFace Hub
hub_model_id = "your-username/Qwen3-8B-ascii-art"  # HF repo to push to

[monitoring]
wandb_project = "qwen3-ascii-art"   # W&B project name (empty = disabled)
wandb_run_name = "qwen3-8b-full-sft"  # Name for this run in W&B
```

```bash
runpodx run ascii_art_sft.toml
```

### GRPO Example

```toml
# ascii_art_grpo.toml

[run]
method = "grpo"
terminate_after_hub_push = true          # Delete pod after pushing model to Hub

[pod]
image = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"

[pod.gpu]
type_ids = ["NVIDIA RTX A6000"]
count = 1

[model]
name = "PetarKal/Qwen3-4B-ascii-art-sft"  # Start from an SFT checkpoint
adapter = "lora"

[dataset]
source = "huggingface"
path = "AvaLovelace/ASCII-Art"
format = "sharegpt"
max_samples = 2048
validation_ratio = 0.1

[training]
epochs = 1
lr = 1e-5
batch_size = 1
gradient_accumulation = 16
grpo_profile = "ascii_art"               # Reward function profile (required for GRPO)
grpo_num_generations = 4                 # Completions sampled per prompt
grpo_max_completion_length = 192         # Max tokens per completion
grpo_beta = 0.05                         # KL regularization
grpo_temperature = 0.9                   # Sampling temperature
grpo_reward_weights = [0.45, 0.75, 1.25, 0.9]  # Per-reward-function weights
sanitize_ascii_outputs = true
generation_eval_enabled = true           # Run generation benchmark after training

[output]
push_to_hub = true
hub_model_id = "your-username/Qwen3-4B-ascii-art-grpo"
```

```bash
runpodx run ascii_art_grpo.toml
```

---

## Training Configuration Reference

A training config is a single TOML file with these sections:

### `[run]` — Run Metadata

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `method` | string | **required** | Training method. One of: `"sft"`, `"grpo"`, `"unsloth_grpo"` |
| `pod_id` | string | `null` | ID of an existing pod to use. Omit to create a new pod from the `[pod]` section |
| `terminate_on_complete` | bool | `false` | Delete the pod after training completes successfully. Saves money by auto-cleaning up |
| `terminate_after_hub_push` | bool | `false` | Delete the pod after a successful training run that pushes the model to HuggingFace Hub. Requires `output.push_to_hub = true` |
| `download_on_complete` | string | `null` | Local directory to download the trained model to after training. Downloads from `[output].save_path`. `null` = skip download |

### `[pod]` — Pod Provisioning

Only used when `run.pod_id` is not set. Defines the GPU pod to create.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | string | `"runpodx-training"` | Name for the new pod |
| `image` | string | `"runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"` | Docker image to run on the pod |
| `cloud_type` | string | `"SECURE"` | Cloud tier. `"SECURE"` (dedicated) or `"COMMUNITY"` (shared, cheaper) |
| `container_disk_gb` | int | `50` | Ephemeral container disk size in GB. Destroyed when the pod is deleted |
| `volume_gb` | int | `50` | Persistent volume size in GB. Survives pod restarts |
| `volume_mount_path` | string | `"/workspace"` | Where the persistent volume is mounted inside the container |

### `[pod.gpu]` — GPU Hardware

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `type_ids` | list[string] | `null` (any available) | Ordered list of acceptable GPU models. Examples: `"NVIDIA A100-SXM4-80GB"`, `"NVIDIA GeForce RTX 4090"`, `"NVIDIA H100 80GB HBM3"`. First available is picked |
| `count` | int | `1` | Number of GPUs to allocate |
| `type_priority` | string | `"availability"` | `"availability"` picks the first available GPU from `type_ids`. `"custom"` respects the exact order you listed |
| `min_ram_per_gpu` | int | `8` | Minimum system RAM in GB per GPU |
| `min_vcpu_per_gpu` | int | `2` | Minimum virtual CPUs per GPU |
| `allowed_cuda_versions` | list[string] | `null` (any) | Restrict to specific CUDA versions, e.g. `["12.0", "12.4"]` |

### `[model]` — Model & Adapter

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | string | **required** | HuggingFace model ID, e.g. `"Qwen/Qwen3-8B"`, `"meta-llama/Llama-3-8B"` |
| `adapter` | string | `"lora"` | Fine-tuning strategy. `"lora"` for parameter-efficient, `"full"` for full fine-tuning |
| `adapter_path` | string | `null` | Path to existing LoRA adapter weights to continue training from. Can be a HuggingFace ID (e.g. `"user/my-adapter"`) or a local path |
| `resume_from_checkpoint` | string | `null` | Path to a training checkpoint on the pod to resume from. Use `"latest"` to auto-detect the most recent checkpoint in the output directory |
| `lora_r` | int | `16` | LoRA rank. Higher = more parameters = more capacity but slower. Common values: 8, 16, 32, 64 |
| `lora_alpha` | int | `32` | LoRA alpha (scaling factor). Typically set to 2x `lora_r` |
| `lora_dropout` | float | `0.05` | Dropout probability for LoRA layers. Set to `0.0` to disable |
| `lora_target_modules` | list[string] | `null` (auto-detect) | Which model layers to apply LoRA to. `null` lets PEFT auto-detect. Example: `["q_proj", "v_proj", "k_proj", "o_proj"]` |

### `[dataset]` — Dataset Source & Format

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `source` | string | **required** | Where the dataset comes from. `"huggingface"` loads directly on the pod (no upload needed). `"local"` normalizes a local file and uploads it |
| `path` | string | **required** | HuggingFace dataset ID (e.g. `"tatsu-lab/alpaca"`) or local file path (e.g. `"./data.jsonl"`) |
| `split` | string | `"train"` | Dataset split to use |
| `format` | string | `"alpaca"` | Input data format. Determines how records are converted for training. See [Dataset Formats](#dataset-formats) below |
| `max_samples` | int | `null` (all) | Limit the number of training samples. Useful for testing |
| `validation_ratio` | float | `null` | Fraction of samples reserved for validation (0.0–1.0 exclusive). `null` disables eval |
| `validation_seed` | int | `42` | Random seed used when splitting train/validation |
| `field_mapping` | dict | `null` | Field mapping for `format = "custom"`. Maps your field names to `instruction`, `input`, `output` |

### `[training]` — Hyperparameters

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `epochs` | int | `3` | Number of full passes over the dataset |
| `lr` | float | `2e-5` | Learning rate. Full fine-tuning typically uses lower (1e-6 to 5e-6), LoRA uses higher (1e-5 to 2e-4) |
| `batch_size` | int | `4` | Per-device training batch size. Lower this if you hit OOM errors |
| `gradient_accumulation` | int | `8` | Number of steps to accumulate gradients before updating. Effective batch = `batch_size * gradient_accumulation` |
| `max_seq_length` | int | `2048` | Maximum token sequence length. Longer sequences use more VRAM. Must fit in the model's context window |
| `warmup_ratio` | float | `0.03` | Fraction of total steps used for learning rate warmup (0.0 to 1.0) |
| `weight_decay` | float | `0.01` | L2 regularization. Helps prevent overfitting |
| `bf16` | bool | `true` | Use bfloat16 mixed precision. Recommended for A100/H100. Mutually exclusive with `fp16` |
| `fp16` | bool | `false` | Use float16 mixed precision. Use if GPU doesn't support bf16 |
| `optimizer` | string | `"adamw_8bit"` | Optimizer. Options: `"adamw_8bit"` (memory efficient), `"adamw_torch"`, `"sgd"`, `"adafactor"` |
| `lr_scheduler_type` | string | `"cosine"` | Learning rate schedule. Options: `"cosine"`, `"linear"`, `"constant"`, `"constant_with_warmup"` |
| `logging_steps` | int | `10` | Log training metrics (loss, lr, etc.) every N steps |
| `save_steps` | int | `500` | Save a checkpoint every N steps |
| `save_total_limit` | int | `3` | Maximum number of checkpoints to keep. Oldest are deleted first |
| `append_eos_token` | bool | `true` | Append EOS token to each assistant/response target during SFT |
| `convert_to_prompt_completion` | bool | `false` | Convert dataset into prompt/completion pairs and train only on completion tokens (when supported by TRL) |
| `sanitize_ascii_outputs` | bool | `false` | Trim pathological repeated tail lines in ASCII-art targets |
| `ascii_max_tail_line_repeats` | int | `null` | Max repeated copies allowed for the final output line. `null` = no limit |
| `generation_eval_enabled` | bool | `false` | Run a prompt-based generation benchmark after training |
| `generation_eval_max_new_tokens` | int | `384` | Max new tokens to generate for benchmark prompts |
| `generation_eval_prompts` | list[string] | `null` | Custom prompt list for the post-training generation benchmark. `null` = use defaults |

#### GRPO-Specific Options

These options only apply when `run.method = "grpo"` or `"unsloth_grpo"`.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `grpo_profile` | string | `null` | **Required for GRPO.** Named profile or import path that supplies reward functions and dataset prep. Example: `"ascii_art"` |
| `grpo_num_generations` | int | `4` | Number of sampled completions per prompt for GRPO |
| `grpo_max_completion_length` | int | `192` | Maximum generated completion length for GRPO rollouts |
| `grpo_soft_overlong_cache` | int | `32` | Token window near the completion cap where a soft overlong penalty starts |
| `grpo_beta` | float | `0.0` | KL regularization coefficient for GRPO. `0.0` = no KL penalty |
| `grpo_temperature` | float | `1.0` | Sampling temperature for GRPO rollouts |
| `grpo_top_p` | float | `1.0` | Top-p (nucleus) sampling value for GRPO rollouts |
| `grpo_reward_weights` | list[float] | `null` | Optional reward weights for the GRPO reward functions. Length must match the number of reward functions in the profile |

### `[output]` — Model Output

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `save_path` | string | `"/workspace/output"` | Directory on the pod where the trained model and checkpoints are saved |
| `push_to_hub` | bool | `false` | Push the final model to HuggingFace Hub after training completes |
| `hub_model_id` | string | `""` | HuggingFace Hub repository ID, e.g. `"your-username/model-name"`. Required if `push_to_hub = true` |
| `hub_token` | string | `null` | HuggingFace API token. Falls back to `HF_TOKEN` environment variable |

### `[monitoring]` — Weights & Biases

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `wandb_project` | string | `""` | W&B project name. Leave empty to disable W&B logging |
| `wandb_run_name` | string | `""` | Name for this W&B run. Auto-generated if empty |
| `wandb_entity` | string | `null` | W&B team/entity name. Uses your default entity if not set |
| `log_interval` | int | `10` | Log metrics every N steps |

### `[ssh]` — SSH Connection

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `key_path` | string | `null` (`~/.ssh/id_ed25519`) | Path to your SSH private key. The matching public key is automatically injected into newly created pods |
| `port` | int | `22` | SSH port (usually auto-mapped by RunPod) |
| `user` | string | `"root"` | SSH username |
| `connect_timeout` | int | `30` | Seconds to wait for SSH connection |

---

## Dataset Formats

| Format | Expected Fields | Description |
|--------|----------------|-------------|
| `alpaca` | `instruction`, `input` (optional), `output` | Standard instruction-following format. Most common for SFT |
| `sharegpt` | `conversations` or `messages` (array of `{role, content}`) | Multi-turn chat format. Extracts the first human-assistant exchange |
| `completion` | `text` | Raw text for continued pre-training |
| `custom` | User-defined via `field_mapping` | Map your field names to the standard format |

### Custom Format Example

If your dataset has fields `question`, `context`, `answer`:

```toml
[dataset]
source = "local"
path = "./my_data.jsonl"
format = "custom"
field_mapping = { instruction = "question", input = "context", output = "answer" }
```

---

## Pod Configuration Reference

For `runpodx pods create`, use a standalone pod config TOML:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | string | `"my pod"` | Pod name (max 191 characters) |
| `image` | string | **required** | Docker image to run |
| `template_id` | string | `null` | Use a pre-configured template instead of manual config |
| `compute_type` | string | `"GPU"` | `"GPU"` or `"CPU"` |
| `cloud_type` | string | `"SECURE"` | `"SECURE"` (dedicated) or `"COMMUNITY"` (shared, cheaper) |
| `interruptible` | bool | `false` | Use spot/interruptible pricing (cheaper, but pod may be preempted) |
| `locked` | bool | `false` | Lock pod to prevent accidental stop/reset |
| `docker_entrypoint` | list[string] | `[]` | Override container ENTRYPOINT |
| `docker_start_cmd` | list[string] | `[]` | Override container CMD / start command |
| `container_registry_auth_id` | string | `null` | Registry credentials ID for private Docker images |

### `[gpu]` — GPU Hardware

Same as `[pod.gpu]` in the training config.

### `[storage]` — Disk & Volumes

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `container_disk_gb` | int | `50` | Ephemeral container disk in GB |
| `volume_gb` | int | `20` | Persistent pod volume in GB |
| `volume_mount_path` | string | `"/workspace"` | Filesystem path where volumes are mounted |
| `network_volume_id` | string | `null` | Attach an existing network volume by its ID |

### `[networking]` — Network & Ports

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `ports` | list[string] | `["8888/http", "22/tcp"]` | Exposed ports, e.g. `["8888/http", "22/tcp", "6006/http"]` |
| `support_public_ip` | bool | `null` | Require a public IP (Community Cloud). **Must be `true` for SSH to work via REST API** |
| `data_center_ids` | list[string] | `null` | Preferred data center IDs, e.g. `["US-TX-3", "EU-RO-1"]` |
| `data_center_priority` | string | `"availability"` | `"availability"` or `"custom"` (respect exact order of `data_center_ids`) |
| `country_codes` | list[string] | `null` | Restrict deployment to specific countries |

### `[env]` — Environment Variables

```toml
[env]
MY_VAR = "value"
CUDA_VISIBLE_DEVICES = "0,1"
```

---

## Endpoint Configuration Reference

For creating serverless endpoints via the SDK.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | string | **required** | Endpoint name |
| `template_id` | string | **required** | Template ID to use |
| `gpu_ids` | string | **required** | GPU type ID string |
| `workers_min` | int | `0` | Minimum active workers |
| `workers_max` | int | `3` | Maximum active workers |
| `idle_timeout` | int | `5` | Seconds before idle workers scale down |
| `flash_boot` | bool | `true` | Enable flash boot for faster cold starts |
| `scaling_type` | string | `null` | Scaling strategy (e.g. `"QUEUE_DELAY"`) |

## Network Volume Configuration Reference

For `runpodx volumes create`:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | string | **required** | Volume name |
| `size_gb` | int | **required** | Volume size in GB (1–4000) |
| `data_center_id` | string | **required** | Data center ID for the volume (e.g. `"US-TX-3"`) |

---

## CLI Reference

### Pods

```bash
runpodx pods list                    # List all pods
runpodx pods create config.toml     # Create from TOML
runpodx pods get <pod_id>           # Get pod details
runpodx pods stop <pod_id>          # Stop a running pod
runpodx pods start <pod_id>         # Start a stopped pod
runpodx pods restart <pod_id>       # Restart a pod
runpodx pods delete <pod_id>        # Delete (with confirmation)
runpodx pods delete <pod_id> --force  # Delete without confirmation
```

### Templates

```bash
runpodx templates list                # List your templates
runpodx templates list --public       # List public templates
runpodx templates create config.toml  # Create from TOML
```

### Network Volumes

```bash
runpodx volumes list                  # List volumes
runpodx volumes create config.toml   # Create from TOML
runpodx volumes delete <id>          # Delete (with confirmation)
runpodx volumes delete <id> --force  # Delete without confirmation
```

### Training

```bash
runpodx run config.toml              # Run training pipeline
runpodx run config.toml --dry-run    # Preview without executing
runpodx run config.toml --pod ID     # Use existing pod
```

### Download

Downloads files from a running pod via scp. Automatically resolves the pod's SSH host and port from the API.

```bash
runpodx download <pod_id>                                  # Download /workspace/output → ./checkpoints
runpodx download <pod_id> --remote /workspace/output       # Custom remote path
runpodx download <pod_id> --local ./my-model               # Custom local directory
runpodx download <pod_id> --ssh-key ~/.ssh/id_ed25519      # Custom SSH key
runpodx download <pod_id> --port 13097                     # Explicit SSH port (normally auto-detected)
```

---

## Development

```bash
git clone https://github.com/your-username/runpodx.git
cd runpodx
uv sync

# Run tests (68 tests)
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/
```

## Requirements

- Python 3.12+
- SSH key pair (for training orchestration)
- RunPod API key
