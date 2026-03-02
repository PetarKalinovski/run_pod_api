# CLAUDE.md — runpodx

## Project Overview

runpodx is a config-driven Python client and CLI for RunPod's REST API (`rest.runpod.io/v1`).
Two layers: (1) typed SDK for pod/template/volume management, (2) training orchestrator
that provisions a pod, SSHs in, uploads scripts + data, and runs fine-tuning end-to-end.

## Quick Reference

```
src/runpodx/
├── client.py                    # RunPodClient — top-level entry point
├── cli/                         # Typer CLI (main.py, pods.py, run.py, templates.py, etc.)
├── configs/                     # Pydantic config schemas (pod.py, template.py, loader.py)
├── models/                      # API response models (pod.py, template.py, etc.)
├── resources/                   # Resource classes with actions (pods.py, templates.py, etc.)
├── http/                        # HttpTransport + error hierarchy
└── training/
    ├── orchestrator.py          # TrainingOrchestrator — 6-step pipeline
    ├── executor.py              # SSHConnection + SSHExecutor (subprocess ssh/scp)
    ├── configs.py               # RunConfig + sub-configs for training
    ├── methods/                 # Registry: BaseMethod + SFTMethod
    ├── normalizers/             # Registry: dataset format normalizers
    └── scripts/                 # Uploaded to pod: sft_train.py, requirements.txt
```

## Build & Test

```bash
uv sync                          # Install deps
uv run pytest tests/ -v          # Run all tests (68 tests, ~1s)
uv run ruff check src/ tests/    # Lint
uv run runpodx --help            # CLI
```

## Key Architecture Decisions

- **REST API only** — no GraphQL. REST returns `portMappings` dict + `publicIp` (not `runtime.ports`).
- **SSH via subprocess** — not paramiko. Uses user's SSH keys. Passphrase handled by creating
  a temporary passphrase-free key copy via `ssh-keygen -p`.
- **File transfer via scp** — not rsync (unavailable on Windows).
- **PUBLIC_KEY env var** — must be injected when creating pods via REST API for SSH to work.
  The dashboard does this automatically; the REST API does not.
- **Registry pattern** for training methods + dataset normalizers. Adding GRPO = drop new files.
- **Training scripts are standalone .py** — uploaded to pod. Heavy deps (torch, transformers)
  only installed on pod, never locally.
- **Config resolution** — `resolve_config(ConfigClass, source)` accepts TOML path, dict, or
  existing config instance.

## Environment Variables

- `RUNPOD_API_KEY` — required for all API calls
- `RUNPOD_PRIVATE_KEY` — SSH key passphrase (optional)
- `WANDB_API_KEY` — forwarded to pod for W&B logging
- `HF_TOKEN` — forwarded to pod for HuggingFace Hub

## CLI Commands

```bash
runpodx pods list|create|get|stop|start|restart|delete
runpodx templates list|create
runpodx volumes list|create|delete
runpodx run <config.toml> [--pod POD_ID] [--dry-run]
runpodx download <pod_id> [--remote PATH] [--local PATH]
```

## Training Pipeline (orchestrator.py)

1. Load RunConfig from TOML
2. Create pod or connect to existing (uses `portMappings` + `publicIp` for SSH endpoint)
3. Wait for pod ready + SSH port mapping
4. SSH connect (with retry)
5. Upload training script + requirements + config JSON
6. pip install --upgrade + run training with live log streaming
7. (optional) Download output to local dir (`run.download_on_complete`)
8. (optional) Delete pod (`run.terminate_on_complete`)

## Common Patterns

- Pydantic v2 with `ConfigDict(extra="allow", populate_by_name=True)` for API models
- Field aliases map camelCase API → snake_case Python (e.g., `publicIp` → `public_ip`)
- `@register_method("sft")` / `@register_normalizer("sft", "alpaca")` decorator pattern
- SFT canonical format: `{"instruction", "input", "output"}`
- Errors: RunPodError → AuthenticationError, NotFoundError, ValidationError, etc.

## Testing

- Mock transport in `conftest.py` — no real API calls in tests
- `sample_pod_data` fixture includes `portMappings` for SSH tests
- Training orchestrator tests mock SSHExecutor and SSHConnection

## Important Gotchas

- RunPod REST API returns `runtime: null` — do NOT rely on `runtime.ports`
- `wait_until_ready` checks truthiness of `public_ip` (empty string = not ready)
- `supportPublicIp: true` required in pod creation for TCP SSH to work
- trl library API varies by version — training script handles SFTConfig/TrainingArguments compat
- Windows: no rsync, use scp. SSH encoding must be `utf-8` with `errors="replace"`
- `download` CLI resolves SSH host/port from the API — no `from_pod_id` classmethod
- `terminate_on_complete` only fires on success — failed training leaves the pod alive for debugging
