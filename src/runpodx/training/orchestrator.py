from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel

from runpodx.configs.loader import resolve_config
from runpodx.configs.pod import NetworkingConfig, PodCreateConfig, StorageConfig
from runpodx.training.configs import RunConfig
from runpodx.training.executor import SSHConnection, SSHExecutor
from runpodx.training.methods import get_method
from runpodx.training.normalizers import get_normalizer

if TYPE_CHECKING:
    from runpodx.resources.pods import Pod, PodManager

logger = logging.getLogger("runpodx.training")

REMOTE_WORK_DIR = "/workspace/runpodx_run"


class TrainingOrchestrator:
    """Orchestrates a full training run on a RunPod GPU pod.

    Pipeline:
    1. Load + validate RunConfig
    2. Provision pod OR connect to existing
    3. Wait for pod ready
    4. Establish SSH connection
    5. Upload training script + requirements
    6. Normalize + upload dataset
    7. Install pip dependencies
    8. Run training (stream logs)
    """

    def __init__(
        self,
        pod_manager: PodManager,
        console: Console | None = None,
    ) -> None:
        self._pods = pod_manager
        self._console = console or Console()
        self._pod: Pod | None = None
        self._executor: SSHExecutor | None = None
        self._created_pod: bool = False

    def run(
        self,
        config_source: str | Path | dict | RunConfig,
        *,
        pod_id_override: str | None = None,
        dry_run: bool = False,
    ) -> None:
        """Execute the full training pipeline."""
        config = resolve_config(RunConfig, config_source)
        if pod_id_override:
            config.run.pod_id = pod_id_override

        method = get_method(config.run.method)

        if dry_run:
            self._print_dry_run(config, method)
            return

        try:
            # Step 1: Pod
            self._console.print("[bold blue]Step 1/6:[/] Provisioning pod...")
            pod = self._get_or_create_pod(config)
            self._pod = pod

            # Step 2: Wait
            self._console.print("[bold blue]Step 2/6:[/] Waiting for pod to be ready...")
            pod.wait_until_ready(timeout=300)
            self._console.print(f"  [green]Pod ready:[/] {pod.id} at {pod.data.public_ip}")

            # Step 3: SSH
            self._console.print("[bold blue]Step 3/6:[/] Establishing SSH connection...")
            ssh_host, ssh_port = self._wait_for_ssh_port(pod)
            key_path = config.ssh.key_path or "~/.ssh/id_ed25519"
            key_passphrase = os.environ.get("RUNPOD_PRIVATE_KEY")
            conn = SSHConnection(
                host=ssh_host,
                port=ssh_port,
                key_path=key_path,
                key_passphrase=key_passphrase,
                connect_timeout=config.ssh.connect_timeout,
            )
            self._executor = SSHExecutor(conn)
            self._wait_for_ssh()
            self._executor.run(f"mkdir -p {REMOTE_WORK_DIR}/data", stream=False)

            # Step 4: Upload scripts
            self._console.print("[bold blue]Step 4/6:[/] Uploading training scripts...")
            self._upload_scripts(method, config)

            # Step 5: Dataset
            self._console.print("[bold blue]Step 5/6:[/] Preparing dataset...")
            self._prepare_and_upload_dataset(config, method)

            # Install deps
            self._console.print("  Installing pip dependencies...")
            self._install_dependencies()

            # Upload remote config
            remote_config = method.generate_remote_config(config)
            self._upload_json_config(remote_config)

            # Step 6: Train
            self._console.print("[bold blue]Step 6/6:[/] Starting training...")
            self._run_training(config)

            self._console.print("[bold green]Training complete![/]")

            # Post-training: download checkpoints
            if config.run.download_on_complete:
                self._console.print(
                    f"[bold blue]Downloading output to {config.run.download_on_complete}...[/]"
                )
                self._download_output(config)

            # Post-training: terminate pod
            if self._should_terminate_pod(config):
                terminate_reason = "successful completion"
                if config.run.terminate_after_hub_push:
                    terminate_reason = "successful Hub push"
                self._console.print(
                    f"[bold yellow]Terminating pod {pod.id} after {terminate_reason}...[/]"
                )
                self._pods.delete(pod.id)
                self._console.print(f"  [red]Pod {pod.id} deleted.[/]")

        except Exception:
            self._console.print("[bold red]Training failed.[/]")
            raise
        finally:
            if self._executor:
                self._executor._conn.cleanup()

    def _get_or_create_pod(self, config: RunConfig) -> Pod:
        if config.run.pod_id:
            pod = self._pods.get(config.run.pod_id)
            if pod.data.is_stopped:
                self._console.print(f"  Starting stopped pod {pod.id}...")
                pod.start()
            return pod

        spec = config.pod
        pod_config = PodCreateConfig(
            name=spec.name,
            image=spec.image,
            cloud_type=spec.cloud_type,
            gpu=spec.gpu,
            storage=StorageConfig(
                container_disk_gb=spec.container_disk_gb,
                volume_gb=spec.volume_gb,
                volume_mount_path=spec.volume_mount_path,
            ),
            networking=NetworkingConfig(ports=["8888/http", "22/tcp"], support_public_ip=True),
            env=self._build_env_vars(config),
        )
        pod = self._pods.create(pod_config)
        self._created_pod = True
        self._console.print(f"  [green]Created pod:[/] {pod.id}")
        return pod

    def _should_terminate_pod(self, config: RunConfig) -> bool:
        if config.run.terminate_after_hub_push:
            return config.output.push_to_hub
        return config.run.terminate_on_complete

    def _build_env_vars(self, config: RunConfig) -> dict[str, str]:
        env: dict[str, str] = {}
        if config.monitoring.wandb_project:
            env["WANDB_PROJECT"] = config.monitoring.wandb_project

        # Inject SSH public key so the RunPod image sets up authorized_keys.
        # The dashboard does this automatically; the REST API does not.
        key_path = config.ssh.key_path or "~/.ssh/id_ed25519"
        pub_key_path = Path(key_path).expanduser().with_suffix(".pub")
        if pub_key_path.exists():
            env["PUBLIC_KEY"] = pub_key_path.read_text().strip()

        return env

    def _wait_for_ssh_port(
        self, pod: Pod, max_attempts: int = 20, interval: int = 10
    ) -> tuple[str, int]:
        """Poll until the pod has a public IP and SSH port mapping.

        The REST API returns port mappings in ``portMappings`` (e.g.
        ``{"22": 13607}``) and the host IP in ``publicIp``.
        """
        for attempt in range(max_attempts):
            pod.refresh()
            data = pod.data.model_dump(by_alias=True)
            public_ip = data.get("publicIp")
            port_mappings = data.get("portMappings") or {}
            ssh_public_port = port_mappings.get("22")
            if public_ip and ssh_public_port:
                self._console.print(
                    f"  [green]SSH endpoint:[/] {public_ip}:{ssh_public_port}"
                )
                return public_ip, int(ssh_public_port)
            self._console.print(
                f"  Waiting for SSH port mapping ({attempt + 1}/{max_attempts})..."
            )
            time.sleep(interval)
        raise RuntimeError(
            "SSH port mapping not available after max retries."
        )

    def _wait_for_ssh(self, max_attempts: int = 12, interval: int = 10) -> None:
        for attempt in range(max_attempts):
            if self._executor.test_connection():
                return
            self._console.print(
                f"  SSH not ready, retrying ({attempt + 1}/{max_attempts})..."
            )
            time.sleep(interval)
        raise RuntimeError("SSH connection failed after max retries.")

    def _upload_scripts(self, method, config: RunConfig) -> None:
        script_path = method.script_path(config)
        result = self._executor.upload(script_path, f"{REMOTE_WORK_DIR}/train.py")
        if result.returncode != 0:
            raise RuntimeError(f"Script upload failed: {result.stderr}")

        result = self._executor.upload(
            method.requirements_path(config), f"{REMOTE_WORK_DIR}/requirements.txt"
        )
        if result.returncode != 0:
            raise RuntimeError(f"Requirements upload failed: {result.stderr}")

        for support_file in script_path.parent.glob("*.py"):
            if support_file == script_path:
                continue
            result = self._executor.upload(
                support_file, f"{REMOTE_WORK_DIR}/{support_file.name}"
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Support file upload failed ({support_file.name}): {result.stderr}"
                )

    def _prepare_and_upload_dataset(self, config: RunConfig, method) -> None:
        dataset_config = config.dataset

        if dataset_config.source == "huggingface":
            self._console.print("  Dataset will be loaded from HuggingFace on the pod.")
            return

        # Local dataset: normalize and upload
        normalizer_kwargs = {}
        if dataset_config.format == "custom":
            normalizer_kwargs["field_mapping"] = dataset_config.field_mapping

        normalizer = get_normalizer(
            method.canonical_format(), dataset_config.format, **normalizer_kwargs
        )

        records = self._load_local_dataset(dataset_config.path)
        if dataset_config.max_samples:
            records = records[: dataset_config.max_samples]

        normalized = normalizer.normalize(records)
        self._console.print(f"  Normalized {len(normalized)} samples.")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in normalized:
                f.write(json.dumps(record) + "\n")
            tmp_path = f.name

        result = self._executor.upload(tmp_path, f"{REMOTE_WORK_DIR}/data/dataset.jsonl")
        Path(tmp_path).unlink()
        if result.returncode != 0:
            raise RuntimeError(f"Dataset upload failed: {result.stderr}")

    def _load_local_dataset(self, path: str) -> list[dict]:
        p = Path(path)
        if p.suffix == ".jsonl":
            records = []
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records
        elif p.suffix == ".csv":
            import csv

            with open(p) as f:
                reader = csv.DictReader(f)
                return list(reader)
        else:
            raise ValueError(f"Unsupported dataset format: {p.suffix}. Use .jsonl or .csv")

    def _upload_json_config(self, remote_config: dict) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(remote_config, f, indent=2)
            tmp_path = f.name

        result = self._executor.upload(tmp_path, f"{REMOTE_WORK_DIR}/config.json")
        Path(tmp_path).unlink()
        if result.returncode != 0:
            raise RuntimeError(f"Config upload failed: {result.stderr}")

    def _install_dependencies(self) -> None:
        proc = self._executor.run(
            f"pip install --upgrade -r {REMOTE_WORK_DIR}/requirements.txt",
            stream=True,
            on_stdout=lambda line: self._console.print(f"  [dim]{line}[/dim]"),
            on_stderr=lambda line: self._console.print(f"  [dim]{line}[/dim]"),
        )
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("Dependency installation failed.")

    def _export_env_vars(self, config: RunConfig) -> None:
        """Forward local API keys to the pod as env vars."""
        env_map = {
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY"),
            "HF_TOKEN": config.output.hub_token or os.environ.get("HF_TOKEN"),
        }
        exports = []
        for key, val in env_map.items():
            if val:
                exports.append(f"echo 'export {key}={val}' >> ~/.bashrc")
                exports.append(f"export {key}={val}")
        if exports:
            cmd = " && ".join(exports)
            self._executor.run(cmd, stream=False)
            self._console.print("  [green]API keys forwarded to pod.[/]")

    def _run_training(self, config: RunConfig) -> None:
        env_prefix = ""
        wandb_key = os.environ.get("WANDB_API_KEY")
        hf_token = config.output.hub_token or os.environ.get("HF_TOKEN")
        env_parts = []
        if wandb_key:
            env_parts.append(f"WANDB_API_KEY={wandb_key}")
        if hf_token:
            env_parts.append(f"HF_TOKEN={hf_token}")
        if env_parts:
            env_prefix = " ".join(env_parts) + " "

        cmd = (
            f"cd {REMOTE_WORK_DIR} && "
            f"{env_prefix}python train.py --config {REMOTE_WORK_DIR}/config.json"
        )
        proc = self._executor.run(
            cmd,
            stream=True,
            on_stdout=lambda line: self._console.print(f"  {line}"),
            on_stderr=lambda line: self._console.print(f"  [dim]{line}[/dim]"),
        )
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Training exited with code {proc.returncode}")

    def _download_output(self, config: RunConfig) -> None:
        """Download training output from the pod to a local directory."""
        remote_path = config.output.save_path.rstrip("/")
        local_dir = Path(config.run.download_on_complete).expanduser()
        local_dir.mkdir(parents=True, exist_ok=True)

        # List remote files first
        result = self._executor.run(
            f"find {remote_path} -type f 2>/dev/null | head -100", stream=False
        )
        if result.returncode != 0 or not result.stdout.strip():
            self._console.print("  [yellow]No output files found to download.[/]")
            return

        files = result.stdout.strip().split("\n")
        self._console.print(f"  Found {len(files)} file(s) to download.")

        for remote_file in files:
            remote_file = remote_file.strip()
            if not remote_file:
                continue
            # Preserve directory structure relative to save_path
            relative = remote_file[len(remote_path) :].lstrip("/")
            local_file = local_dir / relative
            local_file.parent.mkdir(parents=True, exist_ok=True)

            self._console.print(f"  [dim]Downloading {relative}...[/dim]")
            dl_result = self._executor.download(remote_file, str(local_file))
            if dl_result.returncode != 0:
                self._console.print(
                    f"  [red]Failed to download {relative}: {dl_result.stderr}[/red]"
                )

        self._console.print(f"  [green]Output downloaded to {local_dir}[/green]")

    def _print_dry_run(self, config: RunConfig, method) -> None:
        self._console.print(Panel("[bold]Dry Run[/bold]", style="yellow"))
        self._console.print(f"  Method:    {config.run.method}")
        self._console.print(f"  Model:     {config.model.name}")
        self._console.print(f"  Adapter:   {config.model.adapter}")
        self._console.print(f"  Dataset:   {config.dataset.source}:{config.dataset.path}")
        self._console.print(f"  Format:    {config.dataset.format}")
        self._console.print(f"  Epochs:    {config.training.epochs}")
        self._console.print(f"  Batch:     {config.training.batch_size}")
        self._console.print(f"  LR:        {config.training.lr}")
        self._console.print(f"  Script:    {method.script_path(config)}")
        if config.run.pod_id:
            self._console.print(f"  Pod:       {config.run.pod_id} (existing)")
        else:
            gpu_ids = config.pod.gpu.type_ids or ["auto"]
            self._console.print(f"  Pod:       NEW ({gpu_ids})")
        self._console.print(
            f"  W&B:       {config.monitoring.wandb_project or 'disabled'}"
        )
        self._console.print(
            f"  Download:  {config.run.download_on_complete or 'disabled'}"
        )
        self._console.print(
            f"  Terminate: {'yes' if config.run.terminate_on_complete else 'no'}"
        )
        self._console.print(
            "  Hub cleanup: "
            f"{'yes' if config.run.terminate_after_hub_push else 'no'}"
        )
