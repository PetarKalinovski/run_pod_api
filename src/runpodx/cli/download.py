from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from runpodx import RunPodClient
from runpodx.training.executor import SSHConnection, SSHExecutor

console = Console()


def download_checkpoint(
    pod_id: str = typer.Argument(help="RunPod pod ID to download from."),
    remote_path: str = typer.Option(
        "/workspace/output",
        "--remote",
        "-r",
        help="Remote path on the pod to download.",
    ),
    local_path: str = typer.Option(
        "./checkpoints",
        "--local",
        "-l",
        help="Local directory to save downloaded files.",
    ),
    ssh_key: Optional[str] = typer.Option(
        None, "--ssh-key", "-k", help="Path to SSH private key."
    ),
    ssh_port: Optional[int] = typer.Option(
        None, "--port", "-p", help="SSH port (auto-detected from pod if omitted)."
    ),
):
    """Download training checkpoints from a RunPod pod.

    Resolves the pod's SSH endpoint from the RunPod API, then uses
    scp to pull model weights, adapter files, and checkpoints.

    Examples:
        runpodx download pod_abc123
        runpodx download pod_abc123 --remote /workspace/output --local ./my-model
        runpodx download pod_abc123 --port 13097 --local ./ckpt
    """
    local = Path(local_path)
    local.mkdir(parents=True, exist_ok=True)

    with RunPodClient() as client:
        # Resolve SSH host + port from the pod's API data
        pod = client.pods.get(pod_id)
        if pod.data.is_stopped:
            console.print("[bold red]Pod is stopped.[/] Start it first with: "
                          f"runpodx pods start {pod_id}")
            raise typer.Exit(code=1)

        data = pod.data.model_dump(by_alias=True)
        public_ip = data.get("publicIp")
        port_mappings = data.get("portMappings") or {}
        ssh_public_port = port_mappings.get("22")

        if ssh_port:
            # User explicitly provided port — use publicIp as host
            resolved_port = ssh_port
        elif ssh_public_port:
            resolved_port = int(ssh_public_port)
        else:
            # Wait a bit for port mapping to appear
            console.print("  Waiting for SSH port mapping...")
            for _ in range(10):
                time.sleep(5)
                pod.refresh()
                data = pod.data.model_dump(by_alias=True)
                public_ip = data.get("publicIp")
                port_mappings = data.get("portMappings") or {}
                ssh_public_port = port_mappings.get("22")
                if ssh_public_port:
                    resolved_port = int(ssh_public_port)
                    break
            else:
                console.print("[bold red]Could not resolve SSH port.[/] "
                              "Try passing --port manually.")
                raise typer.Exit(code=1)

        if not public_ip:
            console.print("[bold red]Pod has no public IP.[/] "
                          "Was it created with support_public_ip?")
            raise typer.Exit(code=1)

        console.print(f"  [green]SSH endpoint:[/] {public_ip}:{resolved_port}")

    key_path = ssh_key or "~/.ssh/id_ed25519"
    key_passphrase = os.environ.get("RUNPOD_PRIVATE_KEY")

    conn = SSHConnection(
        host=public_ip,
        port=resolved_port,
        key_path=key_path,
        key_passphrase=key_passphrase,
    )
    executor = SSHExecutor(conn)

    try:
        console.print(f"[bold blue]Connecting to pod[/] {pod_id}...")
        if not executor.test_connection():
            console.print("[bold red]SSH connection failed.[/] Is the pod running?")
            raise typer.Exit(code=1)

        # List what's available at the remote path
        console.print(f"[bold blue]Checking remote path:[/] {remote_path}")
        result = executor.run(f"ls -lh {remote_path}", stream=False)
        if result.returncode != 0:
            console.print(f"[bold red]Remote path not found:[/] {remote_path}")
            console.print(f"  {result.stderr.strip()}")
            raise typer.Exit(code=1)

        console.print(f"[dim]{result.stdout.strip()}[/dim]")

        # List all files to download
        remote_path = remote_path.rstrip("/")
        find_result = executor.run(
            f"find {remote_path} -type f 2>/dev/null", stream=False
        )
        if find_result.returncode != 0 or not find_result.stdout.strip():
            console.print("[bold red]No files found at remote path.[/]")
            raise typer.Exit(code=1)

        files = [f.strip() for f in find_result.stdout.strip().split("\n") if f.strip()]
        console.print(f"\n[bold blue]Downloading[/] {len(files)} file(s) → {local.resolve()}")

        for remote_file in files:
            relative = remote_file[len(remote_path):].lstrip("/")
            local_file = local / relative
            local_file.parent.mkdir(parents=True, exist_ok=True)

            console.print(f"  [dim]{relative}[/dim]")
            dl_result = executor.download(remote_file, str(local_file))
            if dl_result.returncode != 0:
                console.print(f"  [red]Failed: {dl_result.stderr.strip()}[/red]")

        console.print(f"\n[bold green]Download complete![/] Files saved to {local.resolve()}")

    finally:
        conn.cleanup()
