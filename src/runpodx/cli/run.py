from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from runpodx import RunPodClient
from runpodx.training.orchestrator import TrainingOrchestrator

console = Console()


def run_training(
    config: str = typer.Argument(help="Path to a training run TOML config file."),
    pod: Optional[str] = typer.Option(
        None, "--pod", help="Use an existing pod ID (overrides config)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would happen without executing."
    ),
):
    """Run a training job on a RunPod GPU pod.

    Reads a TOML config file defining the model, dataset, hyperparameters,
    and pod specification. Provisions a pod (or uses an existing one),
    uploads training scripts and data, and runs the training.

    Examples:
        runpodx run training.toml
        runpodx run training.toml --pod pod_abc123
        runpodx run training.toml --dry-run
    """
    with RunPodClient() as client:
        orchestrator = TrainingOrchestrator(
            pod_manager=client.pods,
            console=console,
        )
        orchestrator.run(config, pod_id_override=pod, dry_run=dry_run)
