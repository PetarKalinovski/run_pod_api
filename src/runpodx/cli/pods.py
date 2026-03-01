import typer
from rich.console import Console
from rich.table import Table

from runpodx import RunPodClient

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("list")
def list_pods():
    """List all pods."""
    with RunPodClient() as client:
        pods = client.pods.list()
    if not pods:
        console.print("[dim]No pods found.[/dim]")
        return
    table = Table(title="Pods")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Status", style="green")
    table.add_column("GPU")
    table.add_column("Cost/hr", justify="right")
    for pod in pods:
        table.add_row(
            pod.id,
            pod.name,
            pod.desired_status,
            pod.gpu_type_id or "N/A",
            f"${pod.cost_per_hr:.4f}" if pod.cost_per_hr else "N/A",
        )
    console.print(table)


@app.command("create")
def create_pod(config: str = typer.Argument(help="Path to a TOML config file.")):
    """Create a pod from a TOML config file."""
    with RunPodClient() as client:
        pod = client.pods.create(config)
    console.print(f"[green]Created pod:[/green] {pod.id} ({pod.name})")


@app.command("get")
def get_pod(pod_id: str = typer.Argument(help="Pod ID.")):
    """Get details for a specific pod."""
    with RunPodClient() as client:
        pod = client.pods.get(pod_id)
    console.print(f"[bold]ID:[/bold]     {pod.id}")
    console.print(f"[bold]Name:[/bold]   {pod.name}")
    console.print(f"[bold]Status:[/bold] {pod.desired_status}")
    console.print(f"[bold]GPU:[/bold]    {pod.gpu_type_id or 'N/A'}")
    console.print(f"[bold]IP:[/bold]     {pod.public_ip or 'pending'}")
    console.print(f"[bold]Cost:[/bold]   ${pod.cost_per_hr:.4f}/hr" if pod.cost_per_hr else "N/A")


@app.command("stop")
def stop_pod(pod_id: str = typer.Argument(help="Pod ID.")):
    """Stop a running pod (preserves data)."""
    with RunPodClient() as client:
        pod = client.pods.get(pod_id)
        pod.stop()
    console.print(f"[yellow]Stopped:[/yellow] {pod_id}")


@app.command("start")
def start_pod(pod_id: str = typer.Argument(help="Pod ID.")):
    """Start a stopped pod."""
    with RunPodClient() as client:
        pod = client.pods.get(pod_id)
        pod.start()
    console.print(f"[green]Started:[/green] {pod_id}")


@app.command("restart")
def restart_pod(pod_id: str = typer.Argument(help="Pod ID.")):
    """Restart a pod."""
    with RunPodClient() as client:
        pod = client.pods.get(pod_id)
        pod.restart()
    console.print(f"[green]Restarted:[/green] {pod_id}")


@app.command("delete")
def delete_pod(
    pod_id: str = typer.Argument(help="Pod ID."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation."),
):
    """Delete a pod permanently."""
    if not force:
        typer.confirm(f"Delete pod {pod_id}? This cannot be undone.", abort=True)
    with RunPodClient() as client:
        client.pods.delete(pod_id)
    console.print(f"[red]Deleted:[/red] {pod_id}")
