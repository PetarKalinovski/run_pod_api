import typer
from rich.console import Console
from rich.table import Table

from runpodx import RunPodClient

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("list")
def list_volumes():
    """List all network volumes."""
    with RunPodClient() as client:
        volumes = client.network_volumes.list()
    if not volumes:
        console.print("[dim]No network volumes found.[/dim]")
        return
    table = Table(title="Network Volumes")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Size (GB)", justify="right")
    table.add_column("Data Center")
    for v in volumes:
        table.add_row(v.id, v.name, str(v.size), v.data_center_id)
    console.print(table)


@app.command("create")
def create_volume(config: str = typer.Argument(help="Path to a TOML config file.")):
    """Create a network volume from a TOML config file."""
    with RunPodClient() as client:
        volume = client.network_volumes.create(config)
    console.print(f"[green]Created volume:[/green] {volume.id} ({volume.name})")


@app.command("delete")
def delete_volume(
    volume_id: str = typer.Argument(help="Volume ID."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation."),
):
    """Delete a network volume."""
    if not force:
        typer.confirm(f"Delete volume {volume_id}? This cannot be undone.", abort=True)
    with RunPodClient() as client:
        client.network_volumes.delete(volume_id)
    console.print(f"[red]Deleted:[/red] {volume_id}")
