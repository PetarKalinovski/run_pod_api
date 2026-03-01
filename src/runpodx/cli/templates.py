import typer
from rich.console import Console
from rich.table import Table

from runpodx import RunPodClient

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("list")
def list_templates(
    include_public: bool = typer.Option(False, "--public", help="Include public templates."),
    include_runpod: bool = typer.Option(False, "--runpod", help="Include RunPod templates."),
):
    """List templates."""
    with RunPodClient() as client:
        templates = client.templates.list(
            include_public=include_public, include_runpod=include_runpod
        )
    if not templates:
        console.print("[dim]No templates found.[/dim]")
        return
    table = Table(title="Templates")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Image")
    table.add_column("Serverless", justify="center")
    for t in templates:
        table.add_row(t.id, t.name, t.image_name, "yes" if t.is_serverless else "no")
    console.print(table)


@app.command("create")
def create_template(config: str = typer.Argument(help="Path to a TOML config file.")):
    """Create a template from a TOML config file."""
    with RunPodClient() as client:
        template = client.templates.create(config)
    console.print(f"[green]Created template:[/green] {template.id} ({template.name})")
