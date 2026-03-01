import typer
from dotenv import load_dotenv

load_dotenv()

from runpodx.cli.download import download_checkpoint
from runpodx.cli.network_volumes import app as volumes_app
from runpodx.cli.pods import app as pods_app
from runpodx.cli.run import run_training
from runpodx.cli.templates import app as templates_app

app = typer.Typer(
    name="runpodx",
    help="Config-driven CLI for RunPod GPU cloud.",
    no_args_is_help=True,
)

app.command("run")(run_training)
app.command("download")(download_checkpoint)
app.add_typer(pods_app, name="pods", help="Manage GPU pods.")
app.add_typer(templates_app, name="templates", help="Manage templates.")
app.add_typer(volumes_app, name="volumes", help="Manage network volumes.")

if __name__ == "__main__":
    app()
