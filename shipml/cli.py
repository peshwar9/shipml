"""Command-line interface for ShipML."""

import sys
from pathlib import Path

import click
import uvicorn

from shipml import __version__
from shipml.errors import UnsupportedModelError, ModelLoadError
from shipml.loaders import detect_framework, get_loader
from shipml.server import create_app
from shipml.utils import get_model_size_mb


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version and exit")
@click.pass_context
def cli(ctx, version):
    """
    ShipML - Turn ML models into APIs with one command.

    Examples:

      shipml serve model.pkl

      shipml serve model.pt --port 5000

      shipml serve model.h5 --name "sentiment-analyzer"
    """
    if version:
        click.echo(f"ShipML version {__version__}")
        ctx.exit(0)

    # If no command is provided, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.argument("model_file", type=click.Path(exists=True))
@click.option("--port", default=8000, type=int, help="Port to run server on (default: 8000)")
@click.option(
    "--host", default="127.0.0.1", type=str, help="Host to bind to (default: 127.0.0.1)"
)
@click.option("--name", default=None, type=str, help="Model name for display")
@click.option("--reload", is_flag=True, help="Enable auto-reload on file changes (dev mode)")
def serve(model_file: str, port: int, host: str, name: str, reload: bool):
    """
    Start API server for your ML model.

    MODEL_FILE can be a path to a trained model file (.pkl, .pt, .h5, etc.)

    Examples:

      shipml serve model.pkl

      shipml serve fraud_detector.pkl --port 5000

      shipml serve model.h5 --name "sentiment-analyzer"

      shipml serve model.pt --reload  # Development mode
    """
    model_path = Path(model_file)
    model_name = name or model_path.stem

    # Display header
    click.echo()
    click.secho("🚀 ShipML", fg="blue", bold=True)
    click.echo(f"   Loading model: {model_path.name}")
    click.echo()

    try:
        # Step 1: Detect framework
        click.echo("🔍 Detecting framework...", nl=False)
        framework = detect_framework(model_path)
        click.secho(f" ✓ {framework}", fg="green")

        # Step 2: Load model
        click.echo("📦 Loading model...", nl=False)
        loader = get_loader(framework)
        model = loader.load(model_path)
        click.secho(" ✓ Success", fg="green")

        # Step 3: Get metadata
        metadata = loader.get_metadata(model)
        model_size = get_model_size_mb(model_path)

        # Display model info
        click.echo()
        click.secho("📊 Model Information:", fg="cyan", bold=True)
        click.echo(f"   Name:       {model_name}")
        click.echo(f"   Type:       {metadata.get('model_type', 'Unknown')}")
        click.echo(f"   Framework:  {metadata.get('framework', 'Unknown')}")
        if "input_features" in metadata:
            click.echo(f"   Features:   {metadata['input_features']}")
        if "output_type" in metadata:
            click.echo(f"   Task:       {metadata['output_type']}")
        click.echo(f"   Size:       {model_size:.2f} MB")
        click.echo()

        # Step 4: Create FastAPI app
        app = create_app(model, loader, model_name)

        # Display server info
        click.secho("🎉 Server starting!", fg="green", bold=True)
        click.echo()
        click.secho(f"   URL:      http://{host}:{port}", fg="blue", bold=True)
        click.secho(f"   API Docs: http://{host}:{port}/docs", fg="blue")
        click.secho(f"   Health:   http://{host}:{port}/health", fg="blue")
        click.echo()
        click.secho("📖 Quick Start:", fg="cyan", bold=True)
        click.echo(f"   curl -X POST http://{host}:{port}/predict \\")
        click.echo('     -H "Content-Type: application/json" \\')

        # Create example input based on framework
        if framework == "huggingface":
            # Text input for HuggingFace models
            task = metadata.get("task", "text-classification")
            if task == "text-generation":
                click.echo('     -d \'{"features": "Once upon a time"}\'')
            else:
                click.echo('     -d \'{"features": "This product is amazing!"}\'')
        elif "input_features" in metadata:
            # Numeric input with correct number of features
            num_features = metadata["input_features"]
            example_features = ", ".join(["1.0"] * min(num_features, 4))
            if num_features > 4:
                example_features += ", ..."
            click.echo(f'     -d \'{{"features": [{example_features}]}}\'')
        else:
            # Default numeric input
            click.echo('     -d \'{"features": [1.0, 2.0, 3.0, 4.0]}\'')

        click.echo()
        click.secho("Press Ctrl+C to stop", fg="yellow")
        click.echo()
        click.echo("─" * 60)
        click.echo()

        # Step 5: Start server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            access_log=True,
        )

    except UnsupportedModelError as e:
        click.echo()
        click.secho("❌ Error: Unsupported Model Format", fg="red", bold=True, err=True)
        click.echo()
        click.echo(str(e), err=True)
        click.echo()
        sys.exit(1)

    except ModelLoadError as e:
        click.echo()
        click.secho("❌ Error: Failed to Load Model", fg="red", bold=True, err=True)
        click.echo()
        click.echo(str(e), err=True)
        click.echo()
        sys.exit(1)

    except KeyboardInterrupt:
        click.echo()
        click.secho("\n👋 Shutting down...", fg="yellow")
        click.echo()
        sys.exit(0)

    except Exception as e:
        click.echo()
        click.secho("❌ Unexpected Error", fg="red", bold=True, err=True)
        click.echo()
        click.echo(f"{type(e).__name__}: {e}", err=True)
        click.echo()
        click.secho(
            "Please report this issue: https://github.com/prabhueshwarla/shipml/issues",
            fg="yellow",
            err=True,
        )
        click.echo()
        sys.exit(1)


if __name__ == "__main__":
    cli()
