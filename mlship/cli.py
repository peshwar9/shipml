"""Command-line interface for mlship."""

import sys
from pathlib import Path

import click
import uvicorn

from mlship import __version__
from mlship.errors import UnsupportedModelError, ModelLoadError
from mlship.loaders import detect_framework, get_loader
from mlship.server import create_app
from mlship.utils import get_model_size_mb


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version and exit")
@click.pass_context
def cli(ctx, version):
    """
    mlship - Turn ML models into APIs with one command.

    Examples:

      mlship serve model.pkl

      mlship serve model.pt --port 5000

      mlship serve model.h5 --name "sentiment-analyzer"
    """
    if version:
        click.echo(f"mlship version {__version__}")
        ctx.exit(0)

    # If no command is provided, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.argument("model_file", type=str)
@click.option("--port", default=8000, type=int, help="Port to run server on (default: 8000)")
@click.option("--host", default="127.0.0.1", type=str, help="Host to bind to (default: 127.0.0.1)")
@click.option("--name", default=None, type=str, help="Model name for display")
@click.option(
    "--pipeline",
    default=None,
    type=str,
    help="Custom pipeline class (e.g. pipeline.SentimentPipeline)",
)
@click.option("--reload", is_flag=True, help="Enable auto-reload on file changes (dev mode)")
@click.option(
    "--source",
    type=click.Choice(["local", "huggingface"], case_sensitive=False),
    default="local",
    help="Model source: 'local' for file paths (default) or 'huggingface' for Hub models",
)
def serve(
    model_file: str, port: int, host: str, name: str, pipeline: str, reload: bool, source: str
):
    """
    Start API server for your ML model.

    MODEL_FILE can be:
      - Path to a trained model file (.pkl, .pt, .h5, etc.) [--source local]
      - HuggingFace Hub model ID (e.g., 'bert-base-uncased') [--source huggingface]

    Examples:

      mlship serve model.pkl

      mlship serve fraud_detector.pkl --port 5000

      mlship serve model.h5 --name "sentiment-analyzer"

      mlship serve model.pt --reload  # Development mode

      mlship serve bert-base-uncased --source huggingface

      mlship serve distilbert-base-uncased-finetuned-sst-2-english --source huggingface

      mlship serve gpt2 --source huggingface --port 5000
    """
    # Handle local vs Hub models
    if source == "local":
        # Validate that local path exists
        model_path_obj = Path(model_file)
        if not model_path_obj.exists():
            click.echo()
            click.secho("❌ Error: File Not Found", fg="red", bold=True, err=True)
            click.echo()
            click.echo(f"The model file does not exist: {model_file}", err=True)
            click.echo()
            click.echo("If you're trying to load from HuggingFace Hub, use:", err=True)
            click.echo(f"  mlship serve {model_file} --source huggingface", err=True)
            click.echo()
            sys.exit(1)

        model_path = model_path_obj
        model_name = name or model_path_obj.stem
    else:  # source == "huggingface"
        # For Hub models, keep as string
        model_path = model_file
        # Extract name from model ID (e.g., "bert-base-uncased" or "org/model" -> "model")
        model_name = name or model_file.split("/")[-1]

    # Display header
    click.echo()
    click.secho("🚀 mlship", fg="blue", bold=True)
    if source == "huggingface":
        click.echo(f"   Loading model from Hub: {model_file}")
    else:
        click.echo(f"   Loading model: {Path(model_file).name}")
    click.echo()

    try:
        # Step 1: Detect framework
        click.echo("🔍 Detecting framework...", nl=False)
        framework = detect_framework(model_path, source=source)
        click.secho(f" ✓ {framework}", fg="green")

        # Step 2: Load model
        if source == "huggingface":
            click.echo("📦 Loading model from HuggingFace Hub...", nl=False)
        else:
            click.echo("📦 Loading model...", nl=False)
        loader = get_loader(framework)
        model = loader.load(model_path)
        click.secho(" ✓ Success", fg="green")

        # Step 3: Get metadata
        metadata = loader.get_metadata(model)

        # For Hub models, size calculation needs special handling
        if source == "huggingface" and isinstance(model_path, str):
            model_size = 0.0
        else:
            model_size = get_model_size_mb(model_path)

        # Display model info
        click.echo()
        click.secho("📊 Model Information:", fg="cyan", bold=True)
        click.echo(f"   Name:       {model_name}")
        click.echo(f"   Type:       {metadata.get('model_type', 'Unknown')}")
        click.echo(f"   Framework:  {metadata.get('framework', 'Unknown')}")
        if source == "huggingface":
            click.echo("   Source:     HuggingFace Hub")
        if "input_features" in metadata:
            click.echo(f"   Features:   {metadata['input_features']}")
        if "output_type" in metadata:
            click.echo(f"   Task:       {metadata['output_type']}")
        if model_size > 0:
            click.echo(f"   Size:       {model_size:.2f} MB")
        click.echo()

        # Step 4: Load custom pipeline if provided
        pipeline_instance = None
        if pipeline:
            click.echo("🔧 Loading custom pipeline...", nl=False)
            try:
                import importlib
                from pathlib import Path as PathLib

                # Add current directory to Python path so users can import local files
                sys.path.insert(0, str(PathLib.cwd()))

                # Parse module.ClassName format
                if "." in pipeline:
                    module_name, class_name = pipeline.rsplit(".", 1)
                else:
                    raise ValueError("Pipeline must be in format 'module.ClassName'")

                # Import module and get class
                module = importlib.import_module(module_name)
                pipeline_class = getattr(module, class_name)

                # Instantiate pipeline with model path
                # For Hub models, pass the model ID string; for local, pass the directory
                if isinstance(model_path, Path):
                    pipeline_path = str(model_path.parent if model_path.is_file() else model_path)
                else:
                    # Hub model ID
                    pipeline_path = model_path
                pipeline_instance = pipeline_class(pipeline_path)

                click.secho(" ✓ Success", fg="green")
            except Exception as e:
                click.secho(" ✗ Failed", fg="red")
                click.secho(f"   Error: {str(e)}", fg="red", err=True)
                click.echo()
                click.echo("Pipeline class should:")
                click.echo("  1. Be in format 'module.ClassName'")
                click.echo("  2. Inherit from mlship.pipeline.Pipeline")
                click.echo("  3. Implement preprocess() and postprocess() methods")
                raise click.Abort()

        # Step 5: Create FastAPI app
        app = create_app(model, loader, model_name, pipeline_instance)

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
            click.echo(f"     -d '{{\"features\": [{example_features}]}}'")
        else:
            # Default numeric input
            click.echo("     -d '{\"features\": [1.0, 2.0, 3.0, 4.0]}'")

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
