"""Utility functions for ShipML."""

from pathlib import Path
from typing import Union


def get_model_size_mb(model_path: Path) -> float:
    """
    Get model file size in megabytes.

    Args:
        model_path: Path to model file or directory

    Returns:
        Size in MB
    """
    if model_path.is_file():
        size_bytes = model_path.stat().st_size
    elif model_path.is_dir():
        # For directories (e.g., TensorFlow SavedModel), sum all files
        size_bytes = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    else:
        return 0.0

    return size_bytes / (1024 * 1024)


def format_error_message(
    error_type: str, message: str, suggestion: Union[str, None] = None
) -> str:
    """
    Format a user-friendly error message.

    Args:
        error_type: Type of error (e.g., "Invalid input shape")
        message: Main error message
        suggestion: Optional suggestion for fixing the error

    Returns:
        Formatted error message
    """
    lines = [f"Error: {error_type}", "", message]

    if suggestion:
        lines.extend(["", suggestion])

    return "\n".join(lines)
