"""Framework detection and loader factory."""

from pathlib import Path
from typing import Union

from mlship.errors import UnsupportedModelError
from mlship.loaders.base import ModelLoader


def detect_framework(model_path: Union[Path, str], source: str = "local") -> str:
    """
    Detect ML framework from file extension, content, or source flag.

    Args:
        model_path: Path to model file/directory OR model ID (e.g., "bert-base-uncased")
        source: Model source - "local" (default) or "huggingface"

    Returns:
        Framework name: 'sklearn' | 'pytorch' | 'tensorflow' | 'xgboost' | 'lightgbm' | 'huggingface'

    Raises:
        UnsupportedModelError: If framework cannot be detected
    """
    # Handle HuggingFace Hub models
    if source == "huggingface":
        return "huggingface"

    # Convert string to Path for local models
    if isinstance(model_path, str):
        model_path = Path(model_path)

    if not model_path.exists():
        raise UnsupportedModelError(f"Model path does not exist: {model_path}")

    extension = model_path.suffix.lower()

    # Extension-based detection
    if extension in [".pkl", ".joblib"]:
        # Could be sklearn, xgboost, or lightgbm
        # Try to detect by loading (implemented in sklearn loader for now)
        return "sklearn"
    elif extension in [".pt", ".pth"]:
        return "pytorch"
    elif extension in [".h5", ".keras"]:
        return "tensorflow"
    elif extension == ".json":
        return "xgboost"
    elif model_path.is_dir():
        # Check for Hugging Face model
        if (model_path / "config.json").exists() and (
            (model_path / "pytorch_model.bin").exists()
            or (model_path / "model.safetensors").exists()
        ):
            return "huggingface"
        # Check for TensorFlow SavedModel
        elif (model_path / "saved_model.pb").exists():
            return "tensorflow"

    # If we can't detect, raise error with helpful message
    raise UnsupportedModelError(
        f"Could not detect framework for: {model_path}\n\n"
        f"Supported formats:\n"
        f"  • Scikit-learn: .pkl, .joblib\n"
        f"  • PyTorch: .pt, .pth\n"
        f"  • TensorFlow: .h5, .keras, SavedModel/\n"
        f"  • XGBoost: .json, .pkl\n"
        f"  • LightGBM: .txt, .pkl\n\n"
        f"Need support for another format?\n"
        f"Open an issue: https://github.com/prabhueshwarla/shipml/issues"
    )


def get_loader(framework: str) -> ModelLoader:
    """
    Get appropriate loader for the framework.

    Args:
        framework: Framework name

    Returns:
        ModelLoader instance

    Raises:
        UnsupportedModelError: If framework is not supported
    """
    if framework == "sklearn":
        from mlship.loaders.sklearn import SklearnLoader

        return SklearnLoader()
    elif framework == "pytorch":
        from mlship.loaders.pytorch import PyTorchLoader

        return PyTorchLoader()
    elif framework == "tensorflow":
        from mlship.loaders.tensorflow import TensorFlowLoader

        return TensorFlowLoader()
    elif framework == "huggingface":
        from mlship.loaders.huggingface import HuggingFaceLoader

        return HuggingFaceLoader()
    elif framework == "xgboost":
        # TODO: Implement XGBoostLoader
        raise UnsupportedModelError("XGBoost support coming soon!")
    elif framework == "lightgbm":
        # TODO: Implement LightGBMLoader
        raise UnsupportedModelError("LightGBM support coming soon!")
    else:
        raise UnsupportedModelError(f"Unsupported framework: {framework}")
