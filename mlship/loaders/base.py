"""Base interface for model loaders."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union
from pathlib import Path


class ModelLoader(ABC):
    """Base interface for all model loaders."""

    @abstractmethod
    def load(self, model_path: Union[Path, str]) -> Any:
        """
        Load model from file or model ID.

        Args:
            model_path: Path to model file/directory OR model ID (e.g., "bert-base-uncased" for HuggingFace Hub)

        Returns:
            Loaded model object

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        pass

    @abstractmethod
    def predict(self, model: Any, features: Any) -> Dict[str, Any]:
        """
        Run prediction on input features.

        Args:
            model: Loaded model object
            features: Input features (numeric arrays for sklearn/pytorch/tensorflow, text for huggingface)

        Returns:
            Dictionary with prediction results

        Raises:
            ValidationError: If input is invalid
        """
        pass

    @abstractmethod
    def get_metadata(self, model: Any) -> Dict[str, Any]:
        """
        Extract model metadata.

        Args:
            model: Loaded model object

        Returns:
            Dictionary with model metadata (type, features, etc.)
        """
        pass

    @abstractmethod
    def validate_input(self, model: Any, features: Any) -> None:
        """
        Validate input shape/format.

        Args:
            model: Loaded model object
            features: Input features to validate (type depends on framework)

        Raises:
            ValidationError: If input is invalid
        """
        pass
