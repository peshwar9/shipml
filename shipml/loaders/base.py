"""Base interface for model loaders."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from pathlib import Path


class ModelLoader(ABC):
    """Base interface for all model loaders."""

    @abstractmethod
    def load(self, model_path: Path) -> Any:
        """
        Load model from file.

        Args:
            model_path: Path to model file or directory

        Returns:
            Loaded model object

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        pass

    @abstractmethod
    def predict(self, model: Any, features: Union[List[float], List[List[float]]]) -> Dict[str, Any]:
        """
        Run prediction on input features.

        Args:
            model: Loaded model object
            features: Input features (single sample or batch)

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
    def validate_input(self, model: Any, features: Union[List[float], List[List[float]]]) -> None:
        """
        Validate input shape/format.

        Args:
            model: Loaded model object
            features: Input features to validate

        Raises:
            ValidationError: If input is invalid
        """
        pass
