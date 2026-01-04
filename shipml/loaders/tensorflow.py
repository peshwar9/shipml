"""TensorFlow/Keras model loader."""

from pathlib import Path
from typing import Any, Dict, List, Union

from shipml.errors import ModelLoadError, ValidationError
from shipml.loaders.base import ModelLoader


class TensorFlowLoader(ModelLoader):
    """Loader for TensorFlow/Keras models."""

    def load(self, model_path: Path) -> Any:
        """Load TensorFlow model from .h5/.keras file or SavedModel directory."""
        try:
            import tensorflow as tf

            # Suppress TensorFlow warnings
            import os

            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

            if model_path.is_dir():
                # SavedModel format
                model = tf.keras.models.load_model(str(model_path))
            else:
                # HDF5 format (.h5 or .keras)
                model = tf.keras.models.load_model(str(model_path))

            return model

        except ImportError:
            raise ModelLoadError(
                "TensorFlow is not installed.\n\n"
                "Install with: uv pip install tensorflow\n"
                "Or: uv pip install shipml[tensorflow]"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load TensorFlow model: {e}")

    def predict(
        self, model: Any, features: Union[List[float], List[List[float]]]
    ) -> Dict[str, Any]:
        """Run prediction on input features."""
        import numpy as np

        # Convert to numpy array
        X = np.array(features, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Run prediction
        output = model.predict(X, verbose=0)

        # Handle different output shapes
        if output.shape[1] > 1:
            # Multi-class classification
            probabilities = output[0]
            prediction = np.argmax(output[0])

            return {
                "prediction": int(prediction),
                "probability": float(probabilities.max()),
            }
        else:
            # Regression or binary classification
            prediction = output[0, 0]

            # Check if binary classification (sigmoid activation)
            if 0 <= prediction <= 1:
                # Treat as probability
                return {
                    "prediction": int(prediction > 0.5),
                    "probability": float(prediction) if prediction > 0.5 else float(1 - prediction),
                }
            else:
                # Regression
                return {"prediction": float(prediction)}

    def get_metadata(self, model: Any) -> Dict[str, Any]:
        """Extract model metadata."""
        metadata = {
            "model_type": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "framework": "tensorflow",
        }

        # Input shape
        if hasattr(model, "input_shape") and model.input_shape:
            input_shape = model.input_shape
            if isinstance(input_shape, tuple) and len(input_shape) > 1:
                # Exclude batch dimension
                metadata["input_features"] = input_shape[1]

        # Output shape to determine type
        if hasattr(model, "output_shape") and model.output_shape:
            output_shape = model.output_shape
            if isinstance(output_shape, tuple) and len(output_shape) > 1:
                output_dim = output_shape[1]
                if output_dim == 1:
                    metadata["output_type"] = "regression"
                else:
                    metadata["output_type"] = "classification"

        return metadata

    def validate_input(
        self, model: Any, features: Union[List[float], List[List[float]]]
    ) -> None:
        """Validate input shape."""
        import numpy as np

        X = np.array(features, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get expected input shape
        if hasattr(model, "input_shape") and model.input_shape:
            expected_features = model.input_shape[1]
            received_features = X.shape[1]

            if received_features != expected_features:
                raise ValidationError(
                    f"Invalid input shape\n\n"
                    f"Expected: {expected_features} features\n"
                    f"Received: {received_features} features\n\n"
                    f"Your model expects {expected_features} input features, "
                    f"but you provided {received_features}.\n\n"
                    f"Example correct input:\n"
                    f'{{"features": [{", ".join(["1.0"] * expected_features)}]}}'
                )
