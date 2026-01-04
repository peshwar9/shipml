"""Scikit-learn model loader."""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Union

from shipml.errors import ModelLoadError, ValidationError
from shipml.loaders.base import ModelLoader


class SklearnLoader(ModelLoader):
    """Loader for scikit-learn models."""

    def load(self, model_path: Path) -> Any:
        """Load sklearn model from pickle/joblib file."""
        try:
            # Try joblib first (recommended for sklearn)
            if model_path.suffix == ".joblib":
                import joblib

                return joblib.load(model_path)
            else:
                # Fallback to pickle
                with open(model_path, "rb") as f:
                    return pickle.load(f)
        except ImportError as e:
            raise ModelLoadError(
                f"Failed to load model. Missing dependency: {e}\n\n"
                f"Install with: uv pip install scikit-learn"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load sklearn model: {e}")

    def predict(
        self, model: Any, features: Union[List[float], List[List[float]]]
    ) -> Dict[str, Any]:
        """Run prediction on input features."""
        import numpy as np

        # Convert to numpy array
        X = np.array(features)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Classification vs Regression
        if hasattr(model, "predict_proba"):
            # Classification
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]

            # For binary classification
            if len(probabilities) == 2:
                return {
                    "prediction": int(prediction),
                    "probability": float(probabilities.max()),
                }
            else:
                # Multi-class classification
                if hasattr(model, "classes_"):
                    prob_dict = {
                        str(cls): float(prob)
                        for cls, prob in zip(model.classes_, probabilities)
                    }
                    return {
                        "prediction": int(prediction),
                        "probabilities": prob_dict,
                    }
                else:
                    return {
                        "prediction": int(prediction),
                        "probability": float(probabilities.max()),
                    }
        else:
            # Regression
            prediction = model.predict(X)[0]
            return {"prediction": float(prediction)}

    def get_metadata(self, model: Any) -> Dict[str, Any]:
        """Extract model metadata."""
        metadata = {
            "model_type": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "framework": "scikit-learn",
        }

        # Number of features
        if hasattr(model, "n_features_in_"):
            metadata["input_features"] = model.n_features_in_

        # Classification vs Regression
        if hasattr(model, "predict_proba"):
            metadata["output_type"] = "classification"
            if hasattr(model, "classes_"):
                metadata["classes"] = model.classes_.tolist()
        else:
            metadata["output_type"] = "regression"

        return metadata

    def validate_input(
        self, model: Any, features: Union[List[float], List[List[float]]]
    ) -> None:
        """Validate input shape."""
        import numpy as np

        X = np.array(features)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if hasattr(model, "n_features_in_"):
            expected = model.n_features_in_
            received = X.shape[1]

            if received != expected:
                raise ValidationError(
                    f"Invalid input shape\n\n"
                    f"Expected: {expected} features\n"
                    f"Received: {received} features\n\n"
                    f"Your model was trained on {expected} features, "
                    f"but you provided {received}.\n\n"
                    f"Example correct input:\n"
                    f'{{"features": [{", ".join(["1.0"] * expected)}]}}'
                )
