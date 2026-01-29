"""Test fixtures for creating models across all supported frameworks."""

import tempfile
from pathlib import Path
from typing import Dict

import pytest


@pytest.fixture(scope="session")
def sklearn_model() -> Path:
    """Create a sklearn model for testing."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    import joblib

    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    temp_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    joblib.dump(model, temp_file.name)
    temp_file.close()

    yield Path(temp_file.name)

    # Cleanup
    Path(temp_file.name).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def pytorch_model() -> Path:
    """Create a PyTorch TorchScript model for testing."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    model.eval()

    # Save as TorchScript
    scripted_model = torch.jit.script(model)

    temp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.jit.save(scripted_model, temp_file.name)
    temp_file.close()

    yield Path(temp_file.name)

    # Cleanup
    Path(temp_file.name).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def tensorflow_model() -> Path:
    """Create a TensorFlow/Keras model for testing."""
    tf = pytest.importorskip("tensorflow")
    import numpy as np

    X = np.random.rand(100, 4).astype(np.float32)
    y = np.random.randint(0, 2, 100)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(8, activation="relu", input_shape=(4,)),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(X, y, epochs=5, verbose=0)

    temp_file = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
    model.save(temp_file.name)
    temp_file.close()

    yield Path(temp_file.name)

    # Cleanup
    Path(temp_file.name).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def huggingface_model_id() -> str:
    """Return a small HuggingFace Hub model ID for testing."""
    pytest.importorskip("transformers")
    # Use a small, fast model for testing
    return "distilbert-base-uncased-finetuned-sst-2-english"


@pytest.fixture(scope="session")
def test_payloads() -> Dict[str, dict]:
    """Return appropriate test payloads for each framework."""
    return {
        "sklearn": {"features": [1.0, 2.0, 3.0, 4.0]},
        "pytorch": {"features": [1.0, 2.0, 3.0, 4.0]},
        "tensorflow": {"features": [0.5, 1.2, -0.3, 0.8]},
        "huggingface": {"features": "This is a test sentence."},
    }
