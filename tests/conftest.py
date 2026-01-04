"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def sklearn_model_path(tmp_path):
    """Create a temporary sklearn model for testing."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        import joblib

        # Create synthetic dataset
        X, y = make_classification(
            n_samples=100, n_features=4, n_classes=2, random_state=42
        )

        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Save model
        model_path = tmp_path / "sklearn_model.pkl"
        joblib.dump(model, model_path)

        return model_path

    except ImportError:
        pytest.skip("scikit-learn not installed")


@pytest.fixture
def sample_features():
    """Sample input features for testing."""
    return [1.0, 2.0, 3.0, 4.0]
