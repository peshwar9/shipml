"""Tests for model loaders."""

import pytest

from shipml.loaders.sklearn import SklearnLoader
from shipml.errors import ValidationError


def test_sklearn_loader_load(sklearn_model_path):
    """Test loading sklearn model."""
    loader = SklearnLoader()
    model = loader.load(sklearn_model_path)

    assert model is not None
    assert hasattr(model, "predict")


def test_sklearn_loader_predict(sklearn_model_path, sample_features):
    """Test sklearn prediction."""
    loader = SklearnLoader()
    model = loader.load(sklearn_model_path)
    result = loader.predict(model, sample_features)

    assert "prediction" in result
    assert "probability" in result or "probabilities" in result
    assert "model_name" not in result  # Added by server, not loader


def test_sklearn_loader_metadata(sklearn_model_path):
    """Test extracting sklearn metadata."""
    loader = SklearnLoader()
    model = loader.load(sklearn_model_path)
    metadata = loader.get_metadata(model)

    assert metadata["framework"] == "scikit-learn"
    assert "model_type" in metadata
    assert "input_features" in metadata
    assert metadata["input_features"] == 4


def test_sklearn_validation_error(sklearn_model_path):
    """Test input validation raises error for wrong shape."""
    loader = SklearnLoader()
    model = loader.load(sklearn_model_path)

    # Wrong number of features (expects 4, giving 2)
    with pytest.raises(ValidationError):
        loader.validate_input(model, [1.0, 2.0])
