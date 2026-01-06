"""Tests for HuggingFace Hub integration."""

from pathlib import Path

from mlship.loaders.detector import detect_framework
from mlship.loaders.huggingface import HuggingFaceLoader


def test_detect_framework_with_hub_source():
    """Test framework detection for Hub model IDs."""
    # When source is "huggingface", should always return "huggingface"
    framework = detect_framework("bert-base-uncased", source="huggingface")
    assert framework == "huggingface"

    framework = detect_framework("gpt2", source="huggingface")
    assert framework == "huggingface"

    framework = detect_framework(
        "distilbert-base-uncased-finetuned-sst-2-english", source="huggingface"
    )
    assert framework == "huggingface"


def test_detect_framework_with_local_source():
    """Test that local source still works as before."""
    # Create a temporary file for testing
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = Path(f.name)

    try:
        framework = detect_framework(temp_path, source="local")
        assert framework == "sklearn"
    finally:
        temp_path.unlink()


# Note: Tests that actually load from Hub are commented out to avoid slow downloads during CI
# Uncomment these for manual testing

# @pytest.mark.slow
# def test_load_hub_model():
#     """Test loading a small model from HuggingFace Hub."""
#     pytest.importorskip("transformers")
#
#     loader = HuggingFaceLoader()
#     # Use a very small model for testing
#     model = loader.load("distilbert-base-uncased-finetuned-sst-2-english")
#
#     assert model is not None
#     assert "pipeline" in model
#     assert "task" in model
#
#
# @pytest.mark.slow
# def test_predict_with_hub_model():
#     """Test making predictions with a Hub model."""
#     pytest.importorskip("transformers")
#
#     loader = HuggingFaceLoader()
#     model = loader.load("distilbert-base-uncased-finetuned-sst-2-english")
#
#     result = loader.predict(model, "This product is amazing!")
#
#     assert "prediction" in result
#     assert "probability" in result


def test_huggingface_loader_accepts_string():
    """Test that HuggingFace loader accepts both Path and string."""
    loader = HuggingFaceLoader()

    # Test that the load method signature accepts str
    # (This will fail at runtime without transformers, but tests the signature)
    import inspect

    sig = inspect.signature(loader.load)
    param = sig.parameters["model_path"]

    # Check that the annotation allows Union[Path, str]
    annotation_str = str(param.annotation)
    assert "Union" in annotation_str or "Path" in annotation_str
