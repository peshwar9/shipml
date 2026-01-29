"""Tests for custom preprocessing/postprocessing pipelines."""

import pytest
import shutil
from fastapi.testclient import TestClient

from mlship.pipeline import Pipeline

# Define PyTorch model at module level (can't pickle local classes)
try:
    import torch.nn as nn

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

except ImportError:
    SimpleNet = None


# ============================================================================
# Sklearn Pipeline Tests
# ============================================================================


class SklearnNormalizationPipeline(Pipeline):
    """Test pipeline for sklearn with normalization."""

    def __init__(self, model_path):
        super().__init__(model_path)
        import numpy as np

        # Mock normalization params
        self.mean = np.array([50.0] * 10)
        self.std = np.array([10.0] * 10)

    def preprocess(self, request_data):
        """Normalize features."""
        import numpy as np

        features = np.array(request_data["features"])
        normalized = (features - self.mean) / self.std
        return normalized.tolist()

    def postprocess(self, model_output):
        """Add metadata to output."""
        import numpy as np

        # model_output is raw sklearn prediction (numpy array)
        # Need to format it first
        if isinstance(model_output, np.ndarray):
            prediction = int(model_output[0])
            # For this test, we'll use a mock probability
            result = {"prediction": prediction, "probability": 0.87}
        else:
            result = model_output

        result["preprocessing"] = "normalized"
        result["normalization"] = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        return result


class TestSklearnPipeline:
    """Test sklearn models with custom pipelines."""

    @pytest.fixture
    def sklearn_model(self, tmp_path):
        """Create sklearn model for testing."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            import joblib

            X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)

            model_path = tmp_path / "sklearn_model.pkl"
            joblib.dump(model, model_path)

            yield model_path

            if model_path.exists():
                model_path.unlink()

        except ImportError:
            pytest.skip("scikit-learn not installed")

    @pytest.fixture
    def sklearn_pipeline(self, sklearn_model):
        """Create sklearn pipeline instance."""
        return SklearnNormalizationPipeline(str(sklearn_model.parent))

    @pytest.fixture
    def sklearn_app_with_pipeline(self, sklearn_model, sklearn_pipeline):
        """Create FastAPI app with sklearn pipeline."""
        from mlship.loaders.detector import detect_framework, get_loader
        from mlship.server import create_app

        framework = detect_framework(sklearn_model)
        loader = get_loader(framework)
        model = loader.load(sklearn_model)
        app = create_app(model, loader, "test_sklearn", sklearn_pipeline)

        return app

    def test_sklearn_pipeline_preprocess(self, sklearn_pipeline):
        """Test sklearn preprocessing."""
        import numpy as np

        request_data = {"features": [100.0] * 10}
        result = sklearn_pipeline.preprocess(request_data)

        # Should normalize: (100 - 50) / 10 = 5.0
        assert isinstance(result, list)
        assert len(result) == 10
        np.testing.assert_array_almost_equal(result, [5.0] * 10)

    def test_sklearn_pipeline_postprocess(self, sklearn_pipeline):
        """Test sklearn postprocessing."""
        model_output = {"prediction": 0, "probability": 0.87}
        result = sklearn_pipeline.postprocess(model_output)

        assert result["prediction"] == 0
        assert result["probability"] == 0.87
        assert result["preprocessing"] == "normalized"
        assert "normalization" in result
        assert "mean" in result["normalization"]
        assert "std" in result["normalization"]

    def test_sklearn_predict_with_pipeline(self, sklearn_app_with_pipeline):
        """Test /predict endpoint with pipeline."""
        client = TestClient(sklearn_app_with_pipeline)
        response = client.post(
            "/predict",
            json={
                "features": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
            },
        )

        if response.status_code != 200:
            print(f"Error response: {response.json()}")

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert data["preprocessing"] == "normalized"
        assert data["model_name"] == "test_sklearn"


# ============================================================================
# PyTorch Pipeline Tests
# ============================================================================


class PyTorchTransformPipeline(Pipeline):
    """Test pipeline for PyTorch with tensor transformations."""

    def preprocess(self, request_data):
        """Convert to tensor and reshape."""
        import torch

        features = request_data["features"]
        # Convert to tensor and add batch dimension
        tensor = torch.tensor([features], dtype=torch.float32)
        return tensor

    def postprocess(self, model_output):
        """Format PyTorch output."""
        import torch

        # Get class probabilities
        probabilities = torch.softmax(model_output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "framework": "pytorch",
        }


class TestPyTorchPipeline:
    """Test PyTorch models with custom pipelines."""

    @pytest.fixture
    def pytorch_model(self, tmp_path):
        """Create PyTorch model for testing."""
        try:
            import torch

            model = SimpleNet()
            model_path = tmp_path / "pytorch_model.pt"
            torch.save(model, model_path)

            yield model_path

            if model_path.exists():
                model_path.unlink()

        except ImportError:
            pytest.skip("PyTorch not installed")

    @pytest.fixture
    def pytorch_pipeline(self, pytorch_model):
        """Create PyTorch pipeline instance."""
        return PyTorchTransformPipeline(str(pytorch_model.parent))

    @pytest.fixture
    def pytorch_app_with_pipeline(self, pytorch_model, pytorch_pipeline):
        """Create FastAPI app with PyTorch pipeline."""
        from mlship.loaders.detector import detect_framework, get_loader
        from mlship.server import create_app

        framework = detect_framework(pytorch_model)
        loader = get_loader(framework)
        model = loader.load(pytorch_model)
        app = create_app(model, loader, "test_pytorch", pytorch_pipeline)

        return app

    def test_pytorch_pipeline_preprocess(self, pytorch_pipeline):
        """Test PyTorch preprocessing."""
        try:
            import torch

            request_data = {"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
            result = pytorch_pipeline.preprocess(request_data)

            assert isinstance(result, torch.Tensor)
            assert result.shape == (1, 10)
            assert result.dtype == torch.float32

        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_pytorch_pipeline_postprocess(self, pytorch_pipeline):
        """Test PyTorch postprocessing."""
        try:
            import torch

            # Mock model output (logits)
            model_output = torch.tensor([[0.2, 0.8]])
            result = pytorch_pipeline.postprocess(model_output)

            assert "prediction" in result
            assert "confidence" in result
            assert result["framework"] == "pytorch"
            assert result["prediction"] in [0, 1]
            assert 0.0 <= result["confidence"] <= 1.0

        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_pytorch_predict_with_pipeline(self, pytorch_app_with_pipeline):
        """Test /predict endpoint with pipeline."""
        client = TestClient(pytorch_app_with_pipeline)
        response = client.post(
            "/predict", json={"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
        )

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert data["framework"] == "pytorch"
        assert data["model_name"] == "test_pytorch"


# ============================================================================
# TensorFlow Pipeline Tests
# ============================================================================


class TensorFlowReshapePipeline(Pipeline):
    """Test pipeline for TensorFlow with reshaping."""

    def preprocess(self, request_data):
        """Reshape features for TensorFlow."""
        import numpy as np

        features = np.array(request_data["features"])
        # TensorFlow expects batch dimension
        return features.reshape(1, -1)

    def postprocess(self, model_output):
        """Format TensorFlow output."""

        # Extract prediction value
        if hasattr(model_output, "numpy"):
            prediction = float(model_output.numpy()[0][0])
        else:
            prediction = float(model_output[0][0])

        return {
            "prediction": round(prediction, 4),
            "framework": "tensorflow",
            "output_shape": "binary_classification",
        }


class TestTensorFlowPipeline:
    """Test TensorFlow models with custom pipelines."""

    @pytest.fixture
    def tensorflow_model(self, tmp_path):
        """Create TensorFlow model for testing."""
        try:
            import tensorflow as tf

            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(10,)),
                    tf.keras.layers.Dense(10, activation="relu"),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ]
            )
            model.compile(optimizer="adam", loss="binary_crossentropy")

            model_path = tmp_path / "tensorflow_model.h5"
            model.save(model_path)

            yield model_path

            if model_path.exists():
                model_path.unlink()

        except ImportError:
            pytest.skip("TensorFlow not installed")

    @pytest.fixture
    def tensorflow_pipeline(self, tensorflow_model):
        """Create TensorFlow pipeline instance."""
        return TensorFlowReshapePipeline(str(tensorflow_model.parent))

    @pytest.fixture
    def tensorflow_app_with_pipeline(self, tensorflow_model, tensorflow_pipeline):
        """Create FastAPI app with TensorFlow pipeline."""
        from mlship.loaders.detector import detect_framework, get_loader
        from mlship.server import create_app

        framework = detect_framework(tensorflow_model)
        loader = get_loader(framework)
        model = loader.load(tensorflow_model)
        app = create_app(model, loader, "test_tensorflow", tensorflow_pipeline)

        return app

    def test_tensorflow_pipeline_preprocess(self, tensorflow_pipeline):
        """Test TensorFlow preprocessing."""
        import numpy as np

        request_data = {"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
        result = tensorflow_pipeline.preprocess(request_data)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 10)

    def test_tensorflow_pipeline_postprocess(self, tensorflow_pipeline):
        """Test TensorFlow postprocessing."""
        import numpy as np

        # Mock model output
        model_output = np.array([[0.87]])
        result = tensorflow_pipeline.postprocess(model_output)

        assert "prediction" in result
        assert result["framework"] == "tensorflow"
        assert result["output_shape"] == "binary_classification"
        assert 0.0 <= result["prediction"] <= 1.0

    def test_tensorflow_predict_with_pipeline(self, tensorflow_app_with_pipeline):
        """Test /predict endpoint with pipeline."""
        client = TestClient(tensorflow_app_with_pipeline)
        response = client.post(
            "/predict", json={"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
        )

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["framework"] == "tensorflow"
        assert data["model_name"] == "test_tensorflow"


# ============================================================================
# HuggingFace Pipeline Tests
# ============================================================================


class HuggingFaceSentimentPipeline(Pipeline):
    """Test pipeline for HuggingFace sentiment analysis."""

    def preprocess(self, request_data):
        """Extract text from request.

        Note: HuggingFace pipeline handles tokenization internally!
        """
        return request_data.get("text", "")

    def postprocess(self, model_output):
        """Format HuggingFace output."""
        # HuggingFace pipeline returns list of dicts
        if isinstance(model_output, list):
            result = model_output[0]
        else:
            result = model_output

        return {
            "sentiment": result["label"],
            "confidence": round(result["score"], 4),
            "framework": "huggingface",
        }


class TestHuggingFacePipeline:
    """Test HuggingFace models with custom pipelines."""

    @pytest.fixture
    def huggingface_model(self, tmp_path):
        """Download and save HuggingFace model for testing."""
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            model_path = tmp_path / "hf_model"

            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

            yield model_path

            if model_path.exists():
                shutil.rmtree(model_path)

        except ImportError:
            pytest.skip("Transformers not installed")
        except Exception as e:
            pytest.skip(f"Failed to download HuggingFace model: {e}")

    @pytest.fixture
    def huggingface_pipeline(self, huggingface_model):
        """Create HuggingFace pipeline instance."""
        return HuggingFaceSentimentPipeline(str(huggingface_model))

    @pytest.fixture
    def huggingface_app_with_pipeline(self, huggingface_model, huggingface_pipeline):
        """Create FastAPI app with HuggingFace pipeline."""
        from mlship.loaders.detector import detect_framework, get_loader
        from mlship.server import create_app

        framework = detect_framework(huggingface_model)
        loader = get_loader(framework)
        model = loader.load(huggingface_model)
        app = create_app(model, loader, "test_huggingface", huggingface_pipeline)

        return app

    def test_huggingface_pipeline_preprocess(self, huggingface_pipeline):
        """Test HuggingFace preprocessing."""
        request_data = {"text": "This product is amazing!"}
        result = huggingface_pipeline.preprocess(request_data)

        # Should just extract text, not tokenize!
        assert isinstance(result, str)
        assert result == "This product is amazing!"

    def test_huggingface_pipeline_postprocess(self, huggingface_pipeline):
        """Test HuggingFace postprocessing."""
        # Mock HuggingFace pipeline output
        model_output = [{"label": "POSITIVE", "score": 0.9999}]
        result = huggingface_pipeline.postprocess(model_output)

        assert result["sentiment"] == "POSITIVE"
        assert result["confidence"] == 0.9999
        assert result["framework"] == "huggingface"

    def test_huggingface_predict_with_pipeline(self, huggingface_app_with_pipeline):
        """Test /predict endpoint with custom input format."""
        client = TestClient(huggingface_app_with_pipeline)

        # Test with custom "text" field instead of "features"
        response = client.post("/predict", json={"text": "This product is amazing!"})

        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert "confidence" in data
        assert data["framework"] == "huggingface"
        assert data["model_name"] == "test_huggingface"
        assert data["sentiment"] in ["POSITIVE", "NEGATIVE"]

    def test_huggingface_predict_negative_sentiment(self, huggingface_app_with_pipeline):
        """Test negative sentiment prediction."""
        client = TestClient(huggingface_app_with_pipeline)
        response = client.post("/predict", json={"text": "This product is terrible and broken."})

        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "NEGATIVE"
        assert data["confidence"] > 0.9  # Should be very confident


# ============================================================================
# Pipeline Base Class Tests
# ============================================================================


class TestPipelineBaseClass:
    """Test the Pipeline base class."""

    def test_pipeline_initialization(self, tmp_path):
        """Test pipeline can be initialized with model_path."""

        class TestPipeline(Pipeline):
            def preprocess(self, request_data):
                return request_data

            def postprocess(self, model_output):
                return model_output

        pipeline = TestPipeline(str(tmp_path))
        assert pipeline.model_path == str(tmp_path)

    def test_pipeline_abstract_methods(self, tmp_path):
        """Test that Pipeline requires preprocess and postprocess."""
        with pytest.raises(TypeError):
            # Should fail because preprocess/postprocess not implemented
            Pipeline(str(tmp_path))


# ============================================================================
# Error Handling Tests
# ============================================================================


class BrokenPreprocessPipeline(Pipeline):
    """Pipeline with broken preprocessing."""

    def preprocess(self, request_data):
        raise ValueError("Preprocessing failed!")

    def postprocess(self, model_output):
        return model_output


class BrokenPostprocessPipeline(Pipeline):
    """Pipeline with broken postprocessing."""

    def preprocess(self, request_data):
        return request_data.get("features", [])

    def postprocess(self, model_output):
        raise ValueError("Postprocessing failed!")


class TestPipelineErrorHandling:
    """Test error handling in pipelines."""

    @pytest.fixture
    def sklearn_model(self, tmp_path):
        """Create sklearn model for testing."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            import joblib

            X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)

            model_path = tmp_path / "sklearn_model.pkl"
            joblib.dump(model, model_path)

            yield model_path

            if model_path.exists():
                model_path.unlink()

        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_preprocess_error_handling(self, sklearn_model):
        """Test error handling when preprocessing fails."""
        from mlship.loaders.detector import detect_framework, get_loader
        from mlship.server import create_app

        framework = detect_framework(sklearn_model)
        loader = get_loader(framework)
        model = loader.load(sklearn_model)

        pipeline = BrokenPreprocessPipeline(str(sklearn_model.parent))
        app = create_app(model, loader, "test", pipeline)

        client = TestClient(app)
        response = client.post("/predict", json={"features": [1.0] * 10})

        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "Prediction Failed" in data["error"]

    def test_postprocess_error_handling(self, sklearn_model):
        """Test error handling when postprocessing fails."""
        from mlship.loaders.detector import detect_framework, get_loader
        from mlship.server import create_app

        framework = detect_framework(sklearn_model)
        loader = get_loader(framework)
        model = loader.load(sklearn_model)

        pipeline = BrokenPostprocessPipeline(str(sklearn_model.parent))
        app = create_app(model, loader, "test", pipeline)

        client = TestClient(app)
        response = client.post("/predict", json={"features": [1.0] * 10})

        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "Prediction Failed" in data["error"]
