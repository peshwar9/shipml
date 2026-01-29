"""Integration tests for all model frameworks.

These tests:
1. Download/create models for each framework
2. Load models and start FastAPI server
3. Test all endpoints (/predict, /health, /info, /docs)
4. Cleanup (shutdown server, delete models)
"""

import pytest
import shutil
from fastapi.testclient import TestClient

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


class TestSklearnIntegration:
    """Integration tests for sklearn models."""

    @pytest.fixture
    def sklearn_model(self, tmp_path):
        """Create sklearn model for testing."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            import joblib

            # Create and train model
            X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)

            # Save with joblib (recommended)
            model_path = tmp_path / "sklearn_model.pkl"
            joblib.dump(model, model_path)

            yield model_path

            # Cleanup
            if model_path.exists():
                model_path.unlink()

        except ImportError:
            pytest.skip("scikit-learn not installed")

    @pytest.fixture
    def sklearn_app(self, sklearn_model):
        """Create FastAPI app for sklearn model."""
        from mlship.loaders.detector import detect_framework, get_loader
        from mlship.server import create_app

        framework = detect_framework(sklearn_model)
        loader = get_loader(framework)
        model = loader.load(sklearn_model)
        app = create_app(model, loader, "test_sklearn")

        return app

    def test_sklearn_health_endpoint(self, sklearn_app):
        """Test /health endpoint."""
        client = TestClient(sklearn_app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "test_sklearn"
        assert "uptime_seconds" in data

    def test_sklearn_info_endpoint(self, sklearn_app):
        """Test /info endpoint."""
        client = TestClient(sklearn_app)
        response = client.get("/info")

        assert response.status_code == 200
        data = response.json()
        assert data["framework"] == "scikit-learn"
        assert data["model_name"] == "test_sklearn"
        assert data["input_features"] == 10
        assert data["output_type"] == "classification"

    def test_sklearn_predict_endpoint(self, sklearn_app):
        """Test /predict endpoint."""
        client = TestClient(sklearn_app)
        response = client.post(
            "/predict", json={"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
        )

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert data["model_name"] == "test_sklearn"

    def test_sklearn_predict_wrong_shape(self, sklearn_app):
        """Test /predict with wrong input shape."""
        client = TestClient(sklearn_app)
        response = client.post("/predict", json={"features": [1.0, 2.0]})

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "Validation Error" in data["error"]

    def test_sklearn_docs_endpoint(self, sklearn_app):
        """Test /docs endpoint returns OpenAPI docs."""
        client = TestClient(sklearn_app)
        response = client.get("/docs")

        assert response.status_code == 200


class TestPyTorchIntegration:
    """Integration tests for PyTorch models."""

    @pytest.fixture
    def pytorch_model(self, tmp_path):
        """Create PyTorch model for testing."""
        try:
            import torch

            model = SimpleNet()
            model_path = tmp_path / "pytorch_model.pt"
            torch.save(model, model_path)

            yield model_path

            # Cleanup
            if model_path.exists():
                model_path.unlink()

        except ImportError:
            pytest.skip("PyTorch not installed")

    @pytest.fixture
    def pytorch_app(self, pytorch_model):
        """Create FastAPI app for PyTorch model."""
        from mlship.loaders.detector import detect_framework, get_loader
        from mlship.server import create_app

        framework = detect_framework(pytorch_model)
        loader = get_loader(framework)
        model = loader.load(pytorch_model)
        app = create_app(model, loader, "test_pytorch")

        return app

    def test_pytorch_health_endpoint(self, pytorch_app):
        """Test /health endpoint."""
        client = TestClient(pytorch_app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_name"] == "test_pytorch"

    def test_pytorch_info_endpoint(self, pytorch_app):
        """Test /info endpoint."""
        client = TestClient(pytorch_app)
        response = client.get("/info")

        assert response.status_code == 200
        data = response.json()
        assert data["framework"] == "pytorch"
        assert data["model_name"] == "test_pytorch"

    def test_pytorch_predict_endpoint(self, pytorch_app):
        """Test /predict endpoint."""
        client = TestClient(pytorch_app)
        response = client.post(
            "/predict", json={"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
        )

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["model_name"] == "test_pytorch"


class TestTensorFlowIntegration:
    """Integration tests for TensorFlow/Keras models."""

    @pytest.fixture
    def tensorflow_model(self, tmp_path):
        """Create TensorFlow model for testing."""
        try:
            import tensorflow as tf

            # Simple sequential model
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

            # Cleanup
            if model_path.exists():
                model_path.unlink()

        except ImportError:
            pytest.skip("TensorFlow not installed")

    @pytest.fixture
    def tensorflow_app(self, tensorflow_model):
        """Create FastAPI app for TensorFlow model."""
        from mlship.loaders.detector import detect_framework, get_loader
        from mlship.server import create_app

        framework = detect_framework(tensorflow_model)
        loader = get_loader(framework)
        model = loader.load(tensorflow_model)
        app = create_app(model, loader, "test_tensorflow")

        return app

    def test_tensorflow_health_endpoint(self, tensorflow_app):
        """Test /health endpoint."""
        client = TestClient(tensorflow_app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_name"] == "test_tensorflow"

    def test_tensorflow_info_endpoint(self, tensorflow_app):
        """Test /info endpoint."""
        client = TestClient(tensorflow_app)
        response = client.get("/info")

        assert response.status_code == 200
        data = response.json()
        assert data["framework"] == "tensorflow"
        assert data["model_name"] == "test_tensorflow"

    def test_tensorflow_predict_endpoint(self, tensorflow_app):
        """Test /predict endpoint."""
        client = TestClient(tensorflow_app)
        response = client.post(
            "/predict", json={"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
        )

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["model_name"] == "test_tensorflow"


class TestHuggingFaceIntegration:
    """Integration tests for Hugging Face models."""

    @pytest.fixture
    def huggingface_model(self, tmp_path):
        """Download and save Hugging Face model for testing."""
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            # Use a small, fast model for testing
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            model_path = tmp_path / "hf_model"

            # Download and save
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

            yield model_path

            # Cleanup
            if model_path.exists():
                shutil.rmtree(model_path)

        except ImportError:
            pytest.skip("Transformers not installed")
        except Exception as e:
            pytest.skip(f"Failed to download Hugging Face model: {e}")

    @pytest.fixture
    def huggingface_app(self, huggingface_model):
        """Create FastAPI app for Hugging Face model."""
        from mlship.loaders.detector import detect_framework, get_loader
        from mlship.server import create_app

        framework = detect_framework(huggingface_model)
        loader = get_loader(framework)
        model = loader.load(huggingface_model)
        app = create_app(model, loader, "test_huggingface")

        return app

    def test_huggingface_health_endpoint(self, huggingface_app):
        """Test /health endpoint."""
        client = TestClient(huggingface_app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_name"] == "test_huggingface"

    def test_huggingface_info_endpoint(self, huggingface_app):
        """Test /info endpoint."""
        client = TestClient(huggingface_app)
        response = client.get("/info")

        assert response.status_code == 200
        data = response.json()
        assert data["framework"] == "huggingface-transformers"
        assert data["model_name"] == "test_huggingface"

    def test_huggingface_predict_endpoint(self, huggingface_app):
        """Test /predict endpoint with text input."""
        client = TestClient(huggingface_app)
        response = client.post("/predict", json={"features": "This product is amazing!"})

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert data["model_name"] == "test_huggingface"

    def test_huggingface_predict_wrong_type(self, huggingface_app):
        """Test /predict with wrong input type (numeric instead of text)."""
        client = TestClient(huggingface_app)
        response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})

        # FastAPI returns 422 for validation errors
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


class TestFrameworkDetection:
    """Test framework auto-detection."""

    def test_sklearn_detection(self, tmp_path):
        """Test sklearn model detection."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            import joblib
            from mlship.loaders.detector import detect_framework

            model = RandomForestClassifier()
            model_path = tmp_path / "test.pkl"
            joblib.dump(model, model_path)

            framework = detect_framework(model_path)
            assert framework == "sklearn"

            model_path.unlink()

        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_pytorch_detection(self, tmp_path):
        """Test PyTorch model detection."""
        try:
            import torch
            import torch.nn as nn
            from mlship.loaders.detector import detect_framework

            model = nn.Linear(10, 2)
            model_path = tmp_path / "test.pt"
            torch.save(model, model_path)

            framework = detect_framework(model_path)
            assert framework == "pytorch"

            model_path.unlink()

        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_tensorflow_detection(self, tmp_path):
        """Test TensorFlow model detection."""
        try:
            import tensorflow as tf
            from mlship.loaders.detector import detect_framework

            model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
            model_path = tmp_path / "test.h5"
            model.save(model_path)

            framework = detect_framework(model_path)
            assert framework == "tensorflow"

            model_path.unlink()

        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_huggingface_detection(self, tmp_path):
        """Test Hugging Face model detection."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from mlship.loaders.detector import detect_framework

            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            model_path = tmp_path / "hf_test"

            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

            framework = detect_framework(model_path)
            assert framework == "huggingface"

            shutil.rmtree(model_path)

        except ImportError:
            pytest.skip("Transformers not installed")
        except Exception as e:
            pytest.skip(f"Failed to download model: {e}")
