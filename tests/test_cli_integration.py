"""Integration tests for mlship CLI commands (serve, benchmark) across all frameworks.

These tests run the actual CLI commands with real model files to ensure
end-to-end functionality works correctly.
"""

import json
import subprocess
import sys
import time

import pytest
import requests

IS_WINDOWS = sys.platform == "win32"


@pytest.fixture(scope="session")
def sklearn_model_file(tmp_path_factory):
    """Create sklearn model file for testing."""
    pytest.importorskip("sklearn")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    import joblib

    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    model_path = tmp_path_factory.mktemp("models") / "sklearn_model.pkl"
    joblib.dump(model, model_path)

    return model_path


@pytest.fixture(scope="session")
def pytorch_model_file(tmp_path_factory):
    """Create PyTorch TorchScript model file for testing."""
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
    scripted_model = torch.jit.script(model)

    model_path = tmp_path_factory.mktemp("models") / "pytorch_model.pt"
    torch.jit.save(scripted_model, model_path)

    return model_path


@pytest.fixture(scope="session")
def tensorflow_model_file(tmp_path_factory):
    """Create TensorFlow model file for testing."""
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

    model_path = tmp_path_factory.mktemp("models") / "tensorflow_model.keras"
    model.save(model_path)

    return model_path


class TestServeCommandCLI:
    """Test mlship serve command via subprocess."""

    def start_serve_process(self, model_path, port, source="local"):
        """Start mlship serve command in subprocess."""
        cmd = ["mlship", "serve", str(model_path), "--port", str(port)]
        if source == "huggingface":
            cmd.extend(["--source", "huggingface"])

        # On Windows, use CREATE_NEW_PROCESS_GROUP so we can kill the entire tree
        kwargs = {}
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs
        )
        return proc

    def stop_serve_process(self, proc):
        """Stop serve subprocess and all its children."""
        if IS_WINDOWS:
            # On Windows, terminate() doesn't kill child processes.
            # Use taskkill /T to kill the process tree.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
        else:
            proc.terminate()
        proc.wait(timeout=10)
        if proc.poll() is None:
            proc.kill()

    def wait_for_server(self, port, timeout=30):
        """Wait for server to become ready."""
        url = f"http://127.0.0.1:{port}/health"
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.5)
        return False

    @pytest.mark.skipif(
        IS_WINDOWS,
        reason="Subprocess-based serve tests unreliable on Windows CI; "
        "serve is covered by test-frameworks CI job on Windows",
    )
    def test_serve_sklearn_cli(self, sklearn_model_file):
        """Test mlship serve with sklearn model."""
        port = 8101
        proc = self.start_serve_process(sklearn_model_file, port)

        try:
            assert self.wait_for_server(port, timeout=30), "Server failed to start"

            # Test predict
            response = requests.post(
                f"http://127.0.0.1:{port}/predict",
                json={"features": [1.0, 2.0, 3.0, 4.0]},
                timeout=10,
            )
            assert response.status_code == 200
            assert "prediction" in response.json()

        finally:
            self.stop_serve_process(proc)

    @pytest.mark.skipif(
        IS_WINDOWS,
        reason="Subprocess-based serve tests unreliable on Windows CI; "
        "serve is covered by test-frameworks CI job on Windows",
    )
    def test_serve_pytorch_cli(self, pytorch_model_file):
        """Test mlship serve with PyTorch model."""
        port = 8102
        proc = self.start_serve_process(pytorch_model_file, port)

        try:
            assert self.wait_for_server(port, timeout=30), "Server failed to start"

            # Test predict
            response = requests.post(
                f"http://127.0.0.1:{port}/predict",
                json={"features": [1.0, 2.0, 3.0, 4.0]},
                timeout=10,
            )
            assert response.status_code == 200
            assert "prediction" in response.json()

        finally:
            self.stop_serve_process(proc)

    @pytest.mark.skipif(
        IS_WINDOWS,
        reason="Subprocess-based serve tests unreliable on Windows CI; "
        "serve is covered by test-frameworks CI job on Windows",
    )
    def test_serve_tensorflow_cli(self, tensorflow_model_file):
        """Test mlship serve with TensorFlow model."""
        port = 8103
        proc = self.start_serve_process(tensorflow_model_file, port)

        try:
            assert self.wait_for_server(port, timeout=30), "Server failed to start"

            # Test predict
            response = requests.post(
                f"http://127.0.0.1:{port}/predict",
                json={"features": [0.5, 1.2, -0.3, 0.8]},
                timeout=10,
            )
            assert response.status_code == 200
            assert "prediction" in response.json()

        finally:
            self.stop_serve_process(proc)

    @pytest.mark.slow
    @pytest.mark.skipif(
        IS_WINDOWS,
        reason="Subprocess-based serve tests unreliable on Windows CI; "
        "serve is covered by test-frameworks CI job on Windows",
    )
    def test_serve_huggingface_hub_cli(self):
        """Test mlship serve with HuggingFace Hub model."""
        pytest.importorskip("transformers")

        port = 8104
        model_id = "distilbert-base-uncased-finetuned-sst-2-english"
        proc = self.start_serve_process(model_id, port, source="huggingface")

        try:
            # HuggingFace models take longer to load
            assert self.wait_for_server(port, timeout=90), "Server failed to start"

            # Test predict
            response = requests.post(
                f"http://127.0.0.1:{port}/predict",
                json={"features": "This is a test sentence."},
                timeout=10,
            )
            assert response.status_code == 200
            assert "prediction" in response.json()

        finally:
            self.stop_serve_process(proc)


class TestBenchmarkCommandCLI:
    """Test mlship benchmark command via subprocess."""

    def run_benchmark_command(
        self, model_path, port, requests=20, warmup=3, source="local", output="json"
    ):
        """Run mlship benchmark command."""
        cmd = [
            "mlship",
            "benchmark",
            str(model_path),
            "--port",
            str(port),
            "--requests",
            str(requests),
            "--warmup",
            str(warmup),
            "--output",
            output,
        ]
        if source == "huggingface":
            cmd.extend(["--source", "huggingface"])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result

    def test_benchmark_sklearn_cli(self, sklearn_model_file):
        """Test mlship benchmark with sklearn model."""
        result = self.run_benchmark_command(sklearn_model_file, port=8201)

        assert result.returncode == 0, f"Benchmark failed: {result.stderr}"

        # Parse JSON output
        data = json.loads(result.stdout)
        assert data["framework"] == "sklearn"
        assert data["benchmark_requests"] == 20
        assert data["avg_ms"] > 0
        assert data["throughput_rps"] > 0
        assert data["min_ms"] <= data["avg_ms"] <= data["max_ms"]

    def test_benchmark_pytorch_cli(self, pytorch_model_file):
        """Test mlship benchmark with PyTorch model."""
        result = self.run_benchmark_command(pytorch_model_file, port=8202)

        assert result.returncode == 0, f"Benchmark failed: {result.stderr}"

        data = json.loads(result.stdout)
        assert data["framework"] == "pytorch"
        assert data["avg_ms"] > 0
        assert data["throughput_rps"] > 0

    def test_benchmark_tensorflow_cli(self, tensorflow_model_file):
        """Test mlship benchmark with TensorFlow model."""
        result = self.run_benchmark_command(tensorflow_model_file, port=8203)

        assert result.returncode == 0, f"Benchmark failed: {result.stderr}"

        data = json.loads(result.stdout)
        assert data["framework"] == "tensorflow"
        assert data["avg_ms"] > 0
        assert data["throughput_rps"] > 0

    @pytest.mark.slow
    def test_benchmark_huggingface_hub_cli(self):
        """Test mlship benchmark with HuggingFace Hub model."""
        pytest.importorskip("transformers")

        model_id = "distilbert-base-uncased-finetuned-sst-2-english"
        result = self.run_benchmark_command(
            model_id, port=8204, requests=10, warmup=2, source="huggingface"
        )

        assert result.returncode == 0, f"Benchmark failed: {result.stderr}"

        data = json.loads(result.stdout)
        assert data["framework"] == "huggingface"
        assert data["avg_ms"] > 0
        assert data["throughput_rps"] > 0

    def test_benchmark_custom_payload_cli(self, sklearn_model_file):
        """Test mlship benchmark with custom payload."""
        cmd = [
            "mlship",
            "benchmark",
            str(sklearn_model_file),
            "--port",
            "8205",
            "--requests",
            "10",
            "--warmup",
            "2",
            "--output",
            "json",
            "--payload",
            '{"features": [5.0, 6.0, 7.0, 8.0]}',
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        assert result.returncode == 0, f"Benchmark failed: {result.stderr}"

        data = json.loads(result.stdout)
        assert data["benchmark_requests"] == 10
        assert data["avg_ms"] > 0

    def test_benchmark_text_output_cli(self, sklearn_model_file):
        """Test mlship benchmark with text output format."""
        result = self.run_benchmark_command(
            sklearn_model_file, port=8206, requests=10, output="text"
        )

        assert result.returncode == 0, f"Benchmark failed: {result.stderr}"

        # Check text output contains expected sections
        assert "BENCHMARK RESULTS" in result.stdout
        assert "Average:" in result.stdout
        assert "Throughput:" in result.stdout
        assert "P50" in result.stdout
        assert "P95" in result.stdout
        assert "P99" in result.stdout
