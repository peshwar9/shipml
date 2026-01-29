"""Tests for benchmarking functionality."""

import json

from click.testing import CliRunner

from mlship.cli import cli


class TestBenchmarkCommand:
    """Test benchmark CLI command."""

    def test_benchmark_sklearn_model(self, tmp_path):
        """Test benchmarking sklearn model."""
        # Create a simple sklearn model
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)

        # Run benchmark
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "benchmark",
                str(model_path),
                "--requests",
                "10",
                "--warmup",
                "2",
                "--port",
                "8765",
            ],
        )

        # Check that it succeeded
        assert result.exit_code == 0, f"Command failed with output: {result.output}"

        # Check output contains expected text
        assert "BENCHMARK RESULTS" in result.output
        assert "Cold Start:" in result.output
        assert "Average:" in result.output
        assert "P50 (Median):" in result.output
        assert "P95:" in result.output
        assert "P99:" in result.output
        assert "Throughput:" in result.output

    def test_benchmark_json_output(self, tmp_path):
        """Test benchmark with JSON output."""
        # Create a simple sklearn model
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)

        # Run benchmark with JSON output
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "benchmark",
                str(model_path),
                "--requests",
                "10",
                "--warmup",
                "2",
                "--port",
                "8766",
                "--output",
                "json",
            ],
        )

        # Check that it succeeded
        assert result.exit_code == 0, f"Command failed with output: {result.output}"

        # Parse JSON output
        output_data = json.loads(result.output)

        # Verify required fields
        assert "model" in output_data
        assert "framework" in output_data
        assert output_data["framework"] == "sklearn"
        assert "cold_start_ms" in output_data
        assert "avg_ms" in output_data
        assert "min_ms" in output_data
        assert "p50_ms" in output_data
        assert "p95_ms" in output_data
        assert "p99_ms" in output_data
        assert "max_ms" in output_data
        assert "throughput_rps" in output_data
        assert "warmup_requests" in output_data
        assert output_data["warmup_requests"] == 2
        assert "benchmark_requests" in output_data
        assert output_data["benchmark_requests"] == 10

        # Verify values are reasonable
        assert output_data["cold_start_ms"] > 0
        assert output_data["avg_ms"] > 0
        assert output_data["throughput_rps"] > 0

    def test_benchmark_custom_payload(self, tmp_path):
        """Test benchmark with custom payload."""
        # Create a simple sklearn model
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)

        # Run benchmark with custom payload
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "benchmark",
                str(model_path),
                "--requests",
                "5",
                "--warmup",
                "1",
                "--port",
                "8767",
                "--payload",
                '{"features": [1.0, 2.0, 3.0, 4.0]}',
            ],
        )

        # Check that it succeeded
        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        assert "BENCHMARK RESULTS" in result.output

    def test_benchmark_invalid_payload(self, tmp_path):
        """Test benchmark with invalid JSON payload."""
        # Create a simple sklearn model
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)

        # Run benchmark with invalid payload
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "benchmark",
                str(model_path),
                "--payload",
                "not valid json",
            ],
        )

        # Should fail with error
        assert result.exit_code != 0
        assert "Invalid JSON Payload" in result.output

    def test_benchmark_nonexistent_file(self):
        """Test benchmark with non-existent file."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["benchmark", "nonexistent_model.pkl"],
        )

        # Should fail with error
        assert result.exit_code != 0
        assert "File Not Found" in result.output

    def test_benchmark_help(self):
        """Test benchmark help message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["benchmark", "--help"])

        assert result.exit_code == 0
        assert "Benchmark model serving performance" in result.output
        assert "--requests" in result.output
        assert "--warmup" in result.output
        assert "--port" in result.output
        assert "--source" in result.output
        assert "--output" in result.output
        assert "--payload" in result.output
