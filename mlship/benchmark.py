"""Benchmarking utilities for mlship."""

import multiprocessing
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

from mlship.loaders.detector import detect_framework
from mlship.server import create_app


def start_server_background(
    model_path: Union[Path, str],
    port: int,
    source: str,
    ready_event: multiprocessing.Event,
) -> None:
    """Start mlship server in background process.

    Args:
        model_path: Path to model file or HuggingFace model ID
        port: Port to run server on
        source: Model source ('local' or 'huggingface')
        ready_event: Event to signal when server is ready
    """
    import uvicorn
    from mlship.loaders import get_loader

    # Detect framework and load model
    framework = detect_framework(model_path, source=source)
    loader = get_loader(framework)
    model = loader.load(model_path)

    # Get model name
    if isinstance(model_path, Path):
        model_name = model_path.stem
    else:
        model_name = str(model_path).split("/")[-1]

    # Create the app
    app = create_app(model, loader, model_name, pipeline=None)

    # Signal that we're ready
    ready_event.set()

    # Run server
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="error",  # Suppress logs during benchmarking
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.run()


def wait_for_server(url: str, timeout: int = 30) -> bool:
    """Wait for server to be ready.

    Args:
        url: Base URL of the server
        timeout: Maximum time to wait in seconds

    Returns:
        True if server is ready, False if timeout
    """
    health_url = f"{url}/health"
    start = time.time()

    while time.time() - start < timeout:
        try:
            response = requests.get(health_url, timeout=1)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.1)

    return False


def generate_test_payload(framework: str) -> Dict[str, Any]:
    """Generate appropriate test payload for framework.

    Args:
        framework: Framework name (sklearn, pytorch, tensorflow, huggingface)

    Returns:
        Test payload dictionary
    """
    if framework == "huggingface":
        return {"features": "This is a test sentence for benchmarking."}
    else:
        # Numeric features for sklearn, pytorch, tensorflow
        return {"features": [1.0, 2.0, 3.0, 4.0]}


def run_benchmark(
    model_path: Union[Path, str],
    port: int = 8000,
    num_requests: int = 100,
    warmup: int = 5,
    source: str = "local",
    payload: Optional[Dict[str, Any]] = None,
    output_format: str = "text",
) -> Dict[str, Any]:
    """Run benchmark on a model.

    Args:
        model_path: Path to model file or HuggingFace model ID
        port: Port to run server on
        num_requests: Number of benchmark requests
        warmup: Number of warmup requests
        source: Model source ('local' or 'huggingface')
        payload: Custom test payload (optional)
        output_format: Output format ('text' or 'json')

    Returns:
        Dictionary with benchmark results
    """
    base_url = f"http://127.0.0.1:{port}"
    predict_url = f"{base_url}/predict"

    # Detect framework to generate appropriate payload
    framework = detect_framework(model_path, source=source)

    # Use custom payload or generate default
    test_payload = payload if payload else generate_test_payload(framework)

    # Start server in background process
    if output_format == "text":
        print(f"Starting mlship server on port {port}...")

    ready_event = multiprocessing.Event()
    server_process = multiprocessing.Process(
        target=start_server_background,
        args=(model_path, port, source, ready_event),
        daemon=True,
    )
    server_process.start()

    # Wait for ready event and server to be accessible
    ready_event.wait(timeout=10)
    time.sleep(1)  # Give server a moment to fully start

    if not wait_for_server(base_url, timeout=30):
        server_process.terminate()
        server_process.join(timeout=5)
        raise RuntimeError("Server failed to start within timeout")

    try:
        if output_format == "text":
            print("Server started. Running benchmark...\n")
            print("=" * 50)
            print(f"Model: {model_path}")
            print(f"Framework: {framework}")
            print(f"Warmup requests: {warmup}")
            print(f"Benchmark requests: {num_requests}")
            print("=" * 50)

        # Measure cold start
        if output_format == "text":
            print("\nMeasuring cold start latency...")

        cold_start_time = time.time()
        try:
            response = requests.post(predict_url, json=test_payload, timeout=30)
            cold_start_duration = time.time() - cold_start_time

            if response.status_code != 200:
                raise RuntimeError(
                    f"Server returned status {response.status_code}: {response.text}"
                )

            if output_format == "text":
                print(f"Cold start: {cold_start_duration:.3f}s")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to server: {e}")

        # Warmup
        if output_format == "text":
            print("\nWarming up...")

        for i in range(warmup):
            requests.post(predict_url, json=test_payload, timeout=30)
            if output_format == "text":
                sys.stdout.write(f"\rWarmup: {i+1}/{warmup}")
                sys.stdout.flush()

        if output_format == "text":
            print()

        # Run benchmark
        if output_format == "text":
            print(f"\nRunning {num_requests} requests...")

        latencies: List[float] = []
        start_time = time.time()

        for i in range(num_requests):
            req_start = time.time()
            response = requests.post(predict_url, json=test_payload, timeout=30)
            req_duration = (time.time() - req_start) * 1000  # Convert to ms
            latencies.append(req_duration)

            if output_format == "text" and (i + 1) % 10 == 0:
                sys.stdout.write(f"\rProgress: {i+1}/{num_requests}")
                sys.stdout.flush()

        total_time = time.time() - start_time

        if output_format == "text":
            print()

        # Calculate statistics
        latencies.sort()
        p50 = statistics.median(latencies)
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        avg = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        throughput = num_requests / total_time

        results = {
            "model": str(model_path),
            "framework": framework,
            "cold_start_ms": cold_start_duration * 1000,
            "warmup_requests": warmup,
            "benchmark_requests": num_requests,
            "avg_ms": avg,
            "min_ms": min_latency,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "max_ms": max_latency,
            "throughput_rps": throughput,
        }

        if output_format == "text":
            print("\n" + "=" * 50)
            print("BENCHMARK RESULTS")
            print("=" * 50)
            print(f"\nCold Start:     {cold_start_duration*1000:.2f}ms")
            print("\nPerformance Metrics:")
            print(f"  Average:       {avg:.2f}ms")
            print(f"  Min:           {min_latency:.2f}ms")
            print(f"  P50 (Median):  {p50:.2f}ms")
            print(f"  P95:           {p95:.2f}ms")
            print(f"  P99:           {p99:.2f}ms")
            print(f"  Max:           {max_latency:.2f}ms")
            print(f"\nThroughput:     ~{throughput:.1f} requests/sec")
            print("=" * 50)

        return results

    finally:
        # Stop server
        if output_format == "text":
            print("\nStopping server...")
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()
