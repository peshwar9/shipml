"""
Benchmark ShipML serve operation to identify bottlenecks.
"""

import time
import sys
from pathlib import Path


def benchmark_sklearn():
    """Benchmark sklearn model loading and serving."""
    print("=" * 60)
    print("BENCHMARK: Scikit-learn Model")
    print("=" * 60)

    # Create a test model
    print("\n1. Creating test sklearn model...")
    start = time.time()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    import joblib

    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    model_path = Path("test_sklearn_model.pkl")
    joblib.dump(model, model_path)
    create_time = time.time() - start
    print(f"   Time: {create_time:.3f}s")

    # Time individual components
    print("\n2. Benchmarking components:")

    # Framework detection
    start = time.time()
    from shipml.loaders import detect_framework
    framework = detect_framework(model_path)
    detect_time = time.time() - start
    print(f"   Framework detection: {detect_time*1000:.2f}ms ({framework})")

    # Model loading
    start = time.time()
    from shipml.loaders import get_loader
    loader = get_loader(framework)
    loaded_model = loader.load(model_path)
    load_time = time.time() - start
    print(f"   Model loading: {load_time*1000:.2f}ms")

    # Metadata extraction
    start = time.time()
    metadata = loader.get_metadata(loaded_model)
    metadata_time = time.time() - start
    print(f"   Metadata extraction: {metadata_time*1000:.2f}ms")

    # FastAPI app creation
    start = time.time()
    from shipml.server import create_app
    app = create_app(loaded_model, loader, "test_model")
    app_time = time.time() - start
    print(f"   FastAPI app creation: {app_time*1000:.2f}ms")

    # Prediction (warmup)
    start = time.time()
    result = loader.predict(loaded_model, [1.0] * 10)
    predict_time = time.time() - start
    print(f"   First prediction: {predict_time*1000:.2f}ms")

    # Prediction (warm)
    times = []
    for _ in range(100):
        start = time.time()
        result = loader.predict(loaded_model, [1.0] * 10)
        times.append((time.time() - start) * 1000)
    avg_predict_time = sum(times) / len(times)
    print(f"   Avg prediction (100 runs): {avg_predict_time:.2f}ms")

    total_time = detect_time + load_time + metadata_time + app_time

    print(f"\n3. Total startup time: {total_time*1000:.2f}ms")
    print("\n4. Breakdown:")
    print(f"   Detection:  {detect_time/total_time*100:.1f}%")
    print(f"   Loading:    {load_time/total_time*100:.1f}%")
    print(f"   Metadata:   {metadata_time/total_time*100:.1f}%")
    print(f"   FastAPI:    {app_time/total_time*100:.1f}%")

    # Cleanup
    model_path.unlink()

    return {
        "framework": "sklearn",
        "total_startup_ms": total_time * 1000,
        "detection_ms": detect_time * 1000,
        "loading_ms": load_time * 1000,
        "metadata_ms": metadata_time * 1000,
        "app_creation_ms": app_time * 1000,
        "first_prediction_ms": predict_time * 1000,
        "avg_prediction_ms": avg_predict_time,
    }


def benchmark_huggingface():
    """Benchmark HuggingFace model loading."""
    print("\n" + "=" * 60)
    print("BENCHMARK: HuggingFace Model")
    print("=" * 60)

    model_path = Path("examples/sentiment-model")

    if not model_path.exists():
        print("\n⚠️  Skipping - model not found at examples/sentiment-model/")
        print("   Run: python examples/huggingface_example.py")
        return None

    print("\n1. Benchmarking components:")

    # Framework detection
    start = time.time()
    from shipml.loaders import detect_framework
    framework = detect_framework(model_path)
    detect_time = time.time() - start
    print(f"   Framework detection: {detect_time*1000:.2f}ms ({framework})")

    # Model loading (THIS IS THE BOTTLENECK)
    print("   Model loading... (this takes ~10 seconds)")
    start = time.time()
    from shipml.loaders import get_loader
    loader = get_loader(framework)
    loaded_model = loader.load(model_path)
    load_time = time.time() - start
    print(f"   Model loading: {load_time:.2f}s (!!)")

    # Metadata extraction
    start = time.time()
    metadata = loader.get_metadata(loaded_model)
    metadata_time = time.time() - start
    print(f"   Metadata extraction: {metadata_time*1000:.2f}ms")

    # FastAPI app creation
    start = time.time()
    from shipml.server import create_app
    app = create_app(loaded_model, loader, "sentiment-model")
    app_time = time.time() - start
    print(f"   FastAPI app creation: {app_time*1000:.2f}ms")

    # Prediction (warmup)
    start = time.time()
    result = loader.predict(loaded_model, "This is great!")
    predict_time = time.time() - start
    print(f"   First prediction: {predict_time*1000:.2f}ms")

    # Prediction (warm)
    times = []
    for _ in range(10):  # Fewer iterations for HF
        start = time.time()
        result = loader.predict(loaded_model, "This is great!")
        times.append((time.time() - start) * 1000)
    avg_predict_time = sum(times) / len(times)
    print(f"   Avg prediction (10 runs): {avg_predict_time:.2f}ms")

    total_time = detect_time + load_time + metadata_time + app_time

    print(f"\n2. Total startup time: {total_time:.2f}s")
    print("\n3. Breakdown:")
    print(f"   Detection:  {detect_time/total_time*100:.1f}%")
    print(f"   Loading:    {load_time/total_time*100:.1f}%  ← BOTTLENECK!")
    print(f"   Metadata:   {metadata_time/total_time*100:.1f}%")
    print(f"   FastAPI:    {app_time/total_time*100:.1f}%")

    return {
        "framework": "huggingface",
        "total_startup_ms": total_time * 1000,
        "detection_ms": detect_time * 1000,
        "loading_ms": load_time * 1000,
        "metadata_ms": metadata_time * 1000,
        "app_creation_ms": app_time * 1000,
        "first_prediction_ms": predict_time * 1000,
        "avg_prediction_ms": avg_predict_time,
    }


def benchmark_cli_overhead():
    """Benchmark CLI overhead (Click framework)."""
    print("\n" + "=" * 60)
    print("BENCHMARK: CLI Overhead")
    print("=" * 60)

    import subprocess

    # Time just importing Click
    start = time.time()
    import click
    click_import_time = time.time() - start
    print(f"\n1. Click import time: {click_import_time*1000:.2f}ms")

    # Time CLI help (no model loading)
    start = time.time()
    result = subprocess.run(
        ["shipml", "--help"],
        capture_output=True,
        text=True
    )
    cli_help_time = time.time() - start
    print(f"2. CLI help time: {cli_help_time*1000:.2f}ms")

    # Time CLI version
    start = time.time()
    result = subprocess.run(
        ["shipml", "--version"],
        capture_output=True,
        text=True
    )
    cli_version_time = time.time() - start
    print(f"3. CLI version time: {cli_version_time*1000:.2f}ms")

    print(f"\n4. CLI overhead: ~{cli_version_time*1000:.0f}ms")
    print("   (This is negligible compared to model loading)")

    return {
        "click_import_ms": click_import_time * 1000,
        "cli_help_ms": cli_help_time * 1000,
        "cli_version_ms": cli_version_time * 1000,
    }


def main():
    print("\n🔬 ShipML Performance Benchmark")
    print("=" * 60)

    results = {}

    # Benchmark sklearn
    try:
        results["sklearn"] = benchmark_sklearn()
    except Exception as e:
        print(f"\n❌ Sklearn benchmark failed: {e}")

    # Benchmark HuggingFace
    try:
        results["huggingface"] = benchmark_huggingface()
    except Exception as e:
        print(f"\n❌ HuggingFace benchmark failed: {e}")

    # Benchmark CLI overhead
    try:
        results["cli"] = benchmark_cli_overhead()
    except Exception as e:
        print(f"\n❌ CLI benchmark failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if "sklearn" in results:
        print(f"\nSklearn model:")
        print(f"  Total startup: {results['sklearn']['total_startup_ms']:.0f}ms")
        print(f"  Avg prediction: {results['sklearn']['avg_prediction_ms']:.2f}ms")

    if "huggingface" in results:
        print(f"\nHuggingFace model:")
        print(f"  Total startup: {results['huggingface']['total_startup_ms']/1000:.1f}s")
        print(f"  Avg prediction: {results['huggingface']['avg_prediction_ms']:.2f}ms")

    if "cli" in results:
        print(f"\nCLI overhead: ~{results['cli']['cli_version_ms']:.0f}ms")

    print("\n" + "=" * 60)
    print("WOULD RUST HELP?")
    print("=" * 60)
    print("\nShort answer: NO")
    print("\nWhere time is spent:")
    if "sklearn" in results:
        print(f"  • Sklearn model loading: {results['sklearn']['loading_ms']:.0f}ms")
        print(f"    - This is Python calling C libraries (joblib, numpy)")
        print(f"    - Rust wouldn't help here")
    if "huggingface" in results:
        print(f"  • HuggingFace loading: {results['huggingface']['loading_ms']/1000:.1f}s")
        print(f"    - This is PyTorch loading weights from disk (I/O bound)")
        print(f"    - Rust wouldn't help here")
    if "cli" in results:
        print(f"  • CLI overhead: {results['cli']['cli_version_ms']:.0f}ms")
        print(f"    - Rust could reduce this to ~5-10ms")
        print(f"    - But it's already negligible!")

    print("\nConclusion:")
    print("  The bottleneck is model loading (Python ML libraries),")
    print("  NOT the CLI. Rewriting in Rust would save <100ms on startup,")
    print("  which is irrelevant when HuggingFace models take 10 seconds to load.")
    print("\n  Keep it in Python! 🐍")


if __name__ == "__main__":
    main()
