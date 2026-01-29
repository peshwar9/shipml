# Contributing to mlship

Thank you for your interest in contributing to mlship! This guide will help you get started.

---

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sudhanvalabs/mlship
cd mlship
```

### 2. Install uv (Recommended)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Create Virtual Environment

```bash
uv venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
```

### 4. Install in Development Mode

```bash
# Install with all dependencies for development
uv pip install -e ".[dev,all]"
```

This installs:
- mlship in editable mode (`-e`)
- All framework support (sklearn, pytorch, tensorflow, huggingface)
- Development tools (pytest, black, ruff, mypy)

---

## Project Structure

```
mlship/
├── mlship/              # Source code
│   ├── __init__.py
│   ├── cli.py          # CLI commands (Click)
│   ├── server.py       # FastAPI app generator
│   ├── models.py       # Pydantic models
│   ├── errors.py       # Custom exceptions
│   ├── utils.py        # Helper functions
│   └── loaders/        # Model loaders
│       ├── base.py     # Base interface
│       ├── detector.py # Framework detection
│       ├── sklearn.py  # Scikit-learn
│       ├── pytorch.py  # PyTorch
│       ├── tensorflow.py # TensorFlow
│       └── huggingface.py # Hugging Face
├── tests/              # Test suite
│   ├── conftest.py     # Pytest fixtures
│   ├── test_cli.py     # CLI tests
│   └── test_loaders.py # Loader tests
├── examples/           # Example scripts
├── pyproject.toml      # Package configuration
└── README.md
```

---

## Running Tests

mlship has comprehensive integration tests that verify all model frameworks work correctly.

### Quick Test Run

**Option 1: Cross-platform Python script (recommended)**
```bash
# Works on Windows, macOS, and Linux
python run_tests.py
```

**Option 2: Bash script (macOS/Linux only)**
```bash
./run_tests.sh
```

Both scripts will:
- Check for missing dependencies
- Install any missing frameworks automatically
- Run all tests with coverage report

**Platform recommendations:**
- Windows: Use `python run_tests.py`
- macOS/Linux: Either script works, but `python run_tests.py` is more portable

### Run All Tests

```bash
pytest
```

### Run Integration Tests Only

**API Integration Tests** - Test FastAPI endpoints directly:

```bash
pytest tests/test_integration.py -v
```

This tests:
- ✅ Sklearn models (created on-the-fly)
- ✅ PyTorch models (created on-the-fly)
- ✅ HuggingFace models (downloads DistilBERT ~256MB)
- ⏭️  TensorFlow models (skipped if not installed)

Each test:
1. Downloads/creates the model
2. Loads it with mlship
3. Tests all endpoints: `/health`, `/info`, `/predict`, `/docs`
4. Tests error handling
5. Cleans up (deletes downloaded models)

**CLI Integration Tests** - Test actual CLI commands (serve, benchmark):

```bash
pytest tests/test_cli_integration.py -v
```

This tests:
- ✅ `mlship serve` command with sklearn, PyTorch, TensorFlow, HuggingFace models
- ✅ `mlship benchmark` command with all model types
- ✅ Custom payloads and output formats
- ✅ End-to-end CLI workflow

### Run Framework-Specific Tests

Test only specific frameworks (if you don't have all dependencies installed):

```bash
# Test only sklearn
pytest tests/test_cli_integration.py::TestServeCommandCLI::test_serve_sklearn_cli -v

# Test only PyTorch
pytest tests/test_cli_integration.py::TestServeCommandCLI::test_serve_pytorch_cli -v

# Test only TensorFlow
pytest tests/test_cli_integration.py::TestServeCommandCLI::test_serve_tensorflow_cli -v

# Test benchmarking
pytest tests/test_cli_integration.py::TestBenchmarkCommandCLI -v
```

### Skip Slow Tests

HuggingFace Hub tests download models (~268MB) and are marked as slow:

```bash
# Skip slow tests
pytest -m "not slow"

# Run only slow tests
pytest -m "slow"
```

### Run with Coverage

```bash
pytest --cov=mlship --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Run Specific Test

```bash
pytest tests/test_cli.py::test_version
```

### Run Tests for Specific Framework

```bash
pytest tests/test_integration.py::TestSklearnIntegration -v
pytest tests/test_integration.py::TestPyTorchIntegration -v
pytest tests/test_integration.py::TestHuggingFaceIntegration -v
```

### Pre-Push Checks

**Recommended:** Run all CI checks locally before pushing to catch issues early:

```bash
./pre_push.sh
```

This runs:
- ✅ Black formatting check
- ✅ Ruff linting
- ✅ MyPy type checking
- ✅ All tests

**Manual checks:**

```bash
# Quick check
pytest tests/test_integration.py --tb=short

# Full check with coverage
pytest --cov=mlship --cov-report=term-missing

# Format check only
black --check mlship/ tests/

# Lint check only
ruff check mlship/ tests/
```

### Regression Testing

When adding new features or making changes, run regression tests to ensure existing functionality still works:

**Full regression test suite:**
```bash
# Activate virtual environment first
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Run all tests
pytest tests/ -v
```

**Test specific areas:**
```bash
# Test core loaders (sklearn, PyTorch, TensorFlow, HuggingFace)
pytest tests/test_loaders.py -v

# Test CLI commands (unit tests)
pytest tests/test_cli.py -v

# Test API endpoints (FastAPI integration)
pytest tests/test_integration.py -v

# Test CLI commands with real models (end-to-end)
pytest tests/test_cli_integration.py -v

# Test benchmark functionality
pytest tests/test_benchmark.py -v
```

**Comprehensive framework coverage testing:**
```bash
# Test all frameworks with actual CLI commands
pytest tests/test_cli_integration.py -v

# This tests:
# - mlship serve with sklearn, PyTorch, TensorFlow, HuggingFace models
# - mlship benchmark with all model types
# - Custom payloads and output formats
# - Full end-to-end workflows
```

**With coverage report:**
```bash
pytest tests/ -v --cov=mlship --cov-report=term-missing

# Or for HTML report
pytest tests/ --cov=mlship --cov-report=html
```

This ensures:
- ✅ All model loaders still work (sklearn, PyTorch, TensorFlow, HuggingFace)
- ✅ CLI commands work correctly (both unit and integration tests)
- ✅ Server endpoints function properly
- ✅ Benchmark command works with all model types
- ✅ Error handling is intact
- ✅ No breaking changes introduced

**Example workflow when adding a feature:**
1. Create feature branch
2. Implement feature
3. Write tests for new feature
4. Run regression tests: `pytest tests/ -v`
5. Fix any broken tests
6. Run pre-push checks: `./pre_push.sh`
7. Commit and push

### GitHub Actions (Automated Testing)

mlship uses GitHub Actions to automatically test on **every push** across:

**Platforms:**
- ✅ Ubuntu (Linux)
- ✅ Windows
- ✅ macOS

**Python versions:**
- ✅ Python 3.9
- ✅ Python 3.10
- ✅ Python 3.11

**Frameworks tested:**
- Individual tests for sklearn, pytorch, tensorflow, huggingface
- Full integration tests
- Code linting and formatting checks

The CI workflow is defined in `.github/workflows/test.yml` and runs automatically on every push to ensure cross-platform compatibility.

**Cost:** Free for public repositories (unlimited minutes)

---

## Code Quality

### Format Code

```bash
black mlship/ tests/
```

### Lint Code

```bash
ruff check mlship/ tests/
```

### Auto-fix Linting Issues

```bash
ruff check --fix mlship/ tests/
```

### Type Check

```bash
mypy mlship/
```

### Run All Checks

```bash
# Format
black mlship/ tests/

# Lint
ruff check --fix mlship/ tests/

# Type check
mypy mlship/

# Test
pytest
```

---

## Adding a New Model Framework

Want to add support for a new ML framework? Here's how:

### 1. Create Loader Class

Create `mlship/loaders/newframework.py`:

```python
from pathlib import Path
from typing import Any, Dict, List, Union
from mlship.errors import ModelLoadError, ValidationError
from mlship.loaders.base import ModelLoader

class NewFrameworkLoader(ModelLoader):
    """Loader for NewFramework models."""

    def load(self, model_path: Path) -> Any:
        """Load model from file."""
        try:
            # Load model using framework's library
            import newframework
            model = newframework.load(str(model_path))
            return model
        except ImportError:
            raise ModelLoadError(
                "NewFramework is not installed.\n\n"
                "Install with: uv pip install newframework"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")

    def predict(self, model: Any, features: Union[List[float], List[List[float]]]) -> Dict[str, Any]:
        """Run prediction."""
        import numpy as np
        X = np.array(features)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        prediction = model.predict(X)[0]
        return {"prediction": float(prediction)}

    def get_metadata(self, model: Any) -> Dict[str, Any]:
        """Extract metadata."""
        return {
            "model_type": model.__class__.__name__,
            "framework": "newframework",
        }

    def validate_input(self, model: Any, features: Union[List[float], List[List[float]]]) -> None:
        """Validate input."""
        # Add validation logic
        pass
```

### 2. Update Detector

Edit `mlship/loaders/detector.py`:

```python
def detect_framework(model_path: Path) -> str:
    # ... existing code ...
    elif extension == ".newext":
        return "newframework"

def get_loader(framework: str) -> ModelLoader:
    # ... existing code ...
    elif framework == "newframework":
        from mlship.loaders.newframework import NewFrameworkLoader
        return NewFrameworkLoader()
```

### 3. Add Dependencies

Edit `pyproject.toml`:

```toml
[project.optional-dependencies]
newframework = ["newframework>=1.0"]
all = [
    # ... existing ...
    "newframework>=1.0",
]
```

### 4. Write Tests

Create `tests/test_newframework.py`:

```python
import pytest
from pathlib import Path
from mlship.loaders.newframework import NewFrameworkLoader

def test_load():
    """Test loading model."""
    loader = NewFrameworkLoader()
    # ... test implementation ...

def test_predict():
    """Test prediction."""
    # ... test implementation ...
```

### 5. Update Documentation

Add to `README.md` supported frameworks table.

---

## Testing Your Changes

### Manual Testing

```bash
# 1. Create a test model
python examples/sklearn_example.py

# 2. Serve it
mlship serve fraud_detector.pkl

# 3. Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}'
```

### Automated Testing

```bash
# Run tests
pytest

# Check coverage
pytest --cov=mlship --cov-report=term-missing
```

---

## Pull Request Process

1. **Fork the repository** on GitHub

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

4. **Run quality checks**
   ```bash
   black mlship/ tests/
   ruff check --fix mlship/ tests/
   pytest
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

   Use conventional commits:
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation
   - `test:` tests
   - `refactor:` code refactoring

6. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

7. **Create Pull Request** on GitHub
   - Describe your changes
   - Reference any related issues
   - Ensure CI checks pass

---

## Code Style Guidelines

### Python Style

- Follow PEP 8
- Use type hints
- Keep functions focused and small
- Write docstrings for public APIs

**Example:**
```python
def load_model(model_path: Path) -> Any:
    """
    Load ML model from file.

    Args:
        model_path: Path to model file

    Returns:
        Loaded model object

    Raises:
        ModelLoadError: If model cannot be loaded
    """
    # Implementation...
```

### Error Messages

Make error messages helpful:

```python
# ❌ Bad
raise ValueError("Invalid input")

# ✅ Good
raise ValidationError(
    f"Invalid input shape\n\n"
    f"Expected: {expected} features\n"
    f"Received: {received} features\n\n"
    f"Example correct input:\n"
    f'{{"features": [1.0, 2.0, 3.0]}}'
)
```

### CLI Output

Use colors and clear formatting:

```python
click.secho("✓ Success", fg="green")
click.secho("❌ Error", fg="red", err=True)
click.secho("⚠️  Warning", fg="yellow")
```

---

## Benchmarking

Run performance benchmarks:

```bash
python benchmark.py
```

See `PERFORMANCE.md` for current benchmarks.

---

## Post-Release Smoke Test

The post-release smoke test validates that a published PyPI package works correctly by installing it in a fresh environment and running through all QUICKSTART examples.

### What It Tests

The smoke test validates:
- ✅ **scikit-learn models** - Creates and serves a RandomForestClassifier
- ✅ **PyTorch models** - Creates and serves a TorchScript model
- ✅ **TensorFlow models** - Creates and serves a Keras model
- ✅ **HuggingFace Hub models** - Serves a model directly from HuggingFace Hub

Each test:
1. Creates a fresh virtual environment with Python 3.12
2. Installs mlship from PyPI (not local files)
3. Installs framework dependencies
4. Creates/downloads model
5. Starts mlship server
6. Makes prediction via curl
7. Validates response contains expected fields

### Usage

**Test latest version from PyPI:**
```bash
./scripts/post_release_smoke_test.sh
```

**Test specific version:**
```bash
./scripts/post_release_smoke_test.sh 0.1.5
```

### When to Run

**Critical times:**
- ⚠️ **After publishing to PyPI** - Validates the package is installable and works
- Before creating git tag
- Before announcing releases

**Optional times:**
- After major changes to loaders
- Before publishing to TestPyPI (to catch issues early)

### Sample Output

```
==========================================
SMOKE TEST SUMMARY
==========================================

Test Environment:
  Python: python3.12
  mlship version: 0.1.5

Test Results:
┌─────────────────────────────────────────────────┬──────────┐
│ Test Case                                       │ Status   │
├─────────────────────────────────────────────────┼──────────┤
│ sklearn model created                           │ ✓ PASS   │
│ sklearn prediction                              │ ✓ PASS   │
│ PyTorch model created                           │ ✓ PASS   │
│ PyTorch prediction                              │ ✓ PASS   │
│ TensorFlow model created                        │ ✓ PASS   │
│ TensorFlow prediction                           │ ✓ PASS   │
│ HuggingFace prediction                          │ ✓ PASS   │
└─────────────────────────────────────────────────┴──────────┘

Summary:
  Total tests: 7
  Passed: 7
  Failed: 0

╔════════════════════════════════════════════════╗
║                                                ║
║  ✓ ALL SMOKE TESTS PASSED!                    ║
║                                                ║
║  mlship 0.1.5 is working correctly!           ║
║                                                ║
╚════════════════════════════════════════════════╝
```

### Troubleshooting

**Python 3.12 not found:**
```bash
# macOS
brew install python@3.12

# Ubuntu/Debian
sudo apt install python3.12 python3.12-venv
```

**Script hangs:**
- Check if port 8765 is already in use
- Kill any existing mlship processes
- The script automatically cleans up on exit

**Tests fail:**
- Check the server log: `/tmp/mlship-smoke-test-*/server.log`
- Verify PyPI package was published correctly
- Wait a few minutes if just published (CDN propagation)

---

## Contributing Checklist

**For Contributors:** Before submitting a pull request, complete this checklist:

### 1. Code Changes

- [ ] **Write/Update Tests**
  - Add test cases for new features in `tests/`
  - Update existing tests if behavior changed
  - Ensure test coverage for edge cases

- [ ] **Run Regression Tests**
  ```bash
  source .venv/bin/activate
  pytest tests/ -v
  ```
  All existing tests must pass.

- [ ] **Run Code Quality Checks**
  ```bash
  # Format code
  black mlship/ tests/

  # Lint code
  ruff check --fix mlship/ tests/

  # Type check (optional but recommended)
  mypy mlship/
  ```

- [ ] **Test Manually**
  - Test your changes with real models
  - Verify error messages are helpful
  - Check edge cases

### 2. Documentation

- [ ] **Update Documentation**
  - Update `README.md` if adding new feature
  - Update `QUICKSTART.md` with examples if user-facing
  - Update `CONTRIBUTING.md` if changing workflow
  - Add docstrings to new functions/classes

- [ ] **Update CHANGELOG** (if exists)
  - Add entry describing your changes

### 3. Submit Pull Request

- [ ] **Create Feature Branch**
  ```bash
  git checkout -b feature/your-feature-name
  ```

- [ ] **Commit with Clear Messages**
  ```bash
  git commit -m "feat: add benchmark command"
  git commit -m "fix: resolve model loading issue"
  git commit -m "docs: update quickstart guide"
  ```

- [ ] **Push Branch**
  ```bash
  git push origin feature/your-feature-name
  ```

- [ ] **Open Pull Request**
  - Describe what your PR does
  - Reference related issues (e.g., "Closes #123")
  - Add screenshots/examples if relevant

### 4. After PR Submission

- [ ] **Wait for CI to Pass**
  - GitHub Actions will run tests automatically
  - Fix any failures before requesting review

- [ ] **Address Review Comments**
  - Respond to reviewer feedback
  - Make requested changes
  - Push updates to same branch

- [ ] **Merge**
  - Maintainer will merge after approval
  - Delete your feature branch after merge

**Quick Pre-PR Command:**
```bash
# Run all checks at once
./pre_push.sh
```

---

## Release Process

**For Maintainers:** Publishing a new version to PyPI

### Pre-Release Checklist

Before starting the release:

- [ ] All tests passing on `main` branch
- [ ] All PRs for this release are merged
- [ ] CHANGELOG updated (if exists)
- [ ] Documentation updated for new features

### Release Steps

#### 1. Update Version Numbers

Update version in **two files**:

**`pyproject.toml`:**
```toml
version = "0.2.0"
```

**`mlship/__init__.py`:**
```python
__version__ = "0.2.0"
```

Commit the changes:
```bash
git add pyproject.toml mlship/__init__.py
git commit -m "chore: bump version to 0.2.0"
git push origin main
```

#### 2. Build Package

```bash
# Clean old builds
rm -rf dist/ build/ *.egg-info

# Build
python -m build
```

Verify the build:
```bash
ls -lh dist/
# Should show: mlship-0.2.0-py3-none-any.whl and mlship-0.2.0.tar.gz
```

#### 3. Test Installation Locally

```bash
# Install from wheel
pip install dist/mlship-0.2.0-py3-none-any.whl

# Verify version
mlship --version
# Should output: mlship version 0.2.0

# Quick functionality test
mlship --help
```

#### 4. Test on TestPyPI (Optional but Recommended)

⚠️ **Recommended for major releases:**

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mlship

# Test it works
mlship --version
```

#### 5. Publish to PyPI

**This is irreversible! Double-check everything.**

```bash
twine upload dist/*
```

Wait 2-3 minutes for PyPI CDN propagation.

#### 6. Run Post-Release Smoke Test

⚠️ **Critical: Validates the published package**

```bash
# Test the latest version from PyPI
./scripts/post_release_smoke_test.sh

# Or test specific version
./scripts/post_release_smoke_test.sh 0.2.0
```

This validates:
- Package installs correctly from PyPI
- All frameworks work (sklearn, PyTorch, TensorFlow, HuggingFace)
- All QUICKSTART examples work
- No import errors or missing dependencies

**If smoke test fails:** The package is already published but broken. Immediately:
1. Fix the issue
2. Bump to patch version (e.g., 0.2.0 → 0.2.1)
3. Republish

#### 7. Create Git Tag

```bash
# Create annotated tag
git tag -a v0.2.0 -m "Release v0.2.0: Add benchmark command"

# Push tag to GitHub
git push origin v0.2.0
```

#### 8. Create GitHub Release

1. Go to: https://github.com/sudhanvalabs/mlship/releases/new
2. Select tag: `v0.2.0`
3. Release title: `v0.2.0 - Benchmark Command`
4. Description (example):

```markdown
## What's New

- 🎯 **Benchmark Command**: Measure model serving performance with `mlship benchmark`
  - Latency metrics (avg, p50, p95, p99)
  - Throughput measurement
  - JSON output for automation
  - Works with all frameworks

## Installation

```bash
pip install mlship
```

## Full Changelog

- feat: add benchmark command (#1)
- fix: improve PyTorch model loading
- docs: enhance quickstart guide

**Full Changelog**: https://github.com/sudhanvalabs/mlship/compare/v0.1.5...v0.2.0
```

5. Click "Publish release"

### Post-Release

- [ ] Announce on LinkedIn/Twitter
- [ ] Update any related blog posts
- [ ] Close milestone (if using GitHub milestones)
- [ ] Thank contributors

### Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **Major** (1.0.0): Breaking changes
- **Minor** (0.2.0): New features, backward compatible
- **Patch** (0.1.6): Bug fixes, backward compatible

For detailed publishing instructions, see [PUBLISHING.md](PUBLISHING.md).

---

## Getting Help

- Read existing code - it's well-documented
- Check tests for usage examples
- Open a [Discussion](https://github.com/sudhanvalabs/mlship/discussions) for questions
- Open an [Issue](https://github.com/sudhanvalabs/mlship/issues) for bugs

---

## Code of Conduct

Be respectful and constructive. We're all here to build something useful together.

---

**Thank you for contributing to mlship!** 🚀
