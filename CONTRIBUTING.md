# Contributing to ShipML

Thank you for your interest in contributing to ShipML! This guide will help you get started.

---

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/prabhueshwarla/shipml
cd shipml
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
- ShipML in editable mode (`-e`)
- All framework support (sklearn, pytorch, tensorflow, huggingface)
- Development tools (pytest, black, ruff, mypy)

---

## Project Structure

```
shipml/
├── shipml/              # Source code
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

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=shipml --cov-report=html
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
pytest tests/test_loaders.py -k sklearn
```

---

## Code Quality

### Format Code

```bash
black shipml/ tests/
```

### Lint Code

```bash
ruff check shipml/ tests/
```

### Auto-fix Linting Issues

```bash
ruff check --fix shipml/ tests/
```

### Type Check

```bash
mypy shipml/
```

### Run All Checks

```bash
# Format
black shipml/ tests/

# Lint
ruff check --fix shipml/ tests/

# Type check
mypy shipml/

# Test
pytest
```

---

## Adding a New Model Framework

Want to add support for a new ML framework? Here's how:

### 1. Create Loader Class

Create `shipml/loaders/newframework.py`:

```python
from pathlib import Path
from typing import Any, Dict, List, Union
from shipml.errors import ModelLoadError, ValidationError
from shipml.loaders.base import ModelLoader

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

Edit `shipml/loaders/detector.py`:

```python
def detect_framework(model_path: Path) -> str:
    # ... existing code ...
    elif extension == ".newext":
        return "newframework"

def get_loader(framework: str) -> ModelLoader:
    # ... existing code ...
    elif framework == "newframework":
        from shipml.loaders.newframework import NewFrameworkLoader
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
from shipml.loaders.newframework import NewFrameworkLoader

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
shipml serve fraud_detector.pkl

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
pytest --cov=shipml --cov-report=term-missing
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
   black shipml/ tests/
   ruff check --fix shipml/ tests/
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

## Release Process

(For maintainers)

1. **Update version** in `shipml/__init__.py`

2. **Update CHANGELOG** (if exists)

3. **Build package**
   ```bash
   python -m build
   ```

4. **Test installation**
   ```bash
   uv pip install dist/shipml-0.2.0-py3-none-any.whl
   shipml --version
   ```

5. **Publish to PyPI**
   ```bash
   twine upload dist/*
   ```

---

## Getting Help

- Read existing code - it's well-documented
- Check tests for usage examples
- Open a [Discussion](https://github.com/prabhueshwarla/shipml/discussions) for questions
- Open an [Issue](https://github.com/prabhueshwarla/shipml/issues) for bugs

---

## Code of Conduct

Be respectful and constructive. We're all here to build something useful together.

---

**Thank you for contributing to ShipML!** 🚀
