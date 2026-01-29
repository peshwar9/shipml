## Test Suite Structure

This directory contains comprehensive tests for mlship across all supported frameworks.

### Test Files

- **`test_loaders.py`** - Unit tests for model loaders (sklearn, PyTorch, TensorFlow, HuggingFace)
- **`test_cli.py`** - Unit tests for CLI commands and argument parsing
- **`test_benchmark.py`** - Unit tests for benchmark module functions
- **`test_integration.py`** - Integration tests for FastAPI endpoints with all model types
- **`test_cli_integration.py`** - End-to-end tests for CLI commands (`mlship serve`, `mlship benchmark`)
- **`fixtures.py`** - Shared test fixtures for creating test models

### Running Tests

**Run all tests:**
```bash
pytest tests/ -v
```

**Run specific test file:**
```bash
pytest tests/test_cli_integration.py -v
```

**Run specific test:**
```bash
pytest tests/test_cli_integration.py::TestServeCommandCLI::test_serve_sklearn_cli -v
```

**Skip slow tests (HuggingFace Hub downloads):**
```bash
pytest -m "not slow"
```

**Run only slow tests:**
```bash
pytest -m "slow"
```

### Test Coverage

The test suite provides coverage across:

| Area | Coverage |
|------|----------|
| **Model Types** | sklearn, PyTorch, TensorFlow, HuggingFace |
| **Commands** | `serve`, `benchmark` |
| **Endpoints** | `/predict`, `/health`, `/info`, `/docs` |
| **Features** | Custom payloads, output formats, error handling |

### CI/CD Integration

These tests run automatically on:
- Every pull request
- Every push to main branch
- Before release

Tests ensure backward compatibility and prevent regressions across all supported frameworks.
