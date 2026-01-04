#!/bin/bash
# Run all tests for ShipML

set -e

echo "🧪 Running ShipML Integration Tests"
echo "===================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated. Activating .venv..."
    source .venv/bin/activate 2>/dev/null || {
        echo "❌ No virtual environment found at .venv/"
        echo "   Run: uv venv && source .venv/bin/activate"
        exit 1
    }
fi

# Environment readiness check
echo "🔍 Checking environment dependencies..."
echo ""

# Function to check if a Python package is installed
check_package() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

# Check core dependencies
MISSING_DEPS=()

if ! check_package "pytest"; then
    echo "  ❌ pytest not found"
    MISSING_DEPS+=("dev")
else
    echo "  ✅ pytest installed"
fi

if ! check_package "sklearn"; then
    echo "  ❌ scikit-learn not found"
    MISSING_DEPS+=("sklearn")
else
    echo "  ✅ scikit-learn installed"
fi

if ! check_package "torch"; then
    echo "  ❌ PyTorch not found"
    MISSING_DEPS+=("pytorch")
else
    echo "  ✅ PyTorch installed"
fi

if ! check_package "tensorflow"; then
    echo "  ❌ TensorFlow not found"
    MISSING_DEPS+=("tensorflow")
else
    echo "  ✅ TensorFlow installed"
fi

if ! check_package "tf_keras"; then
    echo "  ❌ tf-keras not found"
    MISSING_DEPS+=("tensorflow")
else
    echo "  ✅ tf-keras installed"
fi

if ! check_package "transformers"; then
    echo "  ❌ Transformers (HuggingFace) not found"
    MISSING_DEPS+=("huggingface")
else
    echo "  ✅ Transformers installed"
fi

# Install missing dependencies
if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo ""
    echo "📦 Installing missing dependencies..."
    for dep in "${MISSING_DEPS[@]}"; do
        echo "  Installing: $dep"
        if [ "$dep" = "dev" ]; then
            uv pip install -e ".[dev]" --quiet
        else
            uv pip install -e ".[$dep]" --quiet
        fi
    done
    echo "  ✅ All dependencies installed"
else
    echo ""
    echo "✅ All dependencies already installed"
fi

# Always ensure the package itself is installed in editable mode
uv pip install -e . --quiet

echo ""
echo "🔬 Running tests..."
echo ""

# Run all tests with coverage
pytest tests/ -v --cov=shipml --cov-report=term-missing

echo ""
echo "✅ All tests completed!"
