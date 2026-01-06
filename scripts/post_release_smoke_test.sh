#!/bin/bash

# Post-Release Smoke Test for mlship
#
# This script validates that a published PyPI release of mlship works correctly
# by installing it in a fresh environment and running through all QUICKSTART examples.
#
# Usage:
#   ./scripts/post_release_smoke_test.sh [version]
#
# Examples:
#   ./scripts/post_release_smoke_test.sh        # Test latest version
#   ./scripts/post_release_smoke_test.sh 0.1.5  # Test specific version

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VERSION="${1:-}"  # Optional version argument
PYTHON_VERSION="python3.12"
TEST_DIR="/tmp/mlship-smoke-test-$$"
VENV_DIR="$TEST_DIR/venv"
SERVER_PORT=8765  # Use non-standard port to avoid conflicts
SERVER_PID=""

# Test results tracking
declare -A TEST_RESULTS
TEST_COUNT=0

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"

    # Stop server if running
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping mlship server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi

    # Remove test directory
    if [ -d "$TEST_DIR" ]; then
        echo "Removing test directory: $TEST_DIR"
        rm -rf "$TEST_DIR"
    fi

    echo -e "${GREEN}Cleanup complete${NC}"
}

# Set up trap to cleanup on exit
trap cleanup EXIT INT TERM

# Print section header
print_section() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

# Print test result and track it
print_result() {
    local status=$1
    local test_name="$2"

    TEST_COUNT=$((TEST_COUNT + 1))

    if [ $status -eq 0 ]; then
        echo -e "${GREEN}✓ $test_name${NC}"
        TEST_RESULTS["$test_name"]="PASS"
    else
        echo -e "${RED}✗ $test_name${NC}"
        TEST_RESULTS["$test_name"]="FAIL"
        exit 1
    fi
}

# Wait for server to be ready
wait_for_server() {
    local max_attempts=30
    local attempt=0

    echo "Waiting for server to start..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:$SERVER_PORT/health > /dev/null 2>&1; then
            echo -e "${GREEN}Server is ready${NC}"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    echo -e "${RED}Server failed to start${NC}"
    return 1
}

# Start mlship server
start_server() {
    local model_path="$1"
    local extra_args="${2:-}"

    echo "Starting mlship server: mlship serve $model_path --port $SERVER_PORT $extra_args"

    # Stop any existing server
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        sleep 2
    fi

    # Start server in background
    "$VENV_DIR/bin/mlship" serve "$model_path" --port $SERVER_PORT $extra_args > "$TEST_DIR/server.log" 2>&1 &
    SERVER_PID=$!

    # Wait for server to be ready
    wait_for_server
}

# Test prediction endpoint
test_prediction() {
    local payload="$1"
    local expected_field="$2"
    local test_name="$3"

    echo "Testing: $test_name"
    echo "Payload: $payload"

    response=$(curl -s -X POST http://localhost:$SERVER_PORT/predict \
        -H "Content-Type: application/json" \
        -d "$payload")

    echo "Response: $response"

    # Check if response contains expected field
    if echo "$response" | grep -q "\"$expected_field\""; then
        print_result 0 "$test_name"
        return 0
    else
        print_result 1 "$test_name - missing field: $expected_field"
        return 1
    fi
}

# Main test execution
main() {
    print_section "mlship Post-Release Smoke Test"

    echo "Test directory: $TEST_DIR"
    echo "Python version: $PYTHON_VERSION"
    if [ -n "$VERSION" ]; then
        echo "Target version: $VERSION"
    else
        echo "Target version: latest"
    fi

    # Check Python version
    print_section "Checking Python Version"
    if ! command -v $PYTHON_VERSION &> /dev/null; then
        echo -e "${RED}Error: $PYTHON_VERSION not found${NC}"
        echo "Please install Python 3.12"
        exit 1
    fi
    $PYTHON_VERSION --version
    print_result 0 "Python version check"

    # Create test directory
    print_section "Setting Up Test Environment"
    mkdir -p "$TEST_DIR"
    cd "$TEST_DIR"

    # Create virtual environment
    echo "Creating virtual environment..."
    $PYTHON_VERSION -m venv "$VENV_DIR"
    print_result 0 "Virtual environment created"

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Install mlship from PyPI
    print_section "Installing mlship from PyPI"
    if [ -n "$VERSION" ]; then
        echo "Installing mlship==$VERSION..."
        pip install --no-cache-dir "mlship==$VERSION" || exit 1
    else
        echo "Installing latest mlship..."
        pip install --no-cache-dir mlship || exit 1
    fi
    print_result 0 "mlship installed"

    # Verify installation
    echo "Installed version:"
    "$VENV_DIR/bin/mlship" --version

    # Test 1: scikit-learn model
    print_section "Test 1: scikit-learn Model"

    # Install scikit-learn
    echo "Installing scikit-learn..."
    pip install --no-cache-dir scikit-learn joblib

    # Create sklearn model
    cat > train_sklearn_model.py << 'EOF'
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib

X, y = make_classification(n_samples=100, n_features=4, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
joblib.dump(model, 'sklearn_model.pkl')
print('✅ Model saved to sklearn_model.pkl')
EOF

    python train_sklearn_model.py
    print_result 0 "sklearn model created"

    # Serve sklearn model
    start_server "sklearn_model.pkl"

    # Test prediction
    test_prediction '{"features": [1.5, 2.3, -0.5, 1.2]}' "prediction" "sklearn prediction"

    # Test 2: PyTorch TorchScript model
    print_section "Test 2: PyTorch Model (TorchScript)"

    # Install PyTorch
    echo "Installing PyTorch..."
    pip install --no-cache-dir torch

    # Create pytorch model
    cat > train_pytorch_model.py << 'EOF'
import torch
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
torch.jit.save(scripted_model, 'pytorch_model.pt')
print('✅ Model saved to pytorch_model.pt')
EOF

    python train_pytorch_model.py
    print_result 0 "PyTorch model created"

    # Serve pytorch model
    start_server "pytorch_model.pt"

    # Test prediction
    test_prediction '{"features": [1.0, 2.0, 3.0, 4.0]}' "prediction" "PyTorch prediction"

    # Test 3: TensorFlow model
    print_section "Test 3: TensorFlow Model"

    # Install TensorFlow
    echo "Installing TensorFlow..."
    pip install --no-cache-dir tensorflow tf-keras

    # Create tensorflow model
    cat > train_tensorflow_model.py << 'EOF'
import tensorflow as tf
from tensorflow import keras
import numpy as np

X = np.random.rand(100, 4)
y = np.random.randint(0, 2, 100)

model = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(4,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, y, epochs=10, verbose=0)

model.save('tensorflow_model.h5')
print('✅ Model saved to tensorflow_model.h5')
EOF

    python train_tensorflow_model.py
    print_result 0 "TensorFlow model created"

    # Serve tensorflow model
    start_server "tensorflow_model.h5"

    # Test prediction
    test_prediction '{"features": [0.5, 1.2, -0.3, 0.8]}' "prediction" "TensorFlow prediction"

    # Test 4: HuggingFace Hub model
    print_section "Test 4: HuggingFace Hub Model"

    # Install transformers
    echo "Installing transformers..."
    pip install --no-cache-dir transformers

    # Serve HF model (no file creation needed)
    start_server "distilbert-base-uncased-finetuned-sst-2-english" "--source huggingface"

    # Test prediction
    test_prediction '{"features": "This product is amazing!"}' "prediction" "HuggingFace prediction"

    # Final summary
    print_section "SMOKE TEST SUMMARY"

    echo ""
    echo -e "${BLUE}Test Environment:${NC}"
    echo "  Python: $PYTHON_VERSION"
    if [ -n "$VERSION" ]; then
        echo "  mlship version: $VERSION"
    else
        echo "  mlship version: latest"
    fi
    echo ""

    echo -e "${BLUE}Test Results:${NC}"
    echo "┌─────────────────────────────────────────────────┬──────────┐"
    echo "│ Test Case                                       │ Status   │"
    echo "├─────────────────────────────────────────────────┼──────────┤"

    # Sort and display test results
    for test_name in "${!TEST_RESULTS[@]}"; do
        status="${TEST_RESULTS[$test_name]}"
        if [ "$status" = "PASS" ]; then
            printf "│ %-47s │ ${GREEN}%-8s${NC} │\n" "$test_name" "✓ PASS"
        else
            printf "│ %-47s │ ${RED}%-8s${NC} │\n" "$test_name" "✗ FAIL"
        fi
    done

    echo "└─────────────────────────────────────────────────┴──────────┘"
    echo ""

    # Count passed/failed
    passed=0
    failed=0
    for status in "${TEST_RESULTS[@]}"; do
        if [ "$status" = "PASS" ]; then
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
        fi
    done

    echo -e "${BLUE}Summary:${NC}"
    echo "  Total tests: $TEST_COUNT"
    echo "  Passed: $passed"
    echo "  Failed: $failed"
    echo ""

    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║                                                ║${NC}"
        echo -e "${GREEN}║  ✓ ALL SMOKE TESTS PASSED!                    ║${NC}"
        echo -e "${GREEN}║                                                ║${NC}"
        echo -e "${GREEN}║  mlship $VERSION is working correctly!         ║${NC}"
        echo -e "${GREEN}║                                                ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"
        echo ""
    else
        echo -e "${RED}╔════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║                                                ║${NC}"
        echo -e "${RED}║  ✗ SMOKE TESTS FAILED                         ║${NC}"
        echo -e "${RED}║                                                ║${NC}"
        echo -e "${RED}║  Please review the errors above                ║${NC}"
        echo -e "${RED}║                                                ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════╝${NC}"
        echo ""
        exit 1
    fi
}

# Run main function
main
