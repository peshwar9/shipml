#!/bin/bash
# Run all checks before pushing to catch CI failures early

set -e

echo "🔍 Running pre-push checks..."
echo ""

# Format check
echo "1️⃣  Checking code formatting..."
.venv/bin/black --check shipml/ tests/ || {
    echo "❌ Code formatting failed!"
    echo "   Run: .venv/bin/black shipml/ tests/"
    exit 1
}
echo "✅ Formatting OK"
echo ""

# Lint check
echo "2️⃣  Linting code..."
.venv/bin/ruff check shipml/ tests/ || {
    echo "❌ Linting failed!"
    echo "   Run: .venv/bin/ruff check --fix shipml/ tests/"
    exit 1
}
echo "✅ Linting OK"
echo ""

# Type check (allow to fail)
echo "3️⃣  Type checking..."
.venv/bin/mypy shipml/ || echo "⚠️  Type check warnings (non-blocking)"
echo ""

# Run tests
echo "4️⃣  Running tests..."
.venv/bin/python -m pytest tests/ -v || {
    echo "❌ Tests failed!"
    exit 1
}
echo "✅ Tests OK"
echo ""

echo "✅ All pre-push checks passed! Safe to push."
