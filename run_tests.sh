#!/bin/bash

# GPTPlugins4All Test Runner
# Usage: ./run_tests.sh [test_name]

echo "🧪 GPTPlugins4All Test Runner"
echo "=============================="

# Make sure we're in the right directory
if [ ! -f "test_suite.py" ]; then
    echo "❌ Error: test_suite.py not found. Please run from the project root directory."
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found. Please install Python."
    exit 1
fi

# Run the test
if [ $# -eq 0 ]; then
    echo "🚀 Running all tests..."
    python test_suite.py
else
    echo "🚀 Running test: $1"
    python test_suite.py "$1"
fi

echo ""
echo "✅ Test run completed!"