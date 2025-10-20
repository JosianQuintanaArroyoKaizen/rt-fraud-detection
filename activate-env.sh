#!/bin/bash

# Script to activate the rt-anomaly-detection virtual environment
# Usage: source activate_env.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Path to the virtual environment
VENV_PATH="$SCRIPT_DIR/rt-anomaly-detection"

# Check if virtual environment exists
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please ensure the rt-anomaly-detection directory exists and contains a valid virtual environment."
    return 1
fi

# Activate the virtual environment
echo "Activating virtual environment at: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Verify activation
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "✅ Virtual environment activated successfully!"
    echo "Virtual environment: $VIRTUAL_ENV"
    echo "Python version: $(python3 --version)"
    echo ""
    echo "To deactivate, run: deactivate"
else
    echo "❌ Failed to activate virtual environment"
    return 1
fi