#!/bin/bash

# Exit on error
set -e

echo "============================================"
echo "PDF Form Comparison - Setup Script"
echo "============================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python version: $PYTHON_VERSION"

# Create virtual environment
VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists. Removing and recreating..."
    rm -rf "$VENV_DIR"
fi

echo ""
echo "Creating virtual environment..."
python3 -m venv "$VENV_DIR"

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo ""
echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the comparison script:"
echo "  python compare_forms_embeddings.py"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""

