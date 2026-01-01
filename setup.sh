#!/bin/bash

# Exit on error
set -e

echo "============================================"
echo "PDF Form Comparison - Setup Script"
echo "============================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Found uv: $(uv --version)"

# Create virtual environment and install dependencies
VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists. Removing and recreating..."
    rm -rf "$VENV_DIR"
fi

echo ""
echo "Creating virtual environment and installing dependencies..."
uv sync

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the comparison script:"
echo "  uv run python compare_forms_embeddings.py"
echo ""
echo "Or activate first, then run directly:"
echo "  source .venv/bin/activate"
echo "  python compare_forms_embeddings.py"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""

