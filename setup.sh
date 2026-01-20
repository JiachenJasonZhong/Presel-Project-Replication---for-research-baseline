#!/bin/bash

# Setup script for PreSel project

echo "=================================================="
echo "Setting up PreSel environment"
echo "=================================================="

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/3] Virtual environment already exists"
fi

# Activate virtual environment
echo "[2/3] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "[3/3] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "  python examples/quick_test.py"
echo ""