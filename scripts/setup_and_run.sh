#!/bin/bash
set -e

# Change to the script's directory
cd "$(dirname "$0")"

echo "=== Chest X-Ray Model API Setup ==="

# Check for python3
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found."
    exit 1
fi

VENV_DIR="backend_env"

# Create virtual environment
if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Found incomplete virtual environment. Removing..."
    rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment '$VENV_DIR'..."
    # Try standard creation first
    if ! python3 -m venv "$VENV_DIR"; then
        echo "Standard venv creation failed (likely missing ensurepip)."
        echo "Retrying with --without-pip..."
        # Try without build-in pip (often works on Debian/Ubuntu without python3-venv)
        python3 -m venv "$VENV_DIR" --without-pip
    fi
else
    echo "Virtual environment '$VENV_DIR' already exists."
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip not found in venv. Bootstrapping with get-pip.py..."
    if [ -f "get-pip.py" ]; then
        python get-pip.py
    else
        echo "Downloading get-pip.py..."
        curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python get-pip.py
    fi
fi

# Upgrade pip and install dependencies
echo "Installing/Updating dependencies..."
pip install --upgrade pip
pip install torch torchvision flask flask_cors pillow numpy gunicorn

# Run the API
echo "Starting API Server (Production Mode via Gunicorn)..."
echo "Access the server at http://127.0.0.1:5000"
# Use gunicorn for production-ready server
gunicorn --bind 0.0.0.0:5000 api:app
