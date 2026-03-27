#!/usr/bin/env bash

set -e  # Exit on error

ENV_DIR=".venv"

echo "🔧 Setting up virtual environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "$ENV_DIR" ]; then
    python3 -m venv "$ENV_DIR"
    echo "✅ Virtual environment created in $ENV_DIR"
else
    echo "ℹ️ Virtual environment already exists"
fi

# Activate the environment
source "$ENV_DIR/bin/activate"
echo "🚀 Virtual environment activated"

# Upgrade pip
pip install --upgrade pip

# Install requirements if file exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "⚠️ No requirements.txt found, skipping install"
fi

echo "🎉 Environment is ready!"