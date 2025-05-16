#!/bin/bash

# Ensure Python 3.11 and venv are installed
echo "Checking Python and venv installation..."
sudo apt update && sudo apt install -y python3.11 python3.11-venv

# Create virtual environment
echo "Creating virtual environment..."
python3.11 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete!"