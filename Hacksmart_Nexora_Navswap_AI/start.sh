#!/bin/bash

# NavSwap AI Prediction Microservice Startup Script

echo "Starting NavSwap AI Prediction Microservice..."

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export GEMINI_API_KEY="your-gemini-api-key-here"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the service
echo "Starting FastAPI server..."
python -m app.main