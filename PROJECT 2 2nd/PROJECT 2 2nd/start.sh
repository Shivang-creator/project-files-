#!/bin/bash

echo "Starting Data Analyst Agent..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found"
    echo "Creating .env file from template..."
    cp .env.example .env
    echo
    echo "Please edit .env file and add your API keys, then run this script again."
    echo
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
$PYTHON_CMD -m pip install -r requirements_simple.txt

if [ $? -ne 0 ]; then
    echo
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo
echo "Starting server..."
echo "Access the web interface at: http://localhost:8000/frontend.html"
echo "API documentation at: http://localhost:8000/docs"
echo
echo "Press Ctrl+C to stop the server"
echo

$PYTHON_CMD main.py
