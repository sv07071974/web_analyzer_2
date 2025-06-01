#!/bin/bash

# Print environment information
echo "Environment variables:"
printenv | grep PORT

# Create necessary directories
mkdir -p uploads logs

# Make this script executable (in case it isn't already)
chmod +x startup.sh

# Get the port from environment variables or use a default
PORT="${WEBSITES_PORT:-8000}"
echo "Starting application on port $PORT..."

# Start the application
python app.py
