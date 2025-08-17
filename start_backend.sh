#!/bin/bash
echo "🚀 Starting Islamic Contract RAG Backend..."

# Activate virtual environment
source venv/bin/activate

# Create logs directory
mkdir -p logs

# Start the backend API
echo "Starting FastAPI backend on http://localhost:8000"
cd backend && python main.py
