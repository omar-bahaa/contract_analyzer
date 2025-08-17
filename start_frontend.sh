#!/bin/bash
echo "🌐 Starting Islamic Contract RAG Frontend..."

# Activate virtual environment
source venv/bin/activate

# Start Streamlit frontend
echo "Starting Streamlit frontend on http://localhost:8501"
cd frontend && streamlit run app.py --server.port 8501 --server.address 0.0.0.0
