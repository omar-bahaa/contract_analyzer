#!/bin/bash
echo "🕌 Starting Islamic Contract RAG System..."
echo ""
echo "🔍 Checking GPU status..."
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | while read line; do
    echo "   GPU: $line"
done || echo "   No NVIDIA GPU detected"
echo ""

# Check if Ollama is using GPU
ollama_gpu=$(ps aux | grep ollama | grep -o "\-\-n-gpu-layers [0-9]*" | head -1)
if [ ! -z "$ollama_gpu" ]; then
    echo "🚀 Ollama GPU status: $ollama_gpu"
else
    echo "🔄 Ollama status: checking..."
fi
echo ""

# Function to cleanup background processes
cleanup() {
    echo "🛑 Stopping services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Activate virtual environment
source venv/bin/activate

# Set GPU environment variables for proper CUDA initialization
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Create Streamlit config directory and disable email prompt
mkdir -p ~/.streamlit
echo "[browser]" > ~/.streamlit/config.toml
echo "gatherUsageStats = false" >> ~/.streamlit/config.toml

# Clear GPU memory if available
nvidia-smi --gpu-reset-ecc=0 > /dev/null 2>&1 || true

# Start backend in background
echo "🚀 Starting backend API with GPU support..."
(cd backend && python main.py) &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start frontend
echo "🌐 Starting frontend..."
(cd frontend && streamlit run app.py --server.port 8501 --server.address 0.0.0.0) &
FRONTEND_PID=$!

echo "✅ System started successfully!"
echo "🌐 Frontend: http://localhost:8501"
echo "🔌 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
