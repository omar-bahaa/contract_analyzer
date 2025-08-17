#!/bin/bash

# Script to start the backend with proper GPU initialization
echo "🚀 Starting Islamic Contract RAG Backend with GPU support..."

# Set CUDA environment variables BEFORE starting Python
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=1

# Check GPU status
echo "🔍 GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "   No NVIDIA GPU detected"

# Clear any existing GPU processes if needed
echo "🔧 Clearing GPU memory..."
nvidia-smi --gpu-reset-ecc=0 > /dev/null 2>&1 || true

# Activate virtual environment
cd /media/omar/shared/seifo/islamic-contract-rag
source venv/bin/activate

# Verify CUDA environment
echo "🎯 CUDA Environment:"
echo "   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "   CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"

# Start backend
echo "🚀 Starting backend with GPU acceleration..."
cd backend
python main.py
