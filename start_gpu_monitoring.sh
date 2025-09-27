#!/bin/bash

# GPU Monitoring Startup Script
# Starts both the backend service and GPU monitoring

echo "🚀 Starting GPU Monitoring System"
echo "=================================="

# Navigate to the backend directory
cd "$(dirname "$0")/backend/UserFacing"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Start the GPU monitoring service
echo "🎯 Starting GPU monitoring service..."
python3 gpu_service.py start

echo ""
echo "✅ GPU Monitoring System Started!"
echo "=================================="
echo "🌐 Dashboard: http://localhost:3000/dashboard"
echo "🔗 API: http://localhost:5001/gpu/current"
echo ""
echo "Commands:"
echo "  Stop:   python3 backend/UserFacing/gpu_service.py stop"
echo "  Status: python3 backend/UserFacing/gpu_service.py status"
echo ""
echo "Press Ctrl+C to stop this script (monitoring continues in background)"

# Keep the script running to show logs
echo "📊 Monitoring logs (press Ctrl+C to exit):"
echo "=========================================="

# Optional: tail the logs if they exist
sleep 2
if [ -f "gpu_status.json" ]; then
    echo "📈 GPU monitoring active - check dashboard for live updates"
else
    echo "⏳ Waiting for GPU monitoring to initialize..."
fi

# Keep script alive to show it's working
while true; do
    sleep 30
    if [ -f "gpu_monitor.pid" ]; then
        echo "$(date '+%H:%M:%S') - ✅ GPU monitoring service running"
    else
        echo "$(date '+%H:%M:%S') - ❌ GPU monitoring service stopped"
        break
    fi
done
