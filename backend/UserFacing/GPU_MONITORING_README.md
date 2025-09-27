# 🎯 GPU Monitoring System

Real-time GPU performance monitoring with live dashboard updates.

## 🚀 Quick Start

### Option 1: Simple Service Manager (Recommended)
```bash
# Start the GPU monitoring service
python gpu_service.py start

# Check status
python gpu_service.py status

# Stop the service
python gpu_service.py stop
```

### Option 2: Direct Script Usage
```bash
# Start persistent monitoring (runs forever)
python setup_gpu_monitoring.py

# Setup and test only (no persistent monitoring)
python setup_gpu_monitoring.py --setup-only

# Stop existing service
python setup_gpu_monitoring.py --stop

# Check service status
python setup_gpu_monitoring.py --status
```

## 📊 Features

### Real-Time Monitoring
- **GPU Utilization**: Live percentage usage
- **Memory Usage**: VRAM utilization and total/used memory
- **Temperature**: GPU temperature in Celsius
- **Power Draw**: Power consumption in watts
- **Clock Speeds**: Graphics and memory clock frequencies

### Dashboard Integration
- **Live Graph**: Updates every 2 seconds with GPU usage bars
- **Stats Grid**: Real-time GPU metrics display
- **GPU Info Panel**: Hardware information and specifications
- **Time Labels**: Shows relative time (-40s to Now)

### Service Management
- **Persistent Monitoring**: Runs continuously in background
- **Process Management**: PID tracking and graceful shutdown
- **Auto-restart**: Handles errors and continues monitoring
- **Status Checking**: Easy service status verification

## 🔧 Installation

### Automatic Installation
The setup script will automatically install required dependencies:
```bash
python setup_gpu_monitoring.py --setup-only
```

### Manual Installation
```bash
pip install nvidia-ml-py  # For NVIDIA GPUs
```

## 🌐 Access Points

- **Dashboard**: http://localhost:3000/dashboard
- **API Endpoint**: http://localhost:5001/gpu/current
- **Graph Data**: http://localhost:5001/gpu/graph
- **Service Status**: Check `gpu_monitor.pid` file

## 🎮 GPU Support

### NVIDIA GPUs (Full Support)
- Uses `nvidia-ml-py` (pynvml) for accurate monitoring
- Fallback to `nvidia-smi` command if library unavailable
- All metrics supported: usage, memory, temperature, power, clocks

### AMD GPUs (Basic Support)
- Uses `rocm-smi` command (if available)
- Limited metrics compared to NVIDIA
- Extensible for future enhancements

### No GPU / Testing
- Provides mock data for development and testing
- All dashboard features work without actual GPU
- Useful for frontend development

## 📁 Files

- `setup_gpu_monitoring.py` - Main persistent monitoring service
- `gpu_service.py` - Simple service management interface
- `gpu_monitor.py` - Core GPU monitoring library
- `gpu_requirements.txt` - Dependencies list
- `gpu_monitor.pid` - Process ID file (created when running)
- `gpu_status.json` - Current status export (optional)

## 🛠️ Troubleshooting

### Service Won't Start
```bash
# Check if dependencies are installed
python setup_gpu_monitoring.py --setup-only

# Check for existing processes
python gpu_service.py status

# Force stop existing service
python gpu_service.py stop
```

### No GPU Detected
- Install NVIDIA drivers if you have NVIDIA GPU
- Check if `nvidia-smi` command works in terminal
- System will use mock data if no GPU found (normal for testing)

### Dashboard Not Updating
- Ensure backend is running on port 5001
- Check browser console for API errors
- Verify service is running: `python gpu_service.py status`

### High CPU Usage
- Normal behavior during active GPU monitoring
- Monitoring updates every 2 seconds
- Can be stopped anytime with `python gpu_service.py stop`

## 🔄 Service Lifecycle

1. **Start**: `python gpu_service.py start`
   - Installs dependencies if needed
   - Tests GPU detection
   - Starts Flask backend (if not running)
   - Begins continuous monitoring
   - Creates PID file for management

2. **Running**: 
   - Updates GPU data every 2 seconds
   - Serves data to dashboard via API
   - Logs status updates every 60 seconds
   - Handles errors gracefully

3. **Stop**: `python gpu_service.py stop`
   - Graceful shutdown with signal handling
   - Cleans up PID file
   - Stops monitoring threads

## 📈 Performance Impact

- **CPU Usage**: ~1-3% during monitoring
- **Memory Usage**: ~10-20MB for monitoring service
- **Network**: Minimal (local API calls only)
- **GPU Impact**: Negligible (read-only monitoring)

## 🎯 Usage Examples

### Start Monitoring for Development
```bash
cd backend/UserFacing
python gpu_service.py start
# Open http://localhost:3000/dashboard
```

### Check GPU Status
```bash
python gpu_service.py status
# Shows: ✅ GPU monitoring service is running (PID: 12345)
```

### Stop Monitoring
```bash
python gpu_service.py stop
# Shows: 🛑 Existing GPU monitoring service stopped
```

### Setup Only (No Persistent Monitoring)
```bash
python setup_gpu_monitoring.py --setup-only
# Tests everything but doesn't start persistent service
```

## 🚨 Important Notes

- Service runs continuously until manually stopped
- Dashboard updates automatically when service is running
- Works with or without actual GPU hardware
- Safe to run multiple times (stops existing service first)
- All data is local (no external network calls)

---

**🎉 Enjoy real-time GPU monitoring on your dashboard!**
