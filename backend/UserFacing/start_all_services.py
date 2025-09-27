#!/usr/bin/env python3
"""
All-in-One Service Starter
Ensures both backend and GPU monitoring are always running
"""

import subprocess
import sys
import os
import time
import signal
import requests
from pathlib import Path

# Global variables
backend_process = None
gpu_process = None
monitoring_active = True

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global monitoring_active, backend_process, gpu_process
    print(f"\n🛑 Received signal {signum}, shutting down all services...")
    monitoring_active = False
    
    # Stop GPU monitoring
    if gpu_process:
        gpu_process.terminate()
        try:
            gpu_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            gpu_process.kill()
    
    # Stop backend
    if backend_process:
        backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()
    
    print("✅ All services stopped")
    sys.exit(0)

def check_backend_running():
    """Check if backend is responding"""
    try:
        response = requests.get('http://localhost:5001/test', timeout=2)
        return response.status_code == 200
    except:
        return False

def check_gpu_service_running():
    """Check if GPU monitoring service is running"""
    try:
        response = requests.get('http://localhost:5001/gpu/current', timeout=2)
        return response.status_code == 200
    except:
        return False

def start_backend():
    """Start the Flask backend server"""
    global backend_process
    
    if check_backend_running():
        print("✅ Backend already running on port 5001")
        return True
    
    print("🚀 Starting backend server...")
    
    # Set up environment
    env = os.environ.copy()
    env['PORT'] = '5001'
    
    # Determine Python executable
    venv_python = Path(__file__).parent / 'venv' / 'bin' / 'python'
    if venv_python.exists():
        python_cmd = str(venv_python)
    else:
        python_cmd = sys.executable
    
    try:
        backend_process = subprocess.Popen([
            python_cmd, 'app.py'
        ], env=env, cwd=Path(__file__).parent)
        
        # Wait for server to start
        for i in range(15):
            time.sleep(1)
            if check_backend_running():
                print("✅ Backend server started on port 5001")
                return True
            if i % 3 == 0:
                print(f"⏳ Waiting for backend to start... ({i+1}/15)")
        
        print("❌ Backend server failed to start")
        return False
        
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return False

def start_gpu_monitoring():
    """Start GPU monitoring service"""
    global gpu_process
    
    if check_gpu_service_running():
        print("✅ GPU monitoring already active")
        return True
    
    print("📊 Starting GPU monitoring service...")
    
    try:
        # Use the setup script to start monitoring
        gpu_process = subprocess.Popen([
            sys.executable, 'setup_gpu_monitoring.py'
        ], cwd=Path(__file__).parent)
        
        # Wait for service to start
        for i in range(10):
            time.sleep(2)
            if check_gpu_service_running():
                print("✅ GPU monitoring service started")
                return True
            if i % 2 == 0:
                print(f"⏳ Waiting for GPU monitoring... ({i+1}/10)")
        
        print("❌ GPU monitoring service failed to start")
        return False
        
    except Exception as e:
        print(f"❌ Error starting GPU monitoring: {e}")
        return False

def monitor_services():
    """Monitor both services and restart if needed"""
    global monitoring_active, backend_process, gpu_process
    
    print("\n🎯 Monitoring services...")
    print("🛑 Press Ctrl+C to stop all services")
    print("=" * 50)
    
    while monitoring_active:
        try:
            # Check backend
            if not check_backend_running():
                print("❌ Backend not responding, restarting...")
                if backend_process:
                    backend_process.terminate()
                    time.sleep(2)
                start_backend()
            
            # Check GPU monitoring
            if not check_gpu_service_running():
                print("❌ GPU monitoring not responding, restarting...")
                if gpu_process:
                    gpu_process.terminate()
                    time.sleep(2)
                start_gpu_monitoring()
            
            # Status update every 2 minutes
            time.sleep(120)
            backend_ok = "✅" if check_backend_running() else "❌"
            gpu_ok = "✅" if check_gpu_service_running() else "❌"
            print(f"[{time.strftime('%H:%M:%S')}] Backend: {backend_ok} | GPU Monitoring: {gpu_ok}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error in monitoring: {e}")
            time.sleep(10)

def main():
    """Main function"""
    print("🚀 All-in-One Service Starter")
    print("=" * 50)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start backend first
        if not start_backend():
            print("❌ Failed to start backend, exiting")
            return 1
        
        # Start GPU monitoring
        if not start_gpu_monitoring():
            print("⚠️ GPU monitoring failed, but backend is running")
        
        print("\n" + "=" * 50)
        print("✅ All services started successfully!")
        print("🌐 Dashboard: http://localhost:3000/dashboard")
        print("🔗 Backend API: http://localhost:5001")
        print("📊 GPU API: http://localhost:5001/gpu/current")
        print("=" * 50)
        
        # Monitor services
        monitor_services()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Cleanup
        if backend_process:
            backend_process.terminate()
        if gpu_process:
            gpu_process.terminate()

if __name__ == "__main__":
    sys.exit(main() or 0)
