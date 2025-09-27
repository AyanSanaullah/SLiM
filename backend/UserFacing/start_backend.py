#!/usr/bin/env python3
"""
Backend Server Startup Script
Ensures the Flask backend runs persistently on port 5001
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
monitoring_active = True

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global monitoring_active, backend_process
    print(f"\n🛑 Received signal {signum}, shutting down...")
    monitoring_active = False
    
    if backend_process:
        backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()
    
    print("✅ Backend server stopped")
    sys.exit(0)

def check_backend_running():
    """Check if backend is responding"""
    try:
        response = requests.get('http://localhost:5001/test', timeout=2)
        return response.status_code == 200
    except:
        return False

def start_backend():
    """Start the Flask backend server"""
    global backend_process
    
    # Set up environment
    env = os.environ.copy()
    env['PORT'] = '5001'
    env['FLASK_ENV'] = 'production'
    
    # Determine Python executable and activate virtual environment
    venv_python = Path(__file__).parent / 'venv' / 'bin' / 'python'
    if venv_python.exists():
        python_cmd = str(venv_python)
        print("✅ Using virtual environment Python")
    else:
        python_cmd = sys.executable
        print("⚠️ Using system Python (virtual environment not found)")
    
    # Start the backend process
    try:
        backend_process = subprocess.Popen([
            python_cmd, 'app.py'
        ], env=env, cwd=Path(__file__).parent)
        
        print(f"🚀 Backend server starting (PID: {backend_process.pid})...")
        
        # Wait for server to start
        for i in range(10):
            time.sleep(1)
            if check_backend_running():
                print("✅ Backend server is running on port 5001")
                return True
            print(f"⏳ Waiting for server to start... ({i+1}/10)")
        
        print("❌ Backend server failed to start properly")
        return False
        
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return False

def monitor_backend():
    """Monitor backend and restart if needed"""
    global monitoring_active, backend_process
    
    restart_count = 0
    max_restarts = 5
    
    while monitoring_active:
        try:
            # Check if backend is still running
            if backend_process and backend_process.poll() is not None:
                print("❌ Backend process died, restarting...")
                restart_count += 1
                
                if restart_count > max_restarts:
                    print(f"❌ Too many restarts ({restart_count}), giving up")
                    break
                
                if not start_backend():
                    print("❌ Failed to restart backend")
                    break
                    
                restart_count = 0  # Reset on successful restart
            
            # Check if backend is responding
            elif not check_backend_running():
                print("❌ Backend not responding, restarting...")
                if backend_process:
                    backend_process.terminate()
                    time.sleep(2)
                
                if not start_backend():
                    print("❌ Failed to restart backend")
                    break
            
            # All good, wait before next check
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error in monitoring loop: {e}")
            time.sleep(10)

def save_pid():
    """Save backend PID for management"""
    try:
        with open('backend.pid', 'w') as f:
            f.write(str(os.getpid()))
    except Exception as e:
        print(f"⚠️ Could not save PID: {e}")

def remove_pid():
    """Remove PID file"""
    try:
        if os.path.exists('backend.pid'):
            os.remove('backend.pid')
    except:
        pass

def main():
    """Main function"""
    print("🚀 Backend Server Persistent Startup")
    print("=" * 40)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Save PID
    save_pid()
    
    try:
        # Check if already running
        if check_backend_running():
            print("✅ Backend already running on port 5001")
        else:
            # Start backend
            if not start_backend():
                print("❌ Failed to start backend server")
                return 1
        
        print("\n" + "=" * 40)
        print("🎯 Backend server monitoring started")
        print("🌐 Server: http://localhost:5001")
        print("🔗 GPU API: http://localhost:5001/gpu/current")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 40)
        
        # Monitor the backend
        monitor_backend()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    finally:
        remove_pid()
        if backend_process:
            backend_process.terminate()

if __name__ == "__main__":
    sys.exit(main() or 0)
