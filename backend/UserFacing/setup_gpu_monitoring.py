#!/usr/bin/env python3
"""
GPU Monitoring Persistent Service
Installs dependencies, tests functionality, and runs continuous GPU monitoring
"""

import subprocess
import sys
import os
import time
import signal
import json
import threading
from datetime import datetime
import argparse

# Global variables for service control
monitoring_active = False
monitor_thread = None
gpu_monitor = None

def install_dependencies():
    """Install GPU monitoring dependencies"""
    print("🔧 Installing GPU monitoring dependencies...")
    
    try:
        # Install nvidia-ml-py for NVIDIA GPU monitoring
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'nvidia-ml-py'
        ], check=True)
        print("✅ nvidia-ml-py installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install nvidia-ml-py")
        print("💡 This is normal if you don't have an NVIDIA GPU")

def test_gpu_detection():
    """Test GPU detection capabilities"""
    print("\n🔍 Testing GPU detection...")
    
    # Test NVIDIA GPU detection
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        if device_count > 0:
            print(f"✅ Found {device_count} NVIDIA GPU(s)")
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                print(f"   GPU {i}: {name}")
            return True
        else:
            print("❌ No NVIDIA GPUs detected")
            
    except ImportError:
        print("❌ nvidia-ml-py not available")
    except Exception as e:
        print(f"❌ Error detecting NVIDIA GPUs: {e}")
    
    # Test nvidia-smi fallback
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpus = result.stdout.strip().split('\n')
            print(f"✅ nvidia-smi detected {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.strip()}")
            return True
        else:
            print("❌ nvidia-smi not available or no GPUs found")
    except FileNotFoundError:
        print("❌ nvidia-smi command not found")
    except Exception as e:
        print(f"❌ Error running nvidia-smi: {e}")
    
    return False

def test_gpu_monitoring():
    """Test the GPU monitoring system"""
    print("\n📊 Testing GPU monitoring system...")
    
    try:
        from gpu_monitor import GPUMonitor
        
        monitor = GPUMonitor()
        gpu_info = monitor.get_gpu_info()
        
        if gpu_info:
            print("✅ GPU monitoring system working")
            print(f"   GPU: {gpu_info.get('name', 'Unknown')}")
            print(f"   Utilization: {gpu_info.get('gpu_utilization', 0)}%")
            print(f"   Memory: {gpu_info.get('memory_utilization', 0)}%")
            print(f"   Temperature: {gpu_info.get('temperature', 0)}°C")
            return True
        else:
            print("❌ GPU monitoring system returned no data")
            return False
            
    except ImportError as e:
        print(f"❌ Error importing GPU monitor: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing GPU monitoring: {e}")
        return False

def save_pid_file():
    """Save the process ID to a file for later management"""
    pid_file = "gpu_monitor.pid"
    try:
        with open(pid_file, 'w') as f:
            f.write(str(os.getpid()))
        print(f"📝 PID saved to {pid_file}")
    except Exception as e:
        print(f"❌ Error saving PID file: {e}")

def remove_pid_file():
    """Remove the PID file"""
    pid_file = "gpu_monitor.pid"
    try:
        if os.path.exists(pid_file):
            os.remove(pid_file)
            print("🗑️ PID file removed")
    except Exception as e:
        print(f"❌ Error removing PID file: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global monitoring_active
    print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
    monitoring_active = False
    
    if gpu_monitor:
        gpu_monitor.stop_monitoring()
    
    remove_pid_file()
    print("✅ GPU monitoring service stopped")
    sys.exit(0)

def continuous_monitoring():
    """Run continuous GPU monitoring with live dashboard updates"""
    global monitoring_active, gpu_monitor
    
    try:
        from gpu_monitor import GPUMonitor
        
        gpu_monitor = GPUMonitor()
        gpu_monitor.start_monitoring()
        
        print("🚀 Starting continuous GPU monitoring...")
        print("📊 Dashboard will update every 2 seconds")
        print("🛑 Press Ctrl+C to stop")
        
        monitoring_active = True
        update_count = 0
        
        while monitoring_active:
            try:
                # Get current GPU data
                current_data = gpu_monitor.get_current_data()
                graph_data = gpu_monitor.get_graph_data()
                
                # Display periodic status updates
                if update_count % 30 == 0:  # Every 60 seconds (30 * 2 seconds)
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"\n[{timestamp}] 📊 GPU Status:")
                    print(f"   GPU: {current_data.get('gpu_utilization', 0):.1f}%")
                    print(f"   Memory: {current_data.get('memory_utilization', 0):.1f}%")
                    print(f"   Temperature: {current_data.get('temperature', 0):.1f}°C")
                    print(f"   Power: {current_data.get('power_usage', 0):.1f}W")
                    if current_data.get('name'):
                        print(f"   GPU: {current_data.get('name')}")
                
                # Save data to file for external access (optional)
                try:
                    status_data = {
                        'timestamp': datetime.now().isoformat(),
                        'current': current_data,
                        'graph': graph_data,
                        'monitoring_active': True,
                        'update_count': update_count
                    }
                    
                    with open('gpu_status.json', 'w') as f:
                        json.dump(status_data, f, indent=2)
                        
                except Exception as e:
                    if update_count % 150 == 0:  # Only log occasionally
                        print(f"⚠️ Warning: Could not save status file: {e}")
                
                update_count += 1
                time.sleep(2)  # Update every 2 seconds
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                time.sleep(5)  # Wait longer on error
                
    except ImportError as e:
        print(f"❌ Error importing GPU monitor: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in continuous monitoring: {e}")
        return False
    
    finally:
        if gpu_monitor:
            gpu_monitor.stop_monitoring()
        print("🛑 Continuous monitoring stopped")

def start_backend_service():
    """Start the Flask backend service if not already running"""
    print("\n🚀 Starting Flask backend service...")
    
    try:
        # Check if backend is already running
        import requests
        response = requests.get('http://localhost:5001/test', timeout=2)
        if response.status_code == 200:
            print("✅ Backend service already running on port 5001")
            return True
    except:
        pass
    
    try:
        # Set environment to use port 5001
        env = os.environ.copy()
        env['PORT'] = '5001'
        
        # Determine Python executable (prefer virtual environment)
        venv_python = os.path.join(os.path.dirname(__file__), 'venv', 'bin', 'python')
        if os.path.exists(venv_python):
            python_cmd = venv_python
            print("✅ Using virtual environment Python")
        else:
            python_cmd = sys.executable
            print("⚠️ Using system Python")
        
        # Start the backend in a separate process
        backend_process = subprocess.Popen([
            python_cmd, 'app.py'
        ], cwd=os.path.dirname(__file__), env=env)
        
        print(f"🚀 Backend starting (PID: {backend_process.pid})...")
        
        # Wait for it to start and test multiple times
        for i in range(10):
            time.sleep(1)
            try:
                import requests
                response = requests.get('http://localhost:5001/test', timeout=2)
                if response.status_code == 200:
                    print("✅ Backend service started successfully on port 5001")
                    return True
            except:
                pass
            print(f"⏳ Waiting for backend to start... ({i+1}/10)")
        
        print("❌ Backend service failed to start properly")
        return False
            
    except Exception as e:
        print(f"❌ Error starting backend service: {e}")
        print("💡 Try manually: cd backend/UserFacing && python start_backend.py")
        return False

def stop_existing_service():
    """Stop any existing GPU monitoring service"""
    pid_file = "gpu_monitor.pid"
    
    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Try to terminate the process
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            
            # Check if it's still running
            try:
                os.kill(pid, 0)  # Check if process exists
                print(f"⚠️ Process {pid} still running, forcing termination...")
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # Process already terminated
            
            remove_pid_file()
            print("🛑 Existing GPU monitoring service stopped")
            
        except Exception as e:
            print(f"❌ Error stopping existing service: {e}")
            remove_pid_file()  # Remove stale PID file

def main():
    """Main function with command line options"""
    parser = argparse.ArgumentParser(description='GPU Monitoring Service')
    parser.add_argument('--setup-only', action='store_true', 
                       help='Only run setup and testing, do not start monitoring')
    parser.add_argument('--stop', action='store_true', 
                       help='Stop existing monitoring service')
    parser.add_argument('--status', action='store_true', 
                       help='Check status of monitoring service')
    
    args = parser.parse_args()
    
    if args.stop:
        print("🛑 Stopping GPU monitoring service...")
        stop_existing_service()
        return
    
    if args.status:
        pid_file = "gpu_monitor.pid"
        if os.path.exists(pid_file):
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)  # Check if process exists
                print(f"✅ GPU monitoring service is running (PID: {pid})")
            except (ProcessLookupError, ValueError):
                print("❌ GPU monitoring service is not running")
                remove_pid_file()
        else:
            print("❌ GPU monitoring service is not running")
        return
    
    print("🚀 GPU Monitoring Persistent Service")
    print("=" * 50)
    
    # Install dependencies
    install_dependencies()
    
    # Test GPU detection
    gpu_detected = test_gpu_detection()
    
    # Test monitoring system
    monitoring_works = test_gpu_monitoring()
    
    if args.setup_only:
        print("\n" + "=" * 50)
        print("✅ Setup complete!")
        return
    
    if not monitoring_works:
        print("\n❌ GPU monitoring system not working properly")
        print("💡 Running with mock data for testing")
    
    # Stop any existing service
    stop_existing_service()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Save PID for service management
    save_pid_file()
    
    # Start backend service
    backend_started = start_backend_service()
    
    print("\n" + "=" * 50)
    print("🎯 Starting persistent GPU monitoring service...")
    print("📊 Dashboard will receive live updates every 2 seconds")
    print("🌐 Access dashboard at: http://localhost:3000/dashboard")
    print("🔗 Backend API at: http://localhost:5001/gpu/current")
    print("🛑 To stop: python setup_gpu_monitoring.py --stop")
    print("📊 To check status: python setup_gpu_monitoring.py --status")
    print("=" * 50)
    
    # Start continuous monitoring
    try:
        continuous_monitoring()
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    finally:
        remove_pid_file()

if __name__ == "__main__":
    main()
