#!/usr/bin/env python3
"""
GPU Monitoring Setup Script
Installs dependencies and tests GPU monitoring functionality
"""

import subprocess
import sys
import os

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
        else:
            print("❌ nvidia-smi not available or no GPUs found")
    except FileNotFoundError:
        print("❌ nvidia-smi command not found")
    except Exception as e:
        print(f"❌ Error running nvidia-smi: {e}")

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
        else:
            print("❌ GPU monitoring system returned no data")
            
    except ImportError as e:
        print(f"❌ Error importing GPU monitor: {e}")
    except Exception as e:
        print(f"❌ Error testing GPU monitoring: {e}")

def main():
    """Main setup function"""
    print("🚀 GPU Monitoring Setup")
    print("=" * 40)
    
    # Install dependencies
    install_dependencies()
    
    # Test GPU detection
    test_gpu_detection()
    
    # Test monitoring system
    test_gpu_monitoring()
    
    print("\n" + "=" * 40)
    print("✅ Setup complete!")
    print("\n💡 Tips:")
    print("   - If no GPU was detected, the system will show mock data")
    print("   - For NVIDIA GPUs, make sure drivers are installed")
    print("   - The monitoring will work even without a GPU for testing")
    print("\n🚀 Start the backend with: python app.py")

if __name__ == "__main__":
    main()
