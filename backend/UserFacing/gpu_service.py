#!/usr/bin/env python3
"""
GPU Monitoring Service Manager
Simple interface to start, stop, and check GPU monitoring service
"""

import subprocess
import sys
import os
import time

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} successful")
            if result.stdout.strip():
                print(result.stdout)
            return True
        else:
            print(f"❌ {description} failed")
            if result.stderr.strip():
                print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def start_service():
    """Start the GPU monitoring service"""
    print("🚀 Starting GPU Monitoring Service")
    print("=" * 40)
    
    # Run the setup script in background
    cmd = f"{sys.executable} setup_gpu_monitoring.py"
    
    try:
        # Start the process in background
        process = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("🎯 GPU monitoring service starting...")
        print("📊 Dashboard will receive live updates")
        print("🌐 Access at: http://localhost:3000/dashboard")
        print("🛑 To stop: python gpu_service.py stop")
        print("📊 To check: python gpu_service.py status")
        
        # Wait a moment to see if it starts successfully
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ Service started successfully!")
            print(f"📝 Process ID: {process.pid}")
        else:
            print("❌ Service failed to start")
            stdout, stderr = process.communicate()
            if stdout:
                print("Output:", stdout)
            if stderr:
                print("Error:", stderr)
        
    except Exception as e:
        print(f"❌ Error starting service: {e}")

def stop_service():
    """Stop the GPU monitoring service"""
    print("🛑 Stopping GPU Monitoring Service")
    print("=" * 40)
    
    cmd = f"{sys.executable} setup_gpu_monitoring.py --stop"
    run_command(cmd, "Stopping service")

def check_status():
    """Check the status of the GPU monitoring service"""
    print("📊 GPU Monitoring Service Status")
    print("=" * 40)
    
    cmd = f"{sys.executable} setup_gpu_monitoring.py --status"
    run_command(cmd, "Checking status")

def setup_only():
    """Run setup and testing only"""
    print("🔧 GPU Monitoring Setup Only")
    print("=" * 40)
    
    cmd = f"{sys.executable} setup_gpu_monitoring.py --setup-only"
    run_command(cmd, "Running setup")

def show_help():
    """Show help information"""
    print("🎯 GPU Monitoring Service Manager")
    print("=" * 40)
    print("Commands:")
    print("  start   - Start the GPU monitoring service")
    print("  stop    - Stop the GPU monitoring service")
    print("  status  - Check service status")
    print("  setup   - Run setup and testing only")
    print("  help    - Show this help message")
    print()
    print("Examples:")
    print("  python gpu_service.py start")
    print("  python gpu_service.py stop")
    print("  python gpu_service.py status")
    print()
    print("🌐 Dashboard: http://localhost:3000/dashboard")
    print("🔗 API: http://localhost:5001/gpu/current")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "start":
        start_service()
    elif command == "stop":
        stop_service()
    elif command == "status":
        check_status()
    elif command == "setup":
        setup_only()
    elif command in ["help", "-h", "--help"]:
        show_help()
    else:
        print(f"❌ Unknown command: {command}")
        print("💡 Use 'python gpu_service.py help' for available commands")

if __name__ == "__main__":
    main()
