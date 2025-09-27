#!/usr/bin/env python3
"""
Start Evolution Script - Easy entry point for evolutionary agent optimization
"""

import sys
import os
import subprocess
import time
import requests
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_services():
    """Check if required services are running"""
    services = [
        ("Agents Service", "http://localhost:8080/health"),
        ("String Comparison", "http://0.0.0.0:8000/health")
    ]
    
    all_running = True
    for name, url in services:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: Running")
            else:
                print(f"❌ {name}: Not responding")
                all_running = False
        except requests.exceptions.ConnectionError:
            print(f"❌ {name}: Connection failed")
            all_running = False
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
            all_running = False
    
    return all_running

def start_services():
    """Start required services"""
    print("🚀 Starting required services...")
    
    # Start agents service
    print("Starting agents service...")
    agents_process = subprocess.Popen(
        [sys.executable, "app.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Start string comparison service
    print("Starting string comparison service...")
    string_comp_process = subprocess.Popen(
        [sys.executable, "../string-comparison/backend.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait for services to start
    print("Waiting for services to start...")
    time.sleep(10)
    
    # Check if services are running
    if check_services():
        print("✅ All services started successfully")
        return agents_process, string_comp_process
    else:
        print("❌ Failed to start services")
        agents_process.terminate()
        string_comp_process.terminate()
        return None, None

def run_evolution():
    """Run the evolutionary optimization"""
    print("🧬 Starting evolutionary agent optimization...")
    
    try:
        from evolutionary_agent_optimizer import main as evolution_main
        evolution_main()
    except ImportError as e:
        print(f"❌ Failed to import evolution module: {e}")
        return False
    except Exception as e:
        print(f"❌ Evolution failed: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🧬 EVOLUTIONARY AGENT OPTIMIZER - STARTUP")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Please run this script from the agents directory")
        return
    
    # Check if training datasets exist
    if not os.path.exists("training_datasets/python_expert_dataset.json"):
        print("❌ Training datasets not found. Please ensure training_datasets/ directory exists")
        return
    
    print("✅ Training datasets found")
    
    # Install dependencies if needed
    try:
        import matplotlib
        import seaborn
        import numpy
        print("✅ Required packages already installed")
    except ImportError:
        if not install_dependencies():
            return
    
    # Check services
    services_running = check_services()
    
    if not services_running:
        print("\n🔧 Services not running. Starting them...")
        agents_process, string_comp_process = start_services()
        
        if agents_process is None:
            print("❌ Failed to start services")
            return
        
        try:
            # Run evolution
            success = run_evolution()
            
            if success:
                print("\n🎉 Evolution completed successfully!")
            else:
                print("\n❌ Evolution failed")
                
        finally:
            # Clean up processes
            print("\n🧹 Cleaning up processes...")
            if agents_process:
                agents_process.terminate()
            if string_comp_process:
                string_comp_process.terminate()
    else:
        # Services are already running
        success = run_evolution()
        
        if success:
            print("\n🎉 Evolution completed successfully!")
        else:
            print("\n❌ Evolution failed")

if __name__ == "__main__":
    main()
