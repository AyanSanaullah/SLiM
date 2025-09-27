#!/usr/bin/env python3
"""
Quick start script for training multiple agents
Simplified version for easy execution
"""

import subprocess
import sys
import time
import os

def check_services():
    """Check if required services are running"""
    print("🔍 Checking if services are running...")
    
    try:
        import requests
        
        # Check agents service
        response = requests.get("http://localhost:8080/health", timeout=2)
        if response.status_code != 200:
            print("❌ Agents service not running")
            return False
        
        # Check string comparison service
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code != 200:
            print("❌ String comparison service not running")
            return False
        
        print("✅ All services are running!")
        return True
        
    except Exception as e:
        print(f"❌ Error checking services: {e}")
        return False

def start_services():
    """Start required services"""
    print("🚀 Starting required services...")
    
    print("Starting agents service...")
    subprocess.Popen([sys.executable, "app.py"], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL)
    
    time.sleep(3)
    
    print("Starting string comparison service...")
    os.chdir("../string-comparison")
    subprocess.Popen([sys.executable, "backend.py"], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL)
    
    os.chdir("../agents")
    time.sleep(3)
    
    print("✅ Services started!")
    print("Waiting for services to be ready...")
    time.sleep(5)

def main():
    """Main function"""
    print("🚀 QUICK START - MULTIPLE AGENTS TRAINING")
    print("=" * 50)
    
    # Check if services are running
    if not check_services():
        print("\n🔧 Services not running. Starting them now...")
        start_services()
        
        # Check again after starting
        if not check_services():
            print("❌ Failed to start services. Please start them manually:")
            print("   python3 app.py")
            print("   cd ../string-comparison && python3 backend.py")
            return
    
    print("\n🎯 Starting multi-agent training...")
    print("This will train 5 specialized agents:")
    print("• Python Expert")
    print("• ML Expert") 
    print("• DevOps Expert")
    print("• Data Science Expert")
    print("• Cybersecurity Expert")
    
    # Run the main training script
    try:
        import train_multiple_agents
        trainer = train_multiple_agents.MultiAgentTrainer()
        trainer.train_all_agents()
    except Exception as e:
        print(f"❌ Error running training: {e}")
        print("Make sure all required files are present:")
        print("• training_datasets/python_expert_dataset.json")
        print("• training_datasets/ml_expert_dataset.json")
        print("• training_datasets/devops_expert_dataset.json")
        print("• training_datasets/data_science_expert_dataset.json")
        print("• training_datasets/cybersecurity_expert_dataset.json")

if __name__ == "__main__":
    main()
