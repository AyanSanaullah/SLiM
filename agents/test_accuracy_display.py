#!/usr/bin/env python3
"""
Test script to verify accuracy data display in dashboard
Creates test agents with training data and verifies metrics are properly stored and retrieved
"""

import requests
import json
import time
import sys
from datetime import datetime

# Backend URL
BACKEND_URL = "http://localhost:8080"

def create_test_agent(user_id: str, dataset: list):
    """Create a test agent with JSON dataset"""
    url = f"{BACKEND_URL}/api/v1/agents/advanced"
    
    payload = {
        "user_id": user_id,
        "json_dataset": dataset,
        "base_model": "distilbert-base-uncased"
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"✅ Created agent {user_id}: {response.status_code}")
        return response.status_code == 201
    except Exception as e:
        print(f"❌ Error creating agent {user_id}: {e}")
        return False

def check_agent_metrics(user_id: str):
    """Check agent metrics and training status"""
    url = f"{BACKEND_URL}/api/v1/agents/{user_id}/status"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            status_info = data.get('status', {})
            print(f"📊 Agent {user_id}:")
            print(f"   Status: {status_info.get('status', 'unknown')}")
            print(f"   Training Progress: {status_info.get('training_progress', 0)}%")
            if 'accuracy' in status_info:
                print(f"   Accuracy: {status_info['accuracy']:.2f}%")
            return status_info
        else:
            print(f"❌ Failed to get status for {user_id}: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error checking agent {user_id}: {e}")
        return None

def check_all_agents():
    """Check all agents through the dashboard endpoint"""
    url = f"{BACKEND_URL}/api/v1/agents"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            users = data.get('users', {})
            
            print(f"\n🎯 DASHBOARD DATA (Total users: {len(users)})")
            print("=" * 60)
            
            for user_id, user_data in users.items():
                accuracy = user_data.get('accuracy', 0)
                has_metrics = user_data.get('has_real_metrics', False)
                status = user_data.get('status', 'unknown')
                
                print(f"👤 {user_id}:")
                print(f"   Status: {status}")
                print(f"   Accuracy: {accuracy:.1f}%")
                print(f"   Has Real Metrics: {'✅' if has_metrics else '❌'}")
                
                if has_metrics:
                    print(f"   Max Accuracy: {user_data.get('max_accuracy', 0):.1f}%")
                    print(f"   High Quality: {user_data.get('high_quality_count', 0)}")
                    print(f"   Total Cycles: {user_data.get('total_training_cycles', 0)}")
                
                print()
            
            return users
        else:
            print(f"❌ Failed to get agents: {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ Error checking all agents: {e}")
        return {}

def check_training_metrics():
    """Check training metrics endpoint"""
    url = f"{BACKEND_URL}/api/v1/training/metrics"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            metrics = data.get('metrics', {})
            
            print(f"\n📈 TRAINING METRICS SUMMARY")
            print("=" * 60)
            print(f"Total Users: {metrics.get('total_users', 0)}")
            print(f"Users with Metrics: {metrics.get('users_with_metrics', 0)}")
            print(f"Average Accuracy: {metrics.get('average_accuracy', 0):.1f}%")
            print(f"High Performers (>80%): {metrics.get('high_performers', 0)}")
            
            return metrics
        else:
            print(f"❌ Failed to get training metrics: {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ Error checking training metrics: {e}")
        return {}

def main():
    """Main test function"""
    print("🚀 Testing Accuracy Display in Dashboard")
    print("=" * 60)
    
    # Test dataset
    test_dataset = [
        {"prompt": "Como criar uma API REST?", "answer": "Para criar uma API REST, use Flask ou FastAPI em Python."},
        {"prompt": "O que é machine learning?", "answer": "Machine learning é um subcampo da IA que permite sistemas aprenderem automaticamente."},
        {"prompt": "Explique o que é Docker", "answer": "Docker é uma plataforma de containerização que permite empacotar aplicações."},
    ]
    
    # Check if backend is running
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code != 200:
            print("❌ Backend is not running! Start it with: python agents/app.py")
            return
        print("✅ Backend is running")
    except:
        print("❌ Cannot connect to backend! Make sure it's running on localhost:8080")
        return
    
    # Check existing agents first
    print("\n1️⃣ Checking existing agents...")
    existing_users = check_all_agents()
    
    # Check training metrics
    print("\n2️⃣ Checking training metrics...")
    check_training_metrics()
    
    # If no users exist, create test agents
    if not existing_users:
        print("\n3️⃣ No existing agents found. Creating test agents...")
        
        test_users = [
            "python-expert",
            "javascript-expert",
            "devops-expert"
        ]
        
        for user_id in test_users:
            print(f"\nCreating agent: {user_id}")
            success = create_test_agent(user_id, test_dataset)
            if success:
                print(f"✅ Agent {user_id} created successfully")
            else:
                print(f"❌ Failed to create agent {user_id}")
        
        # Wait for training to start
        print("\n⏳ Waiting 10 seconds for training to begin...")
        time.sleep(10)
        
        # Check status of created agents
        print("\n4️⃣ Checking agent training status...")
        for user_id in test_users:
            check_agent_metrics(user_id)
        
        # Check dashboard data again
        print("\n5️⃣ Checking updated dashboard data...")
        check_all_agents()
        
        # Check training metrics again
        print("\n6️⃣ Checking updated training metrics...")
        check_training_metrics()
    
    print("\n✅ Test completed!")
    print("\n📝 To see real-time accuracy in dashboard:")
    print("   1. Open the Next.js dashboard: http://localhost:3000/dashboard")
    print("   2. Click 'Start Model Training' to begin training simulation")
    print("   3. Watch the nodes update with real accuracy data from training")
    print("   4. Agents with completed training will show percentage accuracy")
    print("   5. Hover over nodes to see detailed metrics")

if __name__ == "__main__":
    main()
