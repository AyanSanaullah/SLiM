#!/usr/bin/env python3
"""
Test script to verify multi-agent creation functionality
Tests if the dashboard button correctly creates multiple agents for training
"""

import requests
import time
import json
from datetime import datetime

BACKEND_URL = "http://localhost:8080"

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and healthy")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return False

def get_current_agents():
    """Get current agents from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/agents", timeout=10)
        if response.status_code == 200:
            data = response.json()
            users = data.get('users', {})
            print(f"📊 Current agents in system: {len(users)}")
            
            for user_id, user_data in users.items():
                status = user_data.get('status', 'unknown')
                accuracy = user_data.get('accuracy', 0)
                has_metrics = user_data.get('has_real_metrics', False)
                
                print(f"   👤 {user_id}: {status} | Accuracy: {accuracy:.1f}% | Metrics: {'✅' if has_metrics else '❌'}")
            
            return users
        else:
            print(f"❌ Failed to get agents: {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ Error getting agents: {e}")
        return {}

def create_test_agent(user_id: str):
    """Create a single test agent"""
    test_dataset = [
        {"prompt": "O que é programação?", "answer": "Programação é o processo de criar instruções para computadores executarem tarefas específicas."},
        {"prompt": "Como aprender a programar?", "answer": "Comece com conceitos básicos, pratique regularmente e construa projetos pequenos."},
        {"prompt": "Qual linguagem escolher?", "answer": "Python é excelente para iniciantes devido à sua sintaxe simples e versatilidade."}
    ]
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/agents/advanced",
            json={
                "user_id": user_id,
                "json_dataset": test_dataset,
                "base_model": "distilbert-base-uncased"
            },
            timeout=30
        )
        
        if response.status_code == 201:
            print(f"✅ Successfully created agent: {user_id}")
            return True
        else:
            print(f"❌ Failed to create agent {user_id}: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                pass
            return False
    except Exception as e:
        print(f"❌ Error creating agent {user_id}: {e}")
        return False

def test_multiple_agent_creation():
    """Test creating multiple agents like the dashboard would"""
    print("\n🧪 Testing Multiple Agent Creation (Dashboard Simulation)")
    print("=" * 60)
    
    # Agent configurations similar to what dashboard creates
    test_agents = [
        "python-basic-test",
        "javascript-test", 
        "devops-test",
        "ml-test",
        "web-test"
    ]
    
    created_count = 0
    
    for agent_id in test_agents:
        print(f"\n🔨 Creating agent: {agent_id}")
        if create_test_agent(agent_id):
            created_count += 1
            # Small delay between creations
            time.sleep(1)
    
    print(f"\n📈 Summary: Created {created_count}/{len(test_agents)} agents")
    
    # Wait for potential training to start
    print("\n⏳ Waiting 5 seconds for training to initialize...")
    time.sleep(5)
    
    return created_count

def monitor_training_progress():
    """Monitor training progress for created agents"""
    print("\n📊 Monitoring Training Progress")
    print("=" * 60)
    
    for _ in range(6):  # Check 6 times over 30 seconds
        agents = get_current_agents()
        
        if not agents:
            print("❌ No agents found")
            break
        
        training_count = 0
        completed_count = 0
        
        for user_id, user_data in agents.items():
            status = user_data.get('status', 'unknown')
            if status == 'processing':
                training_count += 1
            elif status == 'ready':
                completed_count += 1
        
        print(f"🔄 Training: {training_count} | Completed: {completed_count} | Total: {len(agents)}")
        
        if training_count == 0 and completed_count > 0:
            print("✅ All training completed!")
            break
        
        time.sleep(5)

def cleanup_test_agents():
    """Clean up test agents"""
    print("\n🧹 Cleaning up test agents...")
    
    agents = get_current_agents()
    test_agent_ids = [uid for uid in agents.keys() if 'test' in uid.lower()]
    
    for agent_id in test_agent_ids:
        try:
            response = requests.delete(f"{BACKEND_URL}/api/v1/agents/{agent_id}")
            if response.status_code == 200:
                print(f"🗑️ Deleted test agent: {agent_id}")
            else:
                print(f"❌ Failed to delete {agent_id}: {response.status_code}")
        except Exception as e:
            print(f"❌ Error deleting {agent_id}: {e}")

def main():
    """Main test function"""
    print("🚀 Multi-Agent Creation Test")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check backend health
    if not check_backend_health():
        print("\n❌ Backend not available. Please start it with:")
        print("   cd agents && python app.py")
        return
    
    # Get initial state
    print("\n1️⃣ Initial State Check")
    initial_agents = get_current_agents()
    
    # Test multiple agent creation
    print("\n2️⃣ Creating Multiple Agents")
    created_count = test_multiple_agent_creation()
    
    if created_count == 0:
        print("\n❌ No agents were created successfully")
        return
    
    # Check final state
    print("\n3️⃣ Final State Check")
    final_agents = get_current_agents()
    
    # Monitor training progress
    print("\n4️⃣ Training Progress Monitor")
    monitor_training_progress()
    
    # Summary
    print(f"\n📋 Test Summary")
    print("=" * 60)
    print(f"Initial agents: {len(initial_agents)}")
    print(f"Created agents: {created_count}")
    print(f"Final agents: {len(final_agents)}")
    print(f"Net increase: {len(final_agents) - len(initial_agents)}")
    
    # Ask about cleanup
    try:
        cleanup_choice = input("\n🧹 Clean up test agents? (y/N): ").lower().strip()
        if cleanup_choice == 'y':
            cleanup_test_agents()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    
    print("\n✅ Multi-agent creation test completed!")
    print("\n📝 Next steps:")
    print("   1. Open dashboard: http://localhost:3000/dashboard")
    print("   2. Click the training button to see it create agents automatically")
    print("   3. Watch the nodes populate with real training data")

if __name__ == "__main__":
    main()
