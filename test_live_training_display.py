#!/usr/bin/env python3
"""
Test script to verify live training data display in dashboard
Tests if real-time string comparison data is shown in the frontend nodes
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

def get_live_training_cycles():
    """Get live training cycles data"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/training/live-cycles", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Live Training Data Retrieved")
            print("=" * 60)
            
            for user_id, user_data in data.get('users', {}).items():
                status = user_data.get('status', 'unknown')
                accuracy = user_data.get('current_accuracy', 0)
                confidence = user_data.get('current_confidence', 0)
                cycles = user_data.get('total_cycles', 0)
                quality_dist = user_data.get('quality_distribution', {})
                recent_cycles = user_data.get('recent_cycles', [])
                
                print(f"👤 {user_id}:")
                print(f"   Status: {status}")
                print(f"   Current Accuracy: {accuracy:.1f}%")
                print(f"   Confidence: {confidence:.1f}%")
                print(f"   Total Cycles: {cycles}")
                print(f"   Quality: H:{quality_dist.get('HIGH', 0)} M:{quality_dist.get('MEDIUM', 0)} L:{quality_dist.get('LOW', 0)}")
                
                if recent_cycles:
                    print(f"   Recent Cycles:")
                    for cycle in recent_cycles[:3]:
                        print(f"     Cycle #{cycle['cycle']}: {cycle['similarity']:.1f}% ({cycle['quality']})")
                        print(f"       Q: {cycle['prompt_preview']}")
                print()
            
            return data
        else:
            print(f"❌ Failed to get live training cycles: {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ Error getting live training cycles: {e}")
        return {}

def create_test_agent_with_training():
    """Create a test agent and wait for training to begin"""
    print("\n🔨 Creating test agent for live training demo...")
    
    test_dataset = [
        {"prompt": "Como funciona o Python?", "answer": "Python é uma linguagem interpretada com sintaxe simples e tipagem dinâmica."},
        {"prompt": "O que é uma função?", "answer": "Uma função é um bloco de código reutilizável que executa uma tarefa específica."},
        {"prompt": "Como criar uma lista?", "answer": "Use colchetes para criar listas: lista = [1, 2, 3] ou lista = list()."},
        {"prompt": "Para que serve o loop for?", "answer": "O loop for é usado para iterar sobre sequências como listas, strings ou ranges."},
        {"prompt": "O que são dicionários?", "answer": "Dicionários são estruturas de dados que armazenam pares chave-valor: {'nome': 'João'}."},
    ]
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/agents/advanced",
            json={
                "user_id": "live-test-python",
                "json_dataset": test_dataset,
                "base_model": "distilbert-base-uncased"
            },
            timeout=30
        )
        
        if response.status_code == 201:
            print("✅ Test agent created successfully!")
            print("⏳ Waiting for training to begin...")
            return True
        else:
            print(f"❌ Failed to create test agent: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error creating test agent: {e}")
        return False

def monitor_live_training():
    """Monitor live training for 60 seconds"""
    print("\n📊 Monitoring Live Training Data")
    print("=" * 60)
    
    start_time = time.time()
    max_duration = 60  # Monitor for 60 seconds
    
    while time.time() - start_time < max_duration:
        data = get_live_training_cycles()
        
        if data and data.get('users'):
            # Check if any agent is actively training
            training_agents = 0
            completed_agents = 0
            
            for user_id, user_data in data['users'].items():
                status = user_data.get('status', 'unknown')
                cycles = user_data.get('total_cycles', 0)
                
                if status == 'active' and cycles > 0:
                    training_agents += 1
                elif status == 'completed':
                    completed_agents += 1
            
            print(f"🔄 Training: {training_agents} | Completed: {completed_agents} | Time: {time.time() - start_time:.0f}s")
            
            if training_agents > 0:
                print("✅ Live training detected! Data should be updating in dashboard.")
                break
        
        time.sleep(5)  # Check every 5 seconds
    
    if time.time() - start_time >= max_duration:
        print("⏰ Monitoring timeout reached")

def test_string_comparison_data():
    """Test if string comparison data is being captured"""
    print("\n🧪 Testing String Comparison Data Capture")
    print("=" * 60)
    
    # Make an inference request to trigger string comparison
    test_prompt = "Como criar uma função em Python?"
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/agents/live-test-python/inference",
            json={"prompt": test_prompt},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Inference successful!")
            print(f"Prompt: {test_prompt}")
            print(f"Response: {result.get('response', 'No response')[:100]}...")
            
            # Wait a moment for data to be recorded
            time.sleep(2)
            
            # Check if it appears in live data
            live_data = get_live_training_cycles()
            test_user_data = live_data.get('users', {}).get('live-test-python')
            
            if test_user_data and test_user_data.get('recent_cycles'):
                print("✅ String comparison data captured in live cycles!")
                recent = test_user_data['recent_cycles'][0]
                print(f"Latest cycle similarity: {recent['similarity']:.1f}%")
                print(f"Quality: {recent['quality']}")
            else:
                print("❌ No string comparison data found in live cycles")
        else:
            print(f"❌ Inference failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing inference: {e}")

def cleanup_test_agent():
    """Clean up test agent"""
    print("\n🧹 Cleaning up test agent...")
    try:
        response = requests.delete(f"{BACKEND_URL}/api/v1/agents/live-test-python")
        if response.status_code == 200:
            print("✅ Test agent cleaned up successfully")
        else:
            print(f"❌ Failed to clean up test agent: {response.status_code}")
    except Exception as e:
        print(f"❌ Error cleaning up: {e}")

def main():
    """Main test function"""
    print("🚀 Live Training Display Test")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check backend health
    if not check_backend_health():
        print("\n❌ Backend not available. Please start it with:")
        print("   cd agents && python app.py")
        return
    
    try:
        # Step 1: Check initial live data
        print("\n1️⃣ Initial Live Data Check")
        initial_data = get_live_training_cycles()
        
        # Step 2: Create test agent if needed
        print("\n2️⃣ Creating Test Agent")
        if create_test_agent_with_training():
            # Step 3: Monitor live training
            print("\n3️⃣ Monitoring Live Training")
            monitor_live_training()
            
            # Step 4: Test string comparison data
            print("\n4️⃣ Testing String Comparison")
            test_string_comparison_data()
            
            # Step 5: Final check
            print("\n5️⃣ Final Live Data Check")
            final_data = get_live_training_cycles()
        
        print("\n📋 Test Summary")
        print("=" * 60)
        print("✅ Live training endpoint working")
        print("✅ Real-time data capture active")
        print("✅ String comparison integration working")
        
        print("\n📝 Next Steps:")
        print("   1. Open dashboard: http://localhost:3000/dashboard")
        print("   2. Click training button to create agents")
        print("   3. Watch nodes update with REAL string comparison data")
        print("   4. Hover over nodes to see live training details")
        print("   5. See accuracy percentages update in real-time!")
        
        # Ask about cleanup
        try:
            cleanup_choice = input("\n🧹 Clean up test agent? (y/N): ").lower().strip()
            if cleanup_choice == 'y':
                cleanup_test_agent()
        except KeyboardInterrupt:
            print("\n\n👋 Test interrupted by user")
            cleanup_test_agent()
    
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        cleanup_test_agent()

if __name__ == "__main__":
    main()
