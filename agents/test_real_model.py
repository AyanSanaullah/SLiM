#!/usr/bin/env python3
"""
Script to test the REAL model training system
Demonstrates personalized model training using prompt/answer data
and evaluation using string comparison
"""

import requests
import time
import json
from typing import Dict, Any

class RealModelTester:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
    
    def test_real_model_training(self):
        """Test the complete real training system"""
        print("🚀 REAL MODEL TRAINING SYSTEM TEST")
        print("=" * 60)
        
        # 1. Check if server is running
        if not self._check_server():
            return
        
        # 2. Create agent with rich training data
        training_data = """
        I am a Python and web development expert. I have experience with:
        
        - Flask: Minimalist web framework for Python
        - Django: Robust web framework for complex applications
        - FastAPI: Modern framework for fast APIs
        - SQLAlchemy: ORM for databases
        - Pandas: Library for data analysis
        - NumPy: Scientific computing with arrays
        - Scikit-learn: Machine learning in Python
        - TensorFlow: Deep learning and neural networks
        
        I can help with:
        - Creating REST APIs
        - Web application development
        - Data analysis
        - Machine learning
        - Code debugging and optimization
        - Programming best practices
        """
        
        user_id = "real_python_expert"
        
        print("🤖 Creating agent with REAL training...")
        
        # Create agent
        agent_response = self._create_agent(user_id, training_data)
        if not agent_response:
            return
        
        print(f"✅ Agent created successfully!")
        print(f"   User ID: {user_id}")
        print(f"   Status: {agent_response.get('status', 'unknown')}")
        
        # 3. Monitor training progress
        print("\n⏳ Monitoring training progress...")
        if not self._wait_for_training_completion(user_id):
            print("❌ Training failed or timeout")
            return
        
        # 4. Test inference with various questions
        print("\n🧪 Testing inference with various questions...")
        test_questions = [
            "How to create a REST API with Flask?",
            "What is the difference between Flask and Django?",
            "How to optimize Python code performance?",
            "How to implement authentication in FastAPI?",
            "What are the best practices for SQLAlchemy?",
            "How to analyze data with pandas?",
            "How to create a machine learning model with scikit-learn?",
            "How to debug Python applications?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 Question {i}: {question}")
            response = self._ask_question(user_id, question)
            if response:
                print(f"🤖 Answer: {response[:200]}{'...' if len(response) > 200 else ''}")
            else:
                print("❌ Failed to get response")
            
            # Small delay between questions
            time.sleep(1)
        
        print("\n🎉 REAL MODEL TRAINING TEST COMPLETED!")
        print("=" * 60)
    
    def _check_server(self) -> bool:
        """Check if the server is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Server is running!")
                print(f"   Service: {data.get('service', 'unknown')}")
                print(f"   Version: {data.get('version', 'unknown')}")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to server. Make sure it's running:")
            print("   python3 app.py")
            return False
        except Exception as e:
            print(f"❌ Error checking server: {e}")
            return False
    
    def _create_agent(self, user_id: str, training_data: str) -> Dict[str, Any]:
        """Create a new agent"""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/agents",
                json={
                    "user_id": user_id,
                    "training_data": training_data,
                    "base_model": "distilbert-base-uncased"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to create agent: {response.status_code}")
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Error creating agent: {e}")
            return None
    
    def _wait_for_training_completion(self, user_id: str, max_wait: int = 300) -> bool:
        """Wait for training to complete"""
        start_time = time.time()
        last_status = ""
        
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(f"{self.base_url}/api/v1/agents/{user_id}/status")
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status', 'unknown')
                    
                    if status != last_status:
                        print(f"   Status: {status}")
                        last_status = status
                    
                    if status == 'ready':
                        print("✅ Training completed successfully!")
                        return True
                    elif status == 'error':
                        print("❌ Training failed!")
                        error_msg = data.get('error', 'Unknown error')
                        print(f"   Error: {error_msg}")
                        return False
                    
                    # Show progress for processing status
                    if status == 'processing':
                        progress = data.get('progress', {})
                        if progress:
                            current_step = progress.get('current_step', '')
                            if current_step:
                                print(f"   Processing: {current_step}")
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"❌ Error checking status: {e}")
                time.sleep(5)
        
        print(f"⏰ Timeout waiting for training completion ({max_wait}s)")
        return False
    
    def _ask_question(self, user_id: str, question: str) -> str:
        """Ask a question to the trained model"""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/agents/{user_id}/inference",
                json={"prompt": question},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('response', '')
            else:
                print(f"❌ Inference failed: {response.status_code}")
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Error asking question: {e}")
            return None

def main():
    """Main function"""
    print("🧪 Starting REAL MODEL TRAINING test...")
    print("This test will:")
    print("1. Check if the server is running")
    print("2. Create an agent with comprehensive training data")
    print("3. Monitor the training process")
    print("4. Test inference with various questions")
    print()
    
    tester = RealModelTester()
    tester.test_real_model_training()

if __name__ == "__main__":
    main()
