#!/usr/bin/env python3
"""
Test script to demonstrate how to make requests to trained models
in the ShellHacks agents system.

This script shows:
1. How to create a personalized agent
2. How to check training status
3. How to make inference (get model responses)
4. How to manage multiple agents
"""

import requests
import time
import json
from typing import Dict, Any, Optional
import sys

class AgentAPITester:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Check if the service is working"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ Service is working!")
                return True
            else:
                print(f"❌ Service has issues: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to service. Make sure it's running on localhost:8080")
            return False
    
    def create_agent(self, user_id: str, training_data: str, base_model: str = "distilbert-base-uncased") -> Dict[str, Any]:
        """Create a personalized agent"""
        print(f"\n🤖 Creating agent for user: {user_id}")
        
        data = {
            "user_id": user_id,
            "training_data": training_data,
            "base_model": base_model
        }
        
        try:
            response = self.session.post(f"{self.base_url}/api/v1/agents", json=data)
            result = response.json()
            
            if response.status_code == 201:
                print(f"✅ Agent created successfully!")
                print(f"   Status: {result.get('status')}")
                print(f"   User ID: {result.get('user_id')}")
            else:
                print(f"❌ Error creating agent: {result.get('error')}")
            
            return result
            
        except Exception as e:
            print(f"❌ Request error: {e}")
            return {"error": str(e)}
    
    def get_agent_status(self, user_id: str) -> Dict[str, Any]:
        """Check agent status"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/agents/{user_id}/status")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def wait_for_ready(self, user_id: str, max_wait: int = 300) -> bool:
        """Wait until the model is ready"""
        print(f"\n⏳ Waiting for model to be ready for {user_id}...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status = self.get_agent_status(user_id)
            
            if "error" in status:
                print(f"❌ Error checking status: {status['error']}")
                return False
            
            current_status = status.get('status', 'unknown')
            print(f"   Current status: {current_status}")
            
            if current_status == 'ready':
                print("✅ Model ready!")
                return True
            elif current_status == 'error':
                print(f"❌ Training error: {status.get('error', 'Unknown error')}")
                return False
            
            time.sleep(5)
        
        print("⏰ Timeout waiting for model to be ready")
        return False
    
    def make_inference(self, user_id: str, prompt: str) -> Optional[str]:
        """Make inference with the trained model"""
        print(f"\n💬 Asking question to {user_id}: {prompt}")
        
        data = {"prompt": prompt}
        
        try:
            response = self.session.post(f"{self.base_url}/api/v1/agents/{user_id}/inference", json=data)
            result = response.json()
            
            if response.status_code == 200:
                response_text = result.get('response', 'No response')
                print(f"🤖 Model response:")
                print(f"   {response_text}")
                return response_text
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"❌ Inference error: {error_msg}")
                return None
                
        except Exception as e:
            print(f"❌ Request error: {e}")
            return None
    
    def delete_agent(self, user_id: str) -> bool:
        """Delete an agent"""
        try:
            response = self.session.delete(f"{self.base_url}/api/v1/agents/{user_id}")
            if response.status_code == 200:
                print(f"✅ Agent {user_id} deleted successfully!")
                return True
            else:
                print(f"❌ Error deleting agent: {response.json().get('error')}")
                return False
        except Exception as e:
            print(f"❌ Request error: {e}")
            return False

def run_automated_tests():
    """Run automated tests with 3 different expert agents"""
    print("🚀 SHELLHACKS AGENTS - AUTOMATED TEST")
    print("=" * 50)
    
    tester = AgentAPITester()
    
    # Check if service is working
    if not tester.health_check():
        return
    
    # Define training data for different experts
    python_expert_data = """
    I am a Python programming expert with extensive experience in:
    - Web development with Flask, Django, FastAPI
    - Data analysis with pandas, numpy
    - Machine learning with scikit-learn, TensorFlow
    - Database programming with SQLAlchemy
    - API development and best practices
    - Code optimization and debugging
    """
    
    ml_expert_data = """
    I am a machine learning expert specializing in:
    - Deep learning with TensorFlow, PyTorch
    - Natural Language Processing (NLP)
    - Computer Vision
    - Data preprocessing and feature engineering
    - Model evaluation and validation
    - MLOps and model deployment
    - Algorithm selection and tuning
    """
    
    devops_expert_data = """
    I am a DevOps expert with experience in:
    - CI/CD pipelines with GitHub Actions, GitLab CI
    - Containerization with Docker and Kubernetes
    - Cloud platforms (AWS, GCP, Azure)
    - Infrastructure as Code (Terraform, Ansible)
    - Monitoring and logging (Prometheus, Grafana)
    - Security best practices
    - Automation and scripting
    """
    
    # Test Python Expert
    print("\n" + "="*50)
    print("🐍 TESTING PYTHON EXPERT AGENT")
    print("="*50)
    
    # Create agent
    agent1 = tester.create_agent("python_expert", python_expert_data)
    
    if agent1.get('status') == 'created':
        if tester.wait_for_ready("python_expert", max_wait=60):
            # Ask some questions
            python_questions = [
                "How to create a REST API with Flask?",
                "What's the difference between list comprehension and generator expression?",
                "How to optimize Python code performance?",
                "How to handle exceptions in Python?",
                "What are Python decorators and how to use them?"
            ]
            
            for question in python_questions:
                tester.make_inference("python_expert", question)
                time.sleep(2)  # Small delay between questions
    
    # Test ML Expert
    print("\n" + "="*50)
    print("🤖 TESTING ML EXPERT AGENT")
    print("="*50)
    
    # Create agent
    agent2 = tester.create_agent("ml_expert", ml_expert_data)
    
    if agent2.get('status') == 'created':
        if tester.wait_for_ready("ml_expert", max_wait=60):
            # Ask some questions
            ml_questions = [
                "How to choose between Random Forest and XGBoost?",
                "What is overfitting and how to prevent it?",
                "How to handle imbalanced datasets?",
                "What are the steps for feature engineering?",
                "How to evaluate model performance?"
            ]
            
            for question in ml_questions:
                tester.make_inference("ml_expert", question)
                time.sleep(2)
    
    # Test DevOps Expert
    print("\n" + "="*50)
    print("🔧 TESTING DEVOPS EXPERT AGENT")
    print("="*50)
    
    # Create agent
    agent3 = tester.create_agent("devops_expert", devops_expert_data)
    
    if agent3.get('status') == 'created':
        if tester.wait_for_ready("devops_expert", max_wait=60):
            # Ask some questions
            devops_questions = [
                "How to setup CI/CD pipeline?",
                "What is the difference between Docker and Kubernetes?",
                "How to monitor application performance?",
                "What are the best practices for container security?",
                "How to implement blue-green deployment?"
            ]
            
            for question in devops_questions:
                tester.make_inference("devops_expert", question)
                time.sleep(2)
    
    print("\n🎉 AUTOMATED TEST COMPLETED!")
    print("=" * 50)

def interactive_mode():
    """Interactive mode for manual testing"""
    print("\n🎮 INTERACTIVE MODE")
    print("Available commands:")
    print("- status <user_id>: Check agent status")
    print("- ask <user_id> <question>: Ask question to an agent")
    print("- list: List all agents")
    print("- delete <user_id>: Delete an agent")
    print("- quit: Exit")
    print()
    
    tester = AgentAPITester()
    
    while True:
        try:
            command = input("Enter command: ").strip().split()
            
            if not command:
                continue
            
            if command[0] == "quit":
                break
            elif command[0] == "status" and len(command) > 1:
                status = tester.get_agent_status(command[1])
                print(json.dumps(status, indent=2))
            elif command[0] == "ask" and len(command) > 2:
                user_id = command[1]
                question = " ".join(command[2:])
                tester.make_inference(user_id, question)
            elif command[0] == "list":
                response = tester.session.get(f"{tester.base_url}/api/v1/agents")
                if response.status_code == 200:
                    data = response.json()
                    print(f"Total agents: {data.get('total_users', 0)}")
                    for user_id, info in data.get('users', {}).items():
                        print(f"- {user_id}: {info.get('status', 'unknown')}")
                else:
                    print("Error listing agents")
            elif command[0] == "delete" and len(command) > 1:
                tester.delete_agent(command[1])
            else:
                print("Invalid command. Try again.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        run_automated_tests()
        
        # Ask if user wants to enter interactive mode
        try:
            choice = input("\nDo you want to enter interactive mode? (y/n): ").lower()
            if choice in ['y', 'yes']:
                interactive_mode()
        except KeyboardInterrupt:
            print("\nGoodbye!")

if __name__ == "__main__":
    main()