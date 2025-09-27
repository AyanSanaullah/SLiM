#!/usr/bin/env python3
"""
Comprehensive script to train 5 specialized agents simultaneously
Creates extensive training datasets and trains multiple expert agents in parallel
"""

import requests
import time
import json
import threading
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

class MultiAgentTrainer:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
        self.training_results = {}
        
    def check_services(self) -> bool:
        """Check if all required services are running"""
        print("🔍 Checking service status...")
        
        # Check main agents service
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code != 200:
                print("❌ Agents service is not running")
                return False
            print("✅ Agents service is running")
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to agents service")
            return False
        
        # Check string comparison service
        try:
            response = self.session.get("http://localhost:8000/health", timeout=5)
            if response.status_code != 200:
                print("❌ String comparison service is not running")
                return False
            print("✅ String comparison service is running")
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to string comparison service")
            return False
        
        return True
    
    def load_training_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Load training dataset from JSON file"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Dataset file not found: {dataset_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in dataset file {dataset_path}: {e}")
            return None
    
    def create_advanced_agent(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Create an advanced agent with comprehensive training data"""
        user_id = dataset['user_id']
        description = dataset['description']
        training_data = dataset['training_data']
        
        print(f"\n🤖 Creating advanced agent: {user_id}")
        print(f"   Description: {description}")
        print(f"   Training samples: {len(training_data)}")
        
        # Prepare JSON dataset format
        json_dataset = []
        for item in training_data:
            json_dataset.append({
                "prompt": item['prompt'],
                "answer": item['answer'],
                "category": item.get('category', 'general')
            })
        
        payload = {
            "user_id": user_id,
            "json_dataset": json_dataset,
            "base_model": "distilbert-base-uncased"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/agents/advanced",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 201:
                result = response.json()
                print(f"✅ Agent {user_id} created successfully!")
                return result
            else:
                error_data = response.json()
                print(f"❌ Failed to create agent {user_id}: {error_data.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Error creating agent {user_id}: {e}")
            return None
    
    def monitor_agent_training(self, user_id: str, max_wait: int = 600) -> bool:
        """Monitor agent training progress"""
        print(f"\n⏳ Monitoring training for {user_id}...")
        
        start_time = time.time()
        last_status = ""
        
        while time.time() - start_time < max_wait:
            try:
                response = self.session.get(f"{self.base_url}/api/v1/agents/{user_id}/status")
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status', 'unknown')
                    
                    if status != last_status:
                        print(f"   {user_id}: {status}")
                        last_status = status
                    
                    if status == 'ready':
                        print(f"✅ {user_id} training completed!")
                        return True
                    elif status == 'error':
                        print(f"❌ {user_id} training failed!")
                        error_msg = data.get('error', 'Unknown error')
                        print(f"   Error: {error_msg}")
                        return False
                    
                    # Show progress for processing status
                    if status == 'processing':
                        progress = data.get('progress', {})
                        if progress:
                            current_step = progress.get('current_step', '')
                            if current_step:
                                print(f"   {user_id}: {current_step}")
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"❌ Error monitoring {user_id}: {e}")
                time.sleep(10)
        
        print(f"⏰ Timeout waiting for {user_id} training completion ({max_wait}s)")
        return False
    
    def test_agent_inference(self, user_id: str, test_questions: List[str]) -> Dict[str, Any]:
        """Test agent with sample questions"""
        print(f"\n🧪 Testing inference for {user_id}...")
        
        results = {
            'user_id': user_id,
            'questions_asked': len(test_questions),
            'successful_responses': 0,
            'failed_responses': 0,
            'responses': []
        }
        
        for i, question in enumerate(test_questions, 1):
            print(f"   Question {i}: {question[:50]}...")
            
            try:
                response = self.session.post(
                    f"{self.base_url}/api/v1/agents/{user_id}/inference",
                    json={"prompt": question},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get('response', '')
                    results['responses'].append({
                        'question': question,
                        'answer': answer,
                        'success': True
                    })
                    results['successful_responses'] += 1
                    print(f"   ✅ Response received ({len(answer)} chars)")
                else:
                    error_data = response.json()
                    results['responses'].append({
                        'question': question,
                        'answer': None,
                        'error': error_data.get('error', 'Unknown error'),
                        'success': False
                    })
                    results['failed_responses'] += 1
                    print(f"   ❌ Failed: {error_data.get('error', 'Unknown error')}")
                    
            except Exception as e:
                results['responses'].append({
                    'question': question,
                    'answer': None,
                    'error': str(e),
                    'success': False
                })
                results['failed_responses'] += 1
                print(f"   ❌ Error: {e}")
            
            time.sleep(2)  # Small delay between questions
        
        return results
    
    def train_single_agent(self, dataset_path: str, test_questions: List[str]) -> Dict[str, Any]:
        """Train a single agent with comprehensive testing"""
        dataset = self.load_training_dataset(dataset_path)
        if not dataset:
            return None
        
        user_id = dataset['user_id']
        
        # Create agent
        agent_result = self.create_advanced_agent(dataset)
        if not agent_result:
            return None
        
        # Monitor training
        training_success = self.monitor_agent_training(user_id)
        if not training_success:
            return None
        
        # Test inference
        test_results = self.test_agent_inference(user_id, test_questions)
        
        return {
            'user_id': user_id,
            'description': dataset['description'],
            'training_samples': len(dataset['training_data']),
            'training_success': training_success,
            'test_results': test_results
        }
    
    def train_all_agents(self):
        """Train all 5 agents with comprehensive datasets"""
        print("🚀 MULTI-AGENT TRAINING SYSTEM")
        print("=" * 60)
        print("Training 5 specialized agents simultaneously:")
        print("1. Python Expert - Web development, frameworks, best practices")
        print("2. ML Expert - Machine learning, deep learning, NLP, MLOps")
        print("3. DevOps Expert - CI/CD, containerization, cloud platforms")
        print("4. Data Science Expert - Statistics, analysis, visualization")
        print("5. Cybersecurity Expert - Security protocols, threat detection")
        print("=" * 60)
        
        if not self.check_services():
            print("❌ Required services are not running. Please start them first:")
            print("   python3 app.py")
            print("   cd ../string-comparison && python3 backend.py")
            return
        
        # Define agent configurations
        agents_config = [
            {
                'dataset_path': 'training_datasets/python_expert_dataset.json',
                'test_questions': [
                    "How to create a REST API with Flask?",
                    "What's the difference between Flask and Django?",
                    "How to optimize Python code performance?",
                    "What are Python decorators?",
                    "How to handle exceptions in Python?"
                ]
            },
            {
                'dataset_path': 'training_datasets/ml_expert_dataset.json',
                'test_questions': [
                    "How to choose between Random Forest and XGBoost?",
                    "What is overfitting and how to prevent it?",
                    "How to handle imbalanced datasets?",
                    "What are the steps for feature engineering?",
                    "How to evaluate model performance?"
                ]
            },
            {
                'dataset_path': 'training_datasets/devops_expert_dataset.json',
                'test_questions': [
                    "How to setup CI/CD pipeline?",
                    "What is the difference between Docker and Kubernetes?",
                    "How to monitor application performance?",
                    "What are the best practices for container security?",
                    "How to implement blue-green deployment?"
                ]
            },
            {
                'dataset_path': 'training_datasets/data_science_expert_dataset.json',
                'test_questions': [
                    "How to perform exploratory data analysis?",
                    "What are the key statistical concepts in data science?",
                    "How to handle missing data in datasets?",
                    "How to detect and handle outliers?",
                    "What is feature engineering and its importance?"
                ]
            },
            {
                'dataset_path': 'training_datasets/cybersecurity_expert_dataset.json',
                'test_questions': [
                    "What are the fundamental principles of cybersecurity?",
                    "How to implement secure authentication systems?",
                    "What is penetration testing and how to conduct it?",
                    "How to detect and respond to security incidents?",
                    "What are common cyber attack vectors?"
                ]
            }
        ]
        
        print("\n🎯 Starting parallel agent training...")
        
        # Train agents in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all training tasks
            future_to_agent = {
                executor.submit(self.train_single_agent, config['dataset_path'], config['test_questions']): 
                config['dataset_path'] for config in agents_config
            }
            
            # Collect results as they complete
            completed_agents = []
            for future in as_completed(future_to_agent):
                dataset_path = future_to_agent[future]
                try:
                    result = future.result()
                    if result:
                        completed_agents.append(result)
                        print(f"\n✅ Agent {result['user_id']} training completed!")
                    else:
                        print(f"\n❌ Agent training failed for {dataset_path}")
                except Exception as e:
                    print(f"\n❌ Error training agent from {dataset_path}: {e}")
        
        # Generate comprehensive report
        self.generate_training_report(completed_agents)
    
    def generate_training_report(self, agents: List[Dict[str, Any]]):
        """Generate comprehensive training report"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TRAINING REPORT")
        print("=" * 80)
        
        total_agents = len(agents)
        successful_agents = len([a for a in agents if a['training_success']])
        
        print(f"Total Agents: {total_agents}")
        print(f"Successfully Trained: {successful_agents}")
        print(f"Training Success Rate: {(successful_agents/total_agents)*100:.1f}%")
        
        print("\n📋 DETAILED RESULTS:")
        print("-" * 80)
        
        for agent in agents:
            user_id = agent['user_id']
            description = agent['description']
            training_samples = agent['training_samples']
            training_success = agent['training_success']
            test_results = agent['test_results']
            
            print(f"\n🤖 {user_id.upper()}")
            print(f"   Description: {description}")
            print(f"   Training Samples: {training_samples}")
            print(f"   Training Status: {'✅ Success' if training_success else '❌ Failed'}")
            
            if training_success and test_results:
                successful_responses = test_results['successful_responses']
                total_questions = test_results['questions_asked']
                success_rate = (successful_responses/total_questions)*100 if total_questions > 0 else 0
                
                print(f"   Test Questions: {total_questions}")
                print(f"   Successful Responses: {successful_responses}")
                print(f"   Inference Success Rate: {success_rate:.1f}%")
        
        # Save detailed report to file
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_agents': total_agents,
            'successful_agents': successful_agents,
            'training_success_rate': (successful_agents/total_agents)*100,
            'agents': agents
        }
        
        report_filename = f"training_report_{int(time.time())}.json"
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            print(f"\n📄 Detailed report saved to: {report_filename}")
        except Exception as e:
            print(f"\n❌ Error saving report: {e}")
        
        print("\n🎉 MULTI-AGENT TRAINING COMPLETED!")
        print("=" * 80)

def main():
    """Main function"""
    print("🚀 Starting Multi-Agent Training System...")
    
    trainer = MultiAgentTrainer()
    trainer.train_all_agents()

if __name__ == "__main__":
    main()
