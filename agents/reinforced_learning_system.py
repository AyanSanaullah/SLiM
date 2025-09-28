#!/usr/bin/env python3
"""
Reinforced Learning System for Agent Optimization
Creates multiple versions of the same agent and uses string-comparison to find the best performing model
"""

import requests
import time
import json
import random
import numpy as np
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import os

@dataclass
class AgentConfig:
    """Configuration for an agent variant"""
    user_id: str
    learning_rate: float
    batch_size: int
    epochs: int
    temperature: float
    top_p: float
    max_length: int
    dropout_rate: float

class StringComparisonClient:
    """Client for string-comparison service"""
    
    def __init__(self, base_url: str = "http://0.0.0.0:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def compare_sentences(self, sentence1: str, sentence2: str) -> Dict[str, Any]:
        """Compare two sentences using string-comparison service"""
        try:
            response = self.session.post(
                f"{self.base_url}/compare",
                json={
                    "sentence1": sentence1,
                    "sentence2": sentence2
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"similarity": 0.0, "error": "Service error"}
                
        except Exception as e:
            return {"similarity": 0.0, "error": str(e)}

class ReinforcedLearningSystem:
    """Main reinforced learning system for agent optimization"""
    
    def __init__(self, agents_base_url: str = "http://localhost:8080", 
                 string_comparison_url: str = "http://0.0.0.0:8000"):
        self.agents_base_url = agents_base_url
        self.string_comparison = StringComparisonClient(string_comparison_url)
        self.session = requests.Session()
        
        # Performance tracking
        self.performance_history = []
        self.agent_performances = {}
        self.best_agent = None
        self.generation = 0
        
        # Test questions for evaluation
        self.test_questions = [
            {
                "prompt": "How to create a REST API with Flask?",
                "expected_answer": "To create a REST API with Flask, you need to install Flask, create routes with @app.route decorator, handle HTTP methods, and return JSON responses using jsonify."
            },
            {
                "prompt": "What is the difference between Flask and Django?",
                "expected_answer": "Flask is a microframework that is lightweight and flexible, while Django is a full-stack framework with built-in features like ORM, admin panel, and authentication."
            },
            {
                "prompt": "How to optimize Python code performance?",
                "expected_answer": "Python optimization techniques include using list comprehensions, avoiding global variables, using generators, profiling with cProfile, using NumPy for numerical operations, and implementing caching."
            },
            {
                "prompt": "What are Python decorators?",
                "expected_answer": "Decorators are functions that modify other functions. They use the @decorator_name syntax and are commonly used for authentication, logging, caching, and performance monitoring."
            },
            {
                "prompt": "How to handle exceptions in Python?",
                "expected_answer": "Python exception handling uses try/except/finally blocks. Use specific exception types, log errors appropriately, and implement proper cleanup in finally blocks."
            }
        ]
    
    def generate_agent_configs(self, base_user_id: str, num_agents: int = 20) -> List[AgentConfig]:
        """Generate multiple agent configurations with varying parameters"""
        configs = []
        
        for i in range(num_agents):
            config = AgentConfig(
                user_id=f"{base_user_id}_v{i+1:02d}",
                learning_rate=random.uniform(1e-5, 1e-3),
                batch_size=random.choice([8, 16, 32, 64]),
                epochs=random.randint(2, 10),
                temperature=random.uniform(0.3, 1.0),
                top_p=random.uniform(0.7, 0.95),
                max_length=random.choice([256, 512, 768, 1024]),
                dropout_rate=random.uniform(0.1, 0.5)
            )
            configs.append(config)
        
        return configs
    
    def create_agent_with_config(self, config: AgentConfig, base_training_data: List[Dict]) -> bool:
        """Create an agent with specific configuration"""
        print(f"🤖 Creating agent: {config.user_id}")
        
        # Prepare training data with configuration-specific modifications
        training_data = self._modify_training_data_for_config(base_training_data, config)
        
        payload = {
            "user_id": config.user_id,
            "json_dataset": training_data,
            "base_model": "distilbert-base-uncased",
            "training_config": {
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_length": config.max_length,
                "dropout_rate": config.dropout_rate
            }
        }
        
        try:
            response = self.session.post(
                f"{self.agents_base_url}/api/v1/agents/advanced",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 201:
                print(f"✅ Agent {config.user_id} created successfully")
                return True
            else:
                print(f"❌ Failed to create agent {config.user_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error creating agent {config.user_id}: {e}")
            return False
    
    def _modify_training_data_for_config(self, base_data: List[Dict], config: AgentConfig) -> List[Dict]:
        """Modify training data based on agent configuration"""
        modified_data = []
        
        for item in base_data:
            # Adjust answer length based on max_length config
            answer = item['answer']
            if len(answer) > config.max_length:
                # Truncate or summarize based on temperature (creativity)
                if config.temperature > 0.7:
                    # More creative - summarize
                    sentences = answer.split('. ')
                    if len(sentences) > 1:
                        answer = '. '.join(sentences[:2]) + '.'
                else:
                    # More conservative - truncate
                    answer = answer[:config.max_length] + "..."
            
            modified_data.append({
                "prompt": item['prompt'],
                "answer": answer,
                "category": item.get('category', 'general')
            })
        
        return modified_data
    
    def evaluate_agent_performance(self, config: AgentConfig) -> Dict[str, Any]:
        """Evaluate agent performance using string comparison"""
        print(f"🧪 Evaluating agent: {config.user_id}")
        
        evaluation_results = {
            'user_id': config.user_id,
            'config': config.__dict__,
            'similarities': [],
            'average_similarity': 0.0,
            'success_rate': 0.0,
            'total_questions': len(self.test_questions),
            'successful_questions': 0
        }
        
        for question_data in self.test_questions:
            prompt = question_data['prompt']
            expected_answer = question_data['expected_answer']
            
            try:
                # Get agent response
                response = self.session.post(
                    f"{self.agents_base_url}/api/v1/agents/{config.user_id}/inference",
                    json={"prompt": prompt},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    agent_answer = result.get('response', '')
                    
                    # Compare with expected answer using string-comparison service
                    comparison = self.string_comparison.compare_sentences(
                        agent_answer, expected_answer
                    )
                    
                    # Use semantic_similarity if available, fallback to similarity
                    similarity = comparison.get('semantic_similarity', comparison.get('similarity', 0.0))
                    evaluation_results['similarities'].append(similarity)
                    
                    if similarity > 0.5:  # Threshold for success
                        evaluation_results['successful_questions'] += 1
                    
                    print(f"   Q: {prompt[:30]}... | Similarity: {similarity:.3f}")
                    
                else:
                    print(f"   ❌ Failed to get response for question")
                    evaluation_results['similarities'].append(0.0)
                    
            except Exception as e:
                print(f"   ❌ Error evaluating question: {e}")
                evaluation_results['similarities'].append(0.0)
            
            time.sleep(1)  # Small delay between questions
        
        # Calculate metrics
        similarities = evaluation_results['similarities']
        if similarities:
            evaluation_results['average_similarity'] = np.mean(similarities)
            evaluation_results['success_rate'] = evaluation_results['successful_questions'] / len(self.test_questions)
        
        print(f"   📊 Average Similarity: {evaluation_results['average_similarity']:.3f}")
        print(f"   📊 Success Rate: {evaluation_results['success_rate']:.3f}")
        
        return evaluation_results
    
    def wait_for_agent_training(self, config: AgentConfig, max_wait: int = 300) -> bool:
        """Wait for agent training to complete"""
        print(f"⏳ Waiting for {config.user_id} training...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = self.session.get(f"{self.agents_base_url}/api/v1/agents/{config.user_id}/status")
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status', 'unknown')
                    
                    if status == 'ready':
                        print(f"✅ {config.user_id} training completed")
                        return True
                    elif status == 'error':
                        print(f"❌ {config.user_id} training failed")
                        return False
                    
                    time.sleep(5)
                    
            except Exception as e:
                print(f"❌ Error checking status for {config.user_id}: {e}")
                time.sleep(5)
        
        print(f"⏰ Timeout waiting for {config.user_id}")
        return False
    
    def train_and_evaluate_agents(self, base_user_id: str, base_training_data: List[Dict], 
                                 num_agents: int = 20) -> List[Dict[str, Any]]:
        """Train multiple agents and evaluate their performance"""
        print(f"🚀 REINFORCED LEARNING SYSTEM - GENERATION {self.generation + 1}")
        print("=" * 80)
        print(f"Training {num_agents} agent variants for optimization...")
        
        # Generate agent configurations
        configs = self.generate_agent_configs(base_user_id, num_agents)
        
        results = []
        successful_agents = []
        
        # Train agents in parallel (limited concurrency to avoid overload)
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit training tasks
            future_to_config = {
                executor.submit(self._train_single_agent, config, base_training_data): config 
                for config in configs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        successful_agents.append(config)
                        print(f"✅ Agent {config.user_id} completed successfully")
                    else:
                        print(f"❌ Agent {config.user_id} failed")
                except Exception as e:
                    print(f"❌ Error with agent {config.user_id}: {e}")
        
        # Update performance tracking
        self.performance_history.append(results)
        self.generation += 1
        
        # Find best performing agent
        if results:
            best_result = max(results, key=lambda x: x['average_similarity'])
            self.best_agent = best_result
            print(f"\n🏆 BEST AGENT: {best_result['user_id']}")
            print(f"   Average Similarity: {best_result['average_similarity']:.3f}")
            print(f"   Success Rate: {best_result['success_rate']:.3f}")
        
        return results
    
    def _train_single_agent(self, config: AgentConfig, base_training_data: List[Dict]) -> Dict[str, Any]:
        """Train and evaluate a single agent"""
        # Create agent
        if not self.create_agent_with_config(config, base_training_data):
            return None
        
        # Wait for training
        if not self.wait_for_agent_training(config):
            return None
        
        # Evaluate performance
        evaluation = self.evaluate_agent_performance(config)
        
        return evaluation
    
    def visualize_performance(self, results: List[Dict[str, Any]]):
        """Create visualizations of agent performance"""
        if not results:
            return
        
        print("\n📊 Creating performance visualizations...")
        
        # Prepare data for visualization
        user_ids = [r['user_id'] for r in results]
        similarities = [r['average_similarity'] for r in results]
        success_rates = [r['success_rate'] for r in results]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Agent Performance Analysis - Generation {self.generation}', fontsize=16)
        
        # 1. Average Similarity Bar Chart
        axes[0, 0].bar(range(len(user_ids)), similarities, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Average Similarity by Agent')
        axes[0, 0].set_xlabel('Agent Index')
        axes[0, 0].set_ylabel('Average Similarity')
        axes[0, 0].set_xticks(range(len(user_ids)))
        axes[0, 0].set_xticklabels([uid.split('_')[-1] for uid in user_ids], rotation=45)
        
        # Highlight best performer
        best_idx = np.argmax(similarities)
        axes[0, 0].bar(best_idx, similarities[best_idx], color='gold', alpha=0.8)
        
        # 2. Success Rate vs Similarity Scatter
        axes[0, 1].scatter(success_rates, similarities, alpha=0.6, s=100)
        axes[0, 1].set_title('Success Rate vs Average Similarity')
        axes[0, 1].set_xlabel('Success Rate')
        axes[0, 1].set_ylabel('Average Similarity')
        
        # Add trend line
        z = np.polyfit(success_rates, similarities, 1)
        p = np.poly1d(z)
        axes[0, 1].plot(success_rates, p(success_rates), "r--", alpha=0.8)
        
        # 3. Performance Distribution Histogram
        axes[1, 0].hist(similarities, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Similarity Score Distribution')
        axes[1, 0].set_xlabel('Average Similarity')
        axes[1, 0].set_ylabel('Number of Agents')
        axes[1, 0].axvline(np.mean(similarities), color='red', linestyle='--', label=f'Mean: {np.mean(similarities):.3f}')
        axes[1, 0].legend()
        
        # 4. Top 5 Performers
        top_5_indices = np.argsort(similarities)[-5:]
        top_5_ids = [user_ids[i].split('_')[-1] for i in top_5_indices]
        top_5_scores = [similarities[i] for i in top_5_indices]
        
        axes[1, 1].barh(range(5), top_5_scores, color='lightcoral', alpha=0.7)
        axes[1, 1].set_title('Top 5 Performing Agents')
        axes[1, 1].set_xlabel('Average Similarity')
        axes[1, 1].set_yticks(range(5))
        axes[1, 1].set_yticklabels(top_5_ids)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"performance_analysis_gen_{self.generation}_{int(time.time())}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Performance visualization saved: {plot_filename}")
        
        # Show plot if in interactive environment
        try:
            plt.show()
        except:
            pass
    
    def generate_evolution_report(self, results: List[Dict[str, Any]]):
        """Generate comprehensive evolution report"""
        print("\n" + "=" * 80)
        print("📊 REINFORCED LEARNING EVOLUTION REPORT")
        print("=" * 80)
        
        if not results:
            print("No results to report")
            return
        
        # Overall statistics
        similarities = [r['average_similarity'] for r in results]
        success_rates = [r['success_rate'] for r in results]
        
        print(f"Generation: {self.generation}")
        print(f"Total Agents: {len(results)}")
        print(f"Average Similarity: {np.mean(similarities):.3f} ± {np.std(similarities):.3f}")
        print(f"Best Similarity: {np.max(similarities):.3f}")
        print(f"Average Success Rate: {np.mean(success_rates):.3f}")
        
        # Top performers
        print(f"\n🏆 TOP 5 PERFORMERS:")
        top_5 = sorted(results, key=lambda x: x['average_similarity'], reverse=True)[:5]
        
        for i, result in enumerate(top_5, 1):
            config = result['config']
            print(f"{i}. {result['user_id']}")
            print(f"   Similarity: {result['average_similarity']:.3f}")
            print(f"   Success Rate: {result['success_rate']:.3f}")
            print(f"   Learning Rate: {config['learning_rate']:.2e}")
            print(f"   Batch Size: {config['batch_size']}")
            print(f"   Temperature: {config['temperature']:.2f}")
            print()
        
        # Configuration analysis
        print(f"🔍 CONFIGURATION ANALYSIS:")
        
        # Learning rate analysis
        lr_groups = {}
        for result in results:
            lr = result['config']['learning_rate']
            lr_range = f"{lr:.2e}"
            if lr_range not in lr_groups:
                lr_groups[lr_range] = []
            lr_groups[lr_range].append(result['average_similarity'])
        
        print("Learning Rate Performance:")
        for lr_range, scores in sorted(lr_groups.items()):
            print(f"  {lr_range}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        
        # Temperature analysis
        temp_groups = {}
        for result in results:
            temp = result['config']['temperature']
            temp_range = f"{temp:.1f}"
            if temp_range not in temp_groups:
                temp_groups[temp_range] = []
            temp_groups[temp_range].append(result['average_similarity'])
        
        print("Temperature Performance:")
        for temp_range, scores in sorted(temp_groups.items()):
            print(f"  {temp_range}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        
        # Save detailed report
        report_data = {
            'generation': self.generation,
            'timestamp': datetime.now().isoformat(),
            'total_agents': len(results),
            'statistics': {
                'mean_similarity': float(np.mean(similarities)),
                'std_similarity': float(np.std(similarities)),
                'max_similarity': float(np.max(similarities)),
                'mean_success_rate': float(np.mean(success_rates))
            },
            'top_performers': top_5,
            'all_results': results
        }
        
        report_filename = f"evolution_report_gen_{self.generation}_{int(time.time())}.json"
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Detailed report saved: {report_filename}")

def main():
    """Main function to run reinforced learning system"""
    print("🧬 REINFORCED LEARNING AGENT OPTIMIZATION SYSTEM")
    print("=" * 80)
    
    # Load base training data
    try:
        with open('training_datasets/python_expert_dataset.json', 'r') as f:
            dataset = json.load(f)
            base_training_data = dataset['training_data']
    except FileNotFoundError:
        print("❌ Base training dataset not found. Please ensure training_datasets/python_expert_dataset.json exists")
        return
    
    # Initialize reinforced learning system
    rl_system = ReinforcedLearningSystem()
    
    # Run evolution
    print("🎯 Starting agent evolution process...")
    results = rl_system.train_and_evaluate_agents(
        base_user_id="python_expert_evolved",
        base_training_data=base_training_data,
        num_agents=20
    )
    
    if results:
        # Generate visualizations and reports
        rl_system.visualize_performance(results)
        rl_system.generate_evolution_report(results)
        
        print("\n🎉 REINFORCED LEARNING COMPLETED!")
        print("=" * 80)
    else:
        print("❌ No agents were successfully trained and evaluated")

if __name__ == "__main__":
    main()
