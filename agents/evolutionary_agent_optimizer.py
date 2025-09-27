#!/usr/bin/env python3
"""
Evolutionary Agent Optimizer - Complete System
Implements reinforced learning with multiple agent generations and real-time monitoring
"""

import requests
import time
import json
import random
import numpy as np
from typing import Dict, Any, List, Tuple
import threading
from datetime import datetime
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our custom modules
from reinforced_learning_system import ReinforcedLearningSystem
from real_time_monitor import RealTimeMonitor

class EvolutionaryAgentOptimizer:
    """Main system for evolutionary agent optimization"""
    
    def __init__(self, agents_base_url: str = "http://localhost:8080",
                 string_comparison_url: str = "http://0.0.0.0:8000"):
        self.agents_base_url = agents_base_url
        self.string_comparison_url = string_comparison_url
        
        # Initialize subsystems
        self.rl_system = ReinforcedLearningSystem(agents_base_url, string_comparison_url)
        self.monitor = RealTimeMonitor(agents_base_url)
        
        # Evolution parameters
        self.max_generations = 5
        self.agents_per_generation = 20
        self.elite_selection_rate = 0.2  # Top 20% for next generation
        self.mutation_rate = 0.3
        self.crossover_rate = 0.7
        
        # Performance tracking
        self.generation_history = []
        self.best_agents_history = []
        self.performance_evolution = []
        
    def check_services(self) -> bool:
        """Check if all required services are running"""
        print("🔍 Checking service status...")
        
        services = [
            ("Agents Service", self.agents_base_url),
            ("String Comparison", self.string_comparison_url)
        ]
        
        all_running = True
        for name, url in services:
            try:
                response = requests.get(f"{url}/health", timeout=5)
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
    
    def load_base_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load base training dataset"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                return dataset['training_data']
        except FileNotFoundError:
            print(f"❌ Dataset not found: {dataset_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in dataset: {e}")
            return None
    
    def create_initial_population(self, base_user_id: str, base_training_data: List[Dict]) -> List[Dict[str, Any]]:
        """Create initial population of agents"""
        print(f"🧬 Creating initial population of {self.agents_per_generation} agents...")
        
        results = self.rl_system.train_and_evaluate_agents(
            base_user_id=f"{base_user_id}_gen0",
            base_training_data=base_training_data,
            num_agents=self.agents_per_generation
        )
        
        if results:
            self.generation_history.append(results)
            self.best_agents_history.append(max(results, key=lambda x: x['average_similarity']))
            
        return results
    
    def select_elite_agents(self, generation_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select elite agents for next generation"""
        elite_count = int(len(generation_results) * self.elite_selection_rate)
        
        # Sort by performance (average similarity)
        sorted_results = sorted(generation_results, 
                              key=lambda x: x['average_similarity'], 
                              reverse=True)
        
        elite_agents = sorted_results[:elite_count]
        
        print(f"🏆 Selected {len(elite_agents)} elite agents for next generation")
        for i, agent in enumerate(elite_agents):
            print(f"   {i+1}. {agent['user_id']}: {agent['average_similarity']:.3f}")
        
        return elite_agents
    
    def create_offspring_configs(self, elite_agents: List[Dict[str, Any]], 
                                generation_num: int) -> List[Dict[str, Any]]:
        """Create offspring configurations using crossover and mutation"""
        offspring_configs = []
        
        # Keep elite agents
        for agent in elite_agents:
            offspring_configs.append(agent['config'])
        
        # Create new agents through crossover and mutation
        while len(offspring_configs) < self.agents_per_generation:
            if random.random() < self.crossover_rate and len(elite_agents) >= 2:
                # Crossover: combine two elite agents
                parent1, parent2 = random.sample(elite_agents, 2)
                offspring_config = self._crossover_configs(
                    parent1['config'], parent2['config']
                )
            else:
                # Mutation: modify a random elite agent
                parent = random.choice(elite_agents)
                offspring_config = self._mutate_config(parent['config'])
            
            offspring_configs.append(offspring_config)
        
        return offspring_configs[:self.agents_per_generation]
    
    def _crossover_configs(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform crossover between two agent configurations"""
        offspring = {}
        
        for key in config1.keys():
            if random.random() < 0.5:
                offspring[key] = config1[key]
            else:
                offspring[key] = config2[key]
        
        # Add some randomness
        offspring['learning_rate'] = random.uniform(
            min(config1['learning_rate'], config2['learning_rate']) * 0.5,
            max(config1['learning_rate'], config2['learning_rate']) * 1.5
        )
        
        return offspring
    
    def _mutate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate an agent configuration"""
        mutated = config.copy()
        
        # Mutate learning rate
        if random.random() < self.mutation_rate:
            mutated['learning_rate'] = random.uniform(1e-5, 1e-3)
        
        # Mutate batch size
        if random.random() < self.mutation_rate:
            mutated['batch_size'] = random.choice([8, 16, 32, 64])
        
        # Mutate temperature
        if random.random() < self.mutation_rate:
            mutated['temperature'] = random.uniform(0.3, 1.0)
        
        # Mutate max length
        if random.random() < self.mutation_rate:
            mutated['max_length'] = random.choice([256, 512, 768, 1024])
        
        return mutated
    
    def train_generation(self, generation_num: int, configs: List[Dict[str, Any]], 
                        base_training_data: List[Dict]) -> List[Dict[str, Any]]:
        """Train a generation of agents"""
        print(f"\n🧬 GENERATION {generation_num}")
        print("=" * 60)
        
        # Create agent configs with generation-specific user IDs
        agent_configs = []
        for i, config in enumerate(configs):
            from reinforced_learning_system import AgentConfig
            agent_config = AgentConfig(
                user_id=f"python_expert_gen{generation_num}_v{i+1:02d}",
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size'],
                epochs=config.get('epochs', 5),
                temperature=config['temperature'],
                top_p=config.get('top_p', 0.9),
                max_length=config['max_length'],
                dropout_rate=config.get('dropout_rate', 0.1)
            )
            agent_configs.append(agent_config)
        
        # Train and evaluate agents
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_config = {
                executor.submit(self.rl_system._train_single_agent, config, base_training_data): config 
                for config in agent_configs
            }
            
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"✅ {config.user_id}: {result['average_similarity']:.3f}")
                    else:
                        print(f"❌ {config.user_id}: Failed")
                except Exception as e:
                    print(f"❌ {config.user_id}: Error - {e}")
        
        # Store generation results
        self.generation_history.append(results)
        
        if results:
            best_agent = max(results, key=lambda x: x['average_similarity'])
            self.best_agents_history.append(best_agent)
            
            avg_performance = np.mean([r['average_similarity'] for r in results])
            self.performance_evolution.append({
                'generation': generation_num,
                'avg_performance': avg_performance,
                'best_performance': best_agent['average_similarity'],
                'agent_count': len(results)
            })
            
            print(f"\n📊 Generation {generation_num} Results:")
            print(f"   Average Performance: {avg_performance:.3f}")
            print(f"   Best Performance: {best_agent['average_similarity']:.3f}")
            print(f"   Best Agent: {best_agent['user_id']}")
        
        return results
    
    def run_evolution(self, base_user_id: str, base_training_data: List[Dict]):
        """Run complete evolutionary optimization"""
        print("🧬 EVOLUTIONARY AGENT OPTIMIZATION")
        print("=" * 80)
        print(f"Max Generations: {self.max_generations}")
        print(f"Agents per Generation: {self.agents_per_generation}")
        print(f"Elite Selection Rate: {self.elite_selection_rate}")
        print(f"Mutation Rate: {self.mutation_rate}")
        print("=" * 80)
        
        # Generation 0: Initial population
        generation_results = self.create_initial_population(base_user_id, base_training_data)
        
        if not generation_results:
            print("❌ Failed to create initial population")
            return None
        
        # Start real-time monitoring in separate thread
        agent_ids = [result['user_id'] for result in generation_results]
        monitor_thread = threading.Thread(
            target=self.monitor.start_monitoring,
            args=(agent_ids, 10),
            daemon=True
        )
        monitor_thread.start()
        
        # Evolution loop
        for generation in range(1, self.max_generations):
            print(f"\n🧬 EVOLUTION GENERATION {generation}")
            print("=" * 60)
            
            # Select elite agents from previous generation
            elite_agents = self.select_elite_agents(generation_results)
            
            # Create offspring configurations
            offspring_configs = self.create_offspring_configs(elite_agents, generation)
            
            # Train new generation
            generation_results = self.train_generation(
                generation, offspring_configs, base_training_data
            )
            
            if not generation_results:
                print(f"❌ Generation {generation} failed")
                break
            
            # Update monitoring with new agent IDs
            new_agent_ids = [result['user_id'] for result in generation_results]
            # Note: In a full implementation, we'd update the monitor's agent list
            
            # Check for convergence
            if len(self.performance_evolution) >= 3:
                recent_improvement = (
                    self.performance_evolution[-1]['best_performance'] - 
                    self.performance_evolution[-3]['best_performance']
                )
                if recent_improvement < 0.01:  # Less than 1% improvement
                    print(f"🎯 Convergence detected. Stopping evolution.")
                    break
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Generate final report
        self.generate_evolution_report()
        
        return self.best_agents_history[-1] if self.best_agents_history else None
    
    def generate_evolution_report(self):
        """Generate comprehensive evolution report"""
        print("\n" + "=" * 80)
        print("📊 EVOLUTIONARY OPTIMIZATION REPORT")
        print("=" * 80)
        
        if not self.performance_evolution:
            print("No evolution data available")
            return
        
        # Overall evolution statistics
        initial_performance = self.performance_evolution[0]['best_performance']
        final_performance = self.performance_evolution[-1]['best_performance']
        improvement = final_performance - initial_performance
        
        print(f"Generations Completed: {len(self.performance_evolution)}")
        print(f"Initial Best Performance: {initial_performance:.3f}")
        print(f"Final Best Performance: {final_performance:.3f}")
        print(f"Total Improvement: {improvement:.3f} ({improvement/initial_performance*100:.1f}%)")
        
        # Best agent across all generations
        if self.best_agents_history:
            best_overall = max(self.best_agents_history, key=lambda x: x['average_similarity'])
            print(f"\n🏆 BEST AGENT OVERALL:")
            print(f"   Agent ID: {best_overall['user_id']}")
            print(f"   Performance: {best_overall['average_similarity']:.3f}")
            print(f"   Success Rate: {best_overall['success_rate']:.3f}")
            print(f"   Configuration:")
            for key, value in best_overall['config'].items():
                print(f"     {key}: {value}")
        
        # Generation-by-generation analysis
        print(f"\n📈 GENERATION ANALYSIS:")
        for i, gen_data in enumerate(self.performance_evolution):
            print(f"Generation {gen_data['generation']}:")
            print(f"  Average: {gen_data['avg_performance']:.3f}")
            print(f"  Best: {gen_data['best_performance']:.3f}")
            print(f"  Agents: {gen_data['agent_count']}")
        
        # Save detailed report
        report_data = {
            'evolution_summary': {
                'total_generations': len(self.performance_evolution),
                'initial_performance': initial_performance,
                'final_performance': final_performance,
                'total_improvement': improvement,
                'improvement_percentage': improvement/initial_performance*100 if initial_performance > 0 else 0
            },
            'performance_evolution': self.performance_evolution,
            'best_agents_history': self.best_agents_history,
            'generation_history': self.generation_history,
            'parameters': {
                'max_generations': self.max_generations,
                'agents_per_generation': self.agents_per_generation,
                'elite_selection_rate': self.elite_selection_rate,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate
            },
            'timestamp': datetime.now().isoformat()
        }
        
        report_filename = f"evolution_report_{int(time.time())}.json"
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Detailed evolution report saved: {report_filename}")
        
        # Create evolution visualization
        self.create_evolution_visualization()
    
    def create_evolution_visualization(self):
        """Create visualization of evolution progress"""
        if not self.performance_evolution:
            return
        
        import matplotlib.pyplot as plt
        
        generations = [g['generation'] for g in self.performance_evolution]
        avg_performances = [g['avg_performance'] for g in self.performance_evolution]
        best_performances = [g['best_performance'] for g in self.performance_evolution]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(generations, avg_performances, 'b-o', label='Average Performance')
        plt.plot(generations, best_performances, 'r-s', label='Best Performance')
        plt.xlabel('Generation')
        plt.ylabel('Performance')
        plt.title('Evolution Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        improvements = [best_performances[i] - best_performances[0] for i in range(len(best_performances))]
        plt.bar(generations, improvements, alpha=0.7, color='green')
        plt.xlabel('Generation')
        plt.ylabel('Improvement from Initial')
        plt.title('Cumulative Improvement')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        agent_counts = [g['agent_count'] for g in self.performance_evolution]
        plt.bar(generations, agent_counts, alpha=0.7, color='orange')
        plt.xlabel('Generation')
        plt.ylabel('Number of Agents')
        plt.title('Population Size by Generation')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        if len(self.best_agents_history) > 1:
            config_evolution = []
            for agent in self.best_agents_history:
                config_evolution.append(agent['config']['learning_rate'])
            plt.plot(generations[:len(config_evolution)], config_evolution, 'g-o')
            plt.xlabel('Generation')
            plt.ylabel('Learning Rate')
            plt.title('Best Agent Learning Rate Evolution')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"evolution_visualization_{int(time.time())}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Evolution visualization saved: {plot_filename}")
        
        # Show plot
        try:
            plt.show()
        except:
            pass

def main():
    """Main function for evolutionary optimization"""
    print("🧬 EVOLUTIONARY AGENT OPTIMIZER")
    print("=" * 80)
    
    # Initialize optimizer
    optimizer = EvolutionaryAgentOptimizer()
    
    # Check services
    if not optimizer.check_services():
        print("\n❌ Required services are not running. Please start them:")
        print("   python3 app.py")
        print("   cd ../string-comparison && python3 backend.py")
        return
    
    # Load base training data
    base_training_data = optimizer.load_base_dataset('training_datasets/python_expert_dataset.json')
    if not base_training_data:
        print("❌ Failed to load base training dataset")
        return
    
    print(f"✅ Loaded base training dataset with {len(base_training_data)} samples")
    
    # Run evolution
    best_agent = optimizer.run_evolution("python_expert_evolved", base_training_data)
    
    if best_agent:
        print(f"\n🎉 EVOLUTION COMPLETED!")
        print(f"🏆 Best Agent: {best_agent['user_id']}")
        print(f"📊 Final Performance: {best_agent['average_similarity']:.3f}")
        print("=" * 80)
    else:
        print("\n❌ Evolution failed to produce results")

if __name__ == "__main__":
    main()
