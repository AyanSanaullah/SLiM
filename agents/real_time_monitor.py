#!/usr/bin/env python3
"""
Real-time monitoring system for reinforced learning agent performance
Provides live visualization and tracking of agent evolution
"""

import requests
import time
import json
import threading
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
from datetime import datetime
import os

class RealTimeMonitor:
    """Real-time monitoring system for agent performance"""
    
    def __init__(self, agents_base_url: str = "http://localhost:8080"):
        self.agents_base_url = agents_base_url
        self.session = requests.Session()
        
        # Performance tracking
        self.agent_data = {}
        self.performance_history = deque(maxlen=100)
        self.best_performers = []
        self.monitoring = False
        
        # Real-time data storage
        self.live_similarities = {}
        self.live_success_rates = {}
        self.training_status = {}
        
    def start_monitoring(self, agent_ids: List[str], update_interval: int = 10):
        """Start real-time monitoring of agents"""
        print("📊 Starting real-time monitoring...")
        self.monitoring = True
        
        # Initialize tracking for all agents
        for agent_id in agent_ids:
            self.live_similarities[agent_id] = deque(maxlen=50)
            self.live_success_rates[agent_id] = deque(maxlen=50)
            self.training_status[agent_id] = "unknown"
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(agent_ids, update_interval),
            daemon=True
        )
        monitor_thread.start()
        
        # Start visualization
        self._start_live_visualization(agent_ids)
    
    def _monitor_loop(self, agent_ids: List[str], update_interval: int):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                for agent_id in agent_ids:
                    self._update_agent_status(agent_id)
                    self._update_agent_performance(agent_id)
                
                time.sleep(update_interval)
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(update_interval)
    
    def _update_agent_status(self, agent_id: str):
        """Update agent training status"""
        try:
            response = self.session.get(f"{self.agents_base_url}/api/v1/agents/{agent_id}/status")
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'unknown')
                self.training_status[agent_id] = status
                
                # Log status changes
                if hasattr(self, '_last_status'):
                    if self._last_status.get(agent_id) != status:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp}] {agent_id}: {status}")
                else:
                    self._last_status = {}
                
                self._last_status[agent_id] = status
                
        except Exception as e:
            print(f"❌ Error checking status for {agent_id}: {e}")
    
    def _update_agent_performance(self, agent_id: str):
        """Update agent performance metrics"""
        if self.training_status.get(agent_id) != 'ready':
            return
        
        try:
            # Get recent performance data (if available)
            # This would typically come from a performance endpoint
            # For now, we'll simulate based on agent responses
            
            # Test with a simple question
            test_response = self.session.post(
                f"{self.agents_base_url}/api/v1/agents/{agent_id}/inference",
                json={"prompt": "What is Python?"},
                timeout=5
            )
            
            if test_response.status_code == 200:
                # Simulate performance metrics based on response quality
                response_data = test_response.json()
                response_length = len(response_data.get('response', ''))
                
                # Simple performance simulation (in real implementation, this would come from evaluation)
                simulated_similarity = min(0.95, response_length / 1000.0 + 0.3)
                simulated_success_rate = min(0.95, simulated_similarity + 0.1)
                
                self.live_similarities[agent_id].append(simulated_similarity)
                self.live_success_rates[agent_id].append(simulated_success_rate)
                
        except Exception as e:
            # If agent is not ready or error, add zero values
            self.live_similarities[agent_id].append(0.0)
            self.live_success_rates[agent_id].append(0.0)
    
    def _start_live_visualization(self, agent_ids: List[str]):
        """Start live visualization of agent performance"""
        print("📈 Starting live visualization...")
        
        # Set up the figure and axis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Real-time Agent Performance Monitoring', fontsize=16)
        
        # Initialize empty plots
        lines_similarity = {}
        lines_success = {}
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(agent_ids)))
        
        for i, agent_id in enumerate(agent_ids):
            # Similarity plot
            line, = ax1.plot([], [], label=agent_id, color=colors[i], alpha=0.7)
            lines_similarity[agent_id] = line
            
            # Success rate plot
            line, = ax2.plot([], [], label=agent_id, color=colors[i], alpha=0.7)
            lines_success[agent_id] = line
        
        # Configure plots
        ax1.set_title('Average Similarity Over Time')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Similarity Score')
        ax1.set_ylim(0, 1)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Success Rate Over Time')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Success Rate')
        ax2.set_ylim(0, 1)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Status plot
        ax3.set_title('Training Status')
        ax3.set_xlabel('Agents')
        ax3.set_ylabel('Status')
        status_bars = ax3.bar(range(len(agent_ids)), [0]*len(agent_ids))
        ax3.set_xticks(range(len(agent_ids)))
        ax3.set_xticklabels([aid.split('_')[-1] for aid in agent_ids], rotation=45)
        ax3.set_ylim(0, 3)
        
        # Performance ranking
        ax4.set_title('Current Performance Ranking')
        ax4.set_xlabel('Similarity Score')
        ax4.set_ylabel('Agents')
        ranking_bars = ax4.barh(range(len(agent_ids)), [0]*len(agent_ids))
        ax4.set_yticks(range(len(agent_ids)))
        ax4.set_yticklabels([aid.split('_')[-1] for aid in agent_ids])
        ax4.set_xlim(0, 1)
        
        def animate(frame):
            """Animation function for live updates"""
            try:
                # Update similarity plot
                for agent_id in agent_ids:
                    similarities = list(self.live_similarities[agent_id])
                    if similarities:
                        lines_similarity[agent_id].set_data(range(len(similarities)), similarities)
                
                # Update success rate plot
                for agent_id in agent_ids:
                    success_rates = list(self.live_success_rates[agent_id])
                    if success_rates:
                        lines_success[agent_id].set_data(range(len(success_rates)), success_rates)
                
                # Auto-scale plots
                for ax in [ax1, ax2]:
                    ax.relim()
                    ax.autoscale_view()
                
                # Update status bars
                status_values = []
                status_colors = []
                for agent_id in agent_ids:
                    status = self.training_status.get(agent_id, 'unknown')
                    if status == 'ready':
                        status_values.append(2)
                        status_colors.append('green')
                    elif status == 'processing':
                        status_values.append(1)
                        status_colors.append('orange')
                    elif status == 'error':
                        status_values.append(0.5)
                        status_colors.append('red')
                    else:
                        status_values.append(0)
                        status_colors.append('gray')
                
                for bar, value, color in zip(status_bars, status_values, status_colors):
                    bar.set_height(value)
                    bar.set_color(color)
                
                # Update performance ranking
                current_performances = []
                for agent_id in agent_ids:
                    similarities = list(self.live_similarities[agent_id])
                    if similarities:
                        current_performances.append(similarities[-1])
                    else:
                        current_performances.append(0.0)
                
                # Sort by performance
                sorted_indices = np.argsort(current_performances)[::-1]
                sorted_performances = [current_performances[i] for i in sorted_indices]
                sorted_agent_labels = [agent_ids[i].split('_')[-1] for i in sorted_indices]
                
                ax4.clear()
                bars = ax4.barh(range(len(sorted_agent_labels)), sorted_performances)
                ax4.set_yticks(range(len(sorted_agent_labels)))
                ax4.set_yticklabels(sorted_agent_labels)
                ax4.set_xlim(0, 1)
                ax4.set_title('Current Performance Ranking')
                ax4.set_xlabel('Similarity Score')
                ax4.set_ylabel('Agents')
                
                # Color bars based on performance
                for i, bar in enumerate(bars):
                    if i == 0:  # Best performer
                        bar.set_color('gold')
                    elif i < 3:  # Top 3
                        bar.set_color('lightblue')
                    else:
                        bar.set_color('lightgray')
                
            except Exception as e:
                print(f"❌ Animation error: {e}")
        
        # Start animation
        ani = animation.FuncAnimation(fig, animate, interval=2000, blit=False)
        
        # Show plot
        plt.tight_layout()
        plt.show()
        
        # Keep monitoring until window is closed
        self.monitoring = True
        try:
            while self.monitoring:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        print("🛑 Stopping real-time monitoring...")
        self.monitoring = False
    
    def generate_performance_summary(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Generate performance summary for all agents"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'agents': {},
            'overall_stats': {}
        }
        
        all_similarities = []
        all_success_rates = []
        
        for agent_id in agent_ids:
            similarities = list(self.live_similarities[agent_id])
            success_rates = list(self.live_success_rates[agent_id])
            status = self.training_status.get(agent_id, 'unknown')
            
            agent_summary = {
                'status': status,
                'latest_similarity': similarities[-1] if similarities else 0.0,
                'latest_success_rate': success_rates[-1] if success_rates else 0.0,
                'avg_similarity': np.mean(similarities) if similarities else 0.0,
                'avg_success_rate': np.mean(success_rates) if success_rates else 0.0,
                'data_points': len(similarities)
            }
            
            summary['agents'][agent_id] = agent_summary
            
            if similarities:
                all_similarities.extend(similarities)
            if success_rates:
                all_success_rates.extend(success_rates)
        
        # Overall statistics
        summary['overall_stats'] = {
            'total_agents': len(agent_ids),
            'ready_agents': len([aid for aid in agent_ids if self.training_status.get(aid) == 'ready']),
            'avg_similarity': np.mean(all_similarities) if all_similarities else 0.0,
            'avg_success_rate': np.mean(all_success_rates) if all_success_rates else 0.0,
            'best_similarity': np.max(all_similarities) if all_similarities else 0.0,
            'best_success_rate': np.max(all_success_rates) if all_success_rates else 0.0
        }
        
        return summary

def main():
    """Main function for real-time monitoring"""
    print("📊 REAL-TIME AGENT PERFORMANCE MONITOR")
    print("=" * 50)
    
    # Example agent IDs (these would come from the reinforced learning system)
    agent_ids = [f"python_expert_evolved_v{i+1:02d}" for i in range(20)]
    
    # Initialize monitor
    monitor = RealTimeMonitor()
    
    try:
        # Start monitoring
        monitor.start_monitoring(agent_ids, update_interval=5)
        
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
        monitor.stop_monitoring()
        
        # Generate final summary
        summary = monitor.generate_performance_summary(agent_ids)
        print("\n📊 FINAL PERFORMANCE SUMMARY:")
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
