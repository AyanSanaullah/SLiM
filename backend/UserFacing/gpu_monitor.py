"""
GPU Monitoring System
Real-time GPU usage, memory, and temperature monitoring
"""

import json
import time
import subprocess
import threading
from datetime import datetime
from typing import Dict, List, Optional
import os

class GPUMonitor:
    def __init__(self):
        self.gpu_data = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.max_data_points = 24  # Store last 24 data points for graph
        
    def get_nvidia_gpu_info(self) -> Optional[Dict]:
        """Get NVIDIA GPU information using nvidia-ml-py or nvidia-smi"""
        try:
            # Try using nvidia-ml-py first (more accurate)
            try:
                import pynvml
                pynvml.nvmlInit()
                
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count == 0:
                    return None
                
                # Get first GPU (can be extended for multiple GPUs)
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Get GPU info
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Get utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                memory_util = util.memory
                
                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used = mem_info.used // (1024 * 1024)  # Convert to MB
                memory_total = mem_info.total // (1024 * 1024)  # Convert to MB
                memory_percent = (memory_used / memory_total) * 100
                
                # Get temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = 0
                
                # Get power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # Convert to watts
                except:
                    power = 0
                
                # Get clock speeds
                try:
                    graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                except:
                    graphics_clock = 0
                    memory_clock = 0
                
                return {
                    'name': name,
                    'gpu_utilization': gpu_util,
                    'memory_utilization': memory_percent,
                    'memory_used': memory_used,
                    'memory_total': memory_total,
                    'temperature': temp,
                    'power_usage': power,
                    'graphics_clock': graphics_clock,
                    'memory_clock': memory_clock,
                    'timestamp': datetime.now().isoformat()
                }
                
            except ImportError:
                # Fallback to nvidia-smi
                return self.get_nvidia_smi_info()
                
        except Exception as e:
            print(f"Error getting NVIDIA GPU info: {e}")
            return None
    
    def get_nvidia_smi_info(self) -> Optional[Dict]:
        """Fallback method using nvidia-smi command"""
        try:
            # Run nvidia-smi command
            cmd = [
                'nvidia-smi',
                '--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.gr,clocks.mem',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                return None
            
            # Parse the output
            line = result.stdout.strip()
            if not line:
                return None
            
            parts = [part.strip() for part in line.split(',')]
            
            if len(parts) >= 6:
                name = parts[0]
                gpu_util = float(parts[1]) if parts[1] != '[Not Supported]' else 0
                memory_util = float(parts[2]) if parts[2] != '[Not Supported]' else 0
                memory_used = float(parts[3]) if parts[3] != '[Not Supported]' else 0
                memory_total = float(parts[4]) if parts[4] != '[Not Supported]' else 0
                temp = float(parts[5]) if parts[5] != '[Not Supported]' else 0
                
                power = 0
                graphics_clock = 0
                memory_clock = 0
                
                if len(parts) > 6 and parts[6] != '[Not Supported]':
                    power = float(parts[6])
                if len(parts) > 7 and parts[7] != '[Not Supported]':
                    graphics_clock = float(parts[7])
                if len(parts) > 8 and parts[8] != '[Not Supported]':
                    memory_clock = float(parts[8])
                
                memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0
                
                return {
                    'name': name,
                    'gpu_utilization': gpu_util,
                    'memory_utilization': memory_percent,
                    'memory_used': memory_used,
                    'memory_total': memory_total,
                    'temperature': temp,
                    'power_usage': power,
                    'graphics_clock': graphics_clock,
                    'memory_clock': memory_clock,
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            print(f"Error running nvidia-smi: {e}")
            return None
    
    def get_amd_gpu_info(self) -> Optional[Dict]:
        """Get AMD GPU information (basic implementation)"""
        try:
            # Try using rocm-smi for AMD GPUs
            cmd = ['rocm-smi', '--showuse', '--showmemuse', '--showtemp']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # Parse rocm-smi output (simplified)
                return {
                    'name': 'AMD GPU',
                    'gpu_utilization': 0,  # Would need to parse actual output
                    'memory_utilization': 0,
                    'memory_used': 0,
                    'memory_total': 0,
                    'temperature': 0,
                    'power_usage': 0,
                    'graphics_clock': 0,
                    'memory_clock': 0,
                    'timestamp': datetime.now().isoformat()
                }
        except:
            pass
        
        return None
    
    def get_gpu_info(self) -> Optional[Dict]:
        """Get GPU information from any available GPU"""
        # Try NVIDIA first
        gpu_info = self.get_nvidia_gpu_info()
        if gpu_info:
            return gpu_info
        
        # Try AMD
        gpu_info = self.get_amd_gpu_info()
        if gpu_info:
            return gpu_info
        
        # Return mock data if no GPU detected (for testing)
        return {
            'name': 'No GPU Detected',
            'gpu_utilization': 0,
            'memory_utilization': 0,
            'memory_used': 0,
            'memory_total': 0,
            'temperature': 0,
            'power_usage': 0,
            'graphics_clock': 0,
            'memory_clock': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def monitor_loop(self):
        """Main monitoring loop"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.is_monitoring:
            try:
                gpu_info = self.get_gpu_info()
                if gpu_info:
                    self.gpu_data.append(gpu_info)
                    
                    # Keep only the last N data points
                    if len(self.gpu_data) > self.max_data_points:
                        self.gpu_data.pop(0)
                    
                    # Reset error counter on success
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                
                time.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                consecutive_errors += 1
                print(f"Error in monitoring loop: {e}")
                
                # If too many consecutive errors, increase sleep time
                if consecutive_errors > max_consecutive_errors:
                    print(f"Too many consecutive errors ({consecutive_errors}), slowing down monitoring...")
                    time.sleep(10)  # Wait longer after many errors
                else:
                    time.sleep(5)  # Wait longer on error
                
                # Reset error counter if it gets too high
                if consecutive_errors > max_consecutive_errors * 2:
                    consecutive_errors = 0
    
    def start_monitoring(self):
        """Start GPU monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.monitor_thread.start()
            print("GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("GPU monitoring stopped")
    
    def get_current_data(self) -> Dict:
        """Get current GPU data"""
        if not self.gpu_data:
            return self.get_gpu_info() or {}
        return self.gpu_data[-1]
    
    def get_historical_data(self) -> List[Dict]:
        """Get historical GPU data for graphing"""
        return self.gpu_data.copy()
    
    def get_graph_data(self) -> Dict:
        """Get data formatted for the dashboard graph"""
        if not self.gpu_data:
            # Return empty data
            return {
                'labels': [],
                'gpu_usage': [],
                'memory_usage': [],
                'temperature': [],
                'current': self.get_gpu_info() or {}
            }
        
        # Generate time labels (last 24 points, 2 seconds apart = 48 seconds of data)
        labels = []
        gpu_usage = []
        memory_usage = []
        temperature = []
        
        for i, data in enumerate(self.gpu_data):
            # Create time labels (relative to now)
            seconds_ago = (len(self.gpu_data) - 1 - i) * 2
            if seconds_ago == 0:
                labels.append('Now')
            elif seconds_ago < 60:
                labels.append(f'-{seconds_ago}s')
            else:
                minutes_ago = seconds_ago // 60
                labels.append(f'-{minutes_ago}m')
            
            gpu_usage.append(data.get('gpu_utilization', 0))
            memory_usage.append(data.get('memory_utilization', 0))
            temperature.append(data.get('temperature', 0))
        
        return {
            'labels': labels,
            'gpu_usage': gpu_usage,
            'memory_usage': memory_usage,
            'temperature': temperature,
            'current': self.gpu_data[-1] if self.gpu_data else {}
        }

# Global GPU monitor instance
gpu_monitor = GPUMonitor()

def install_gpu_dependencies():
    """Install required GPU monitoring dependencies"""
    try:
        import pynvml
        print("✅ pynvml already installed")
    except ImportError:
        print("Installing pynvml for NVIDIA GPU monitoring...")
        try:
            subprocess.run(['pip', 'install', 'nvidia-ml-py'], check=True)
            print("✅ pynvml installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install pynvml. GPU monitoring may be limited.")

if __name__ == "__main__":
    # Test the GPU monitor
    install_gpu_dependencies()
    
    monitor = GPUMonitor()
    monitor.start_monitoring()
    
    try:
        for i in range(10):
            time.sleep(3)
            current = monitor.get_current_data()
            print(f"GPU: {current.get('gpu_utilization', 0)}% | "
                  f"Memory: {current.get('memory_utilization', 0):.1f}% | "
                  f"Temp: {current.get('temperature', 0)}°C")
    finally:
        monitor.stop_monitoring()
