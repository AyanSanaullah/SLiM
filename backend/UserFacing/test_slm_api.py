#!/usr/bin/env python3
"""
SLM API Test Script

This script demonstrates how to interact with your SLM API endpoints.
Run this after starting the API server to test functionality.

Usage:
    python test_slm_api.py [--host HOST] [--port PORT]
"""

import requests
import json
import time
import argparse
from typing import Dict, Any

class SLMAPITester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        
    def test_connection(self) -> bool:
        """Test if the API server is running"""
        try:
            response = requests.get(f"{self.base_url}/test", timeout=5)
            if response.status_code == 200:
                print("✅ API server is running")
                return True
            else:
                print(f"❌ API server returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to API server: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        print("\n📋 Getting model information...")
        try:
            response = requests.get(f"{self.base_url}/slm/info")
            data = response.json()
            
            print(f"   Model loaded: {data.get('model_loaded', False)}")
            print(f"   Device: {data.get('device', 'unknown')}")
            print(f"   Base model: {data.get('base_model', 'unknown')}")
            print(f"   CUDA available: {data.get('cuda_available', False)}")
            print(f"   MPS available: {data.get('mps_available', False)}")
            
            return data
        except Exception as e:
            print(f"❌ Error getting model info: {e}")
            return {}
    
    def load_model(self) -> bool:
        """Load the SLM model"""
        print("\n🔄 Loading SLM model...")
        try:
            response = requests.post(f"{self.base_url}/slm/load")
            data = response.json()
            
            if data.get('success', False):
                print(f"✅ {data.get('message', 'Model loaded')}")
                print(f"   Device: {data.get('device', 'unknown')}")
                return True
            else:
                print(f"❌ Failed to load model: {data.get('error', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def test_generation(self, prompt: str = "Explain what machine learning is in simple terms.") -> bool:
        """Test non-streaming generation"""
        print(f"\n🤖 Testing generation with prompt: '{prompt}'")
        try:
            payload = {
                "prompt": prompt,
                "max_new_tokens": 100,
                "temperature": 0.7
            }
            
            start_time = time.time()
            response = requests.post(f"{self.base_url}/slm/generate", json=payload)
            end_time = time.time()
            
            data = response.json()
            
            if data.get('success', False):
                print(f"✅ Generation successful (took {end_time - start_time:.2f}s)")
                print(f"   Device used: {data.get('device', 'unknown')}")
                print(f"   Response: {data.get('response', '')[:200]}...")
                return True
            else:
                print(f"❌ Generation failed: {data.get('error', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return False
    
    def test_streaming(self, prompt: str = "Write a short story about a robot.") -> bool:
        """Test streaming generation"""
        print(f"\n📡 Testing streaming with prompt: '{prompt}'")
        try:
            payload = {
                "prompt": prompt,
                "max_new_tokens": 80,
                "temperature": 0.8
            }
            
            response = requests.post(f"{self.base_url}/slm/stream", json=payload, stream=True)
            
            if response.status_code == 200:
                print("✅ Streaming response:")
                print("   ", end="", flush=True)
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])  # Remove 'data: ' prefix
                                if 'text' in data:
                                    print(data['text'], end="", flush=True)
                                elif 'status' in data:
                                    print(f"\n   Status: {data['status']}")
                                elif 'error' in data:
                                    print(f"\n❌ Error: {data['error']}")
                                    return False
                            except json.JSONDecodeError:
                                continue
                
                print("\n✅ Streaming completed")
                return True
            else:
                print(f"❌ Streaming failed with status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error during streaming: {e}")
            return False
    
    def unload_model(self) -> bool:
        """Unload the model"""
        print("\n🗑️  Unloading model...")
        try:
            response = requests.post(f"{self.base_url}/slm/unload")
            data = response.json()
            
            if data.get('success', False):
                print(f"✅ {data.get('message', 'Model unloaded')}")
                return True
            else:
                print(f"❌ Failed to unload model: {data.get('error', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"❌ Error unloading model: {e}")
            return False
    
    def run_full_test(self):
        """Run a complete test suite"""
        print("🧪 Running SLM API Test Suite")
        print("=" * 50)
        
        # Test connection
        if not self.test_connection():
            return False
        
        # Get initial model info
        self.get_model_info()
        
        # Load model
        if not self.load_model():
            return False
        
        # Test generation
        if not self.test_generation():
            return False
        
        # Test streaming
        if not self.test_streaming():
            return False
        
        # Get final model info
        self.get_model_info()
        
        print("\n🎉 All tests completed successfully!")
        print("💡 Your SLM API is working correctly!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Test the SLM API')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='API host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='API port (default: 5000)')
    parser.add_argument('--prompt', type=str, help='Custom prompt to test')
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    tester = SLMAPITester(base_url)
    
    if args.prompt:
        # Test with custom prompt
        print(f"🧪 Testing SLM API at {base_url}")
        if tester.test_connection():
            tester.load_model()
            tester.test_generation(args.prompt)
    else:
        # Run full test suite
        tester.run_full_test()

if __name__ == '__main__':
    main()
