#!/usr/bin/env python3
"""
SLM API Startup Script

This script starts the Flask API server with your trained Small Language Model.
It provides a simple way to serve your LoRA fine-tuned model via REST API.

Usage:
    python start_slm_api.py [--port PORT] [--host HOST] [--debug]

Examples:
    python start_slm_api.py                    # Start on localhost:5000
    python start_slm_api.py --port 8080        # Start on localhost:8080
    python start_slm_api.py --host 0.0.0.0     # Allow external connections
    python start_slm_api.py --debug            # Enable debug mode
"""

import argparse
import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'torch', 'transformers', 'peft', 'flask', 'flask-cors'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_model_files():
    """Check if trained model files exist"""
    cuda_model_path = "../SLMInit/cuda_lora_out"
    cpu_model_path = "../SLMInit/cpu_lora_out"
    
    cuda_exists = os.path.exists(cuda_model_path)
    cpu_exists = os.path.exists(cpu_model_path)
    
    if cuda_exists:
        print(f"✅ CUDA model found at: {cuda_model_path}")
    if cpu_exists:
        print(f"✅ CPU model found at: {cpu_model_path}")
    
    if not cuda_exists and not cpu_exists:
        print("❌ No trained models found!")
        print("💡 Train a model first using:")
        print("   - For CUDA: python ../SLMInit/cudaInit.py")
        print("   - For CPU:  python ../SLMInit/cudaInit_cpu.py")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Start the SLM API server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on (default: 5000)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on (default: 127.0.0.1)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-checks', action='store_true', help='Skip dependency and model checks')
    
    args = parser.parse_args()
    
    print("🚀 Starting SLM API Server")
    print("=" * 50)
    
    if not args.no_checks:
        print("🔍 Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        
        print("\n🔍 Checking for trained models...")
        if not check_model_files():
            sys.exit(1)
    
    print(f"\n🌐 Starting server on http://{args.host}:{args.port}")
    print("📋 Available endpoints:")
    print(f"   • GET  http://{args.host}:{args.port}/slm/info      - Model information")
    print(f"   • POST http://{args.host}:{args.port}/slm/load      - Load model")
    print(f"   • POST http://{args.host}:{args.port}/slm/generate  - Generate response")
    print(f"   • POST http://{args.host}:{args.port}/slm/stream    - Stream response")
    print(f"   • POST http://{args.host}:{args.port}/slm/unload    - Unload model")
    print("\n💡 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Set environment variables for Flask
    os.environ['FLASK_APP'] = 'app.py'
    if args.debug:
        os.environ['FLASK_ENV'] = 'development'
        os.environ['FLASK_DEBUG'] = '1'
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True  # Enable threading for better performance
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
