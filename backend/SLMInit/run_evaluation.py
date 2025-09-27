#!/usr/bin/env python3
"""
Model Evaluation Runner

This script runs model evaluation and sends results to the frontend API.
It automatically detects which models are available and runs the appropriate tests.

Usage:
    python run_evaluation.py [--model-type cuda|cpu|both] [--api-url URL]
"""

import argparse
import os
import sys
import subprocess
import time

def check_model_exists(model_type):
    """Check if a trained model exists"""
    if model_type.lower() == "cuda":
        return os.path.exists("./cuda_lora_out")
    elif model_type.lower() == "cpu":
        return os.path.exists("./cpu_lora_out")
    return False

def check_test_data_exists(model_type):
    """Check if test data exists"""
    if model_type.lower() == "cuda":
        return os.path.exists("../UserFacing/db/LLMTestData.json")
    elif model_type.lower() == "cpu":
        return os.path.exists("../UserFacing/db/LLMTestData_CPU.json")
    return False

def run_evaluation_script(model_type):
    """Run the appropriate evaluation script"""
    if model_type.lower() == "cuda":
        script_name = "testSuite_enhanced.py"
    elif model_type.lower() == "cpu":
        script_name = "testSuite_enhanced_cpu.py"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if not os.path.exists(script_name):
        raise FileNotFoundError(f"Evaluation script not found: {script_name}")
    
    print(f"🚀 Running {model_type.upper()} evaluation...")
    print(f"   Script: {script_name}")
    
    # Run the evaluation script
    result = subprocess.run([sys.executable, script_name], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {model_type.upper()} evaluation completed successfully")
        return True
    else:
        print(f"❌ {model_type.upper()} evaluation failed:")
        print(f"   Error: {result.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run model evaluation')
    parser.add_argument('--model-type', choices=['cuda', 'cpu', 'both'], 
                       default='both', help='Which model to evaluate')
    parser.add_argument('--api-url', default='http://localhost:5000',
                       help='Base URL for the API server')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip model and data existence checks')
    
    args = parser.parse_args()
    
    print("🧪 Model Evaluation Runner")
    print("=" * 50)
    
    models_to_test = []
    
    if args.model_type == 'both':
        models_to_test = ['cuda', 'cpu']
    else:
        models_to_test = [args.model_type]
    
    success_count = 0
    total_count = 0
    
    for model_type in models_to_test:
        total_count += 1
        print(f"\n📋 Checking {model_type.upper()} model...")
        
        if not args.skip_checks:
            # Check if model exists
            if not check_model_exists(model_type):
                print(f"❌ {model_type.upper()} model not found")
                print(f"   Train the model first using: python cudaInit{'_cpu' if model_type == 'cpu' else ''}.py")
                continue
            
            # Check if test data exists
            if not check_test_data_exists(model_type):
                print(f"❌ {model_type.upper()} test data not found")
                print(f"   Run training first to generate test data")
                continue
            
            print(f"✅ {model_type.upper()} model and test data found")
        
        # Run evaluation
        try:
            if run_evaluation_script(model_type):
                success_count += 1
            time.sleep(2)  # Brief pause between evaluations
        except Exception as e:
            print(f"❌ Error running {model_type.upper()} evaluation: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total evaluations: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    
    if success_count > 0:
        print(f"\n✅ {success_count} evaluation(s) completed successfully!")
        print("📈 Check the API endpoints for detailed results:")
        print(f"   • {args.api_url}/test_results/summary")
        print(f"   • {args.api_url}/test_results/list")
    else:
        print("\n❌ No evaluations completed successfully")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
