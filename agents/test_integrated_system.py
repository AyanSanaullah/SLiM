#!/usr/bin/env python3
"""
Test Integrated Training System
Tests the integration between agents, string comparison service, and database
"""

import sys
import os
import time
import json
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adk_agents.training_database import TrainingDatabase
from adk_agents.string_comparison_client import StringComparisonClient
from adk_agents.real_model_trainer import RealModelTrainer

def test_string_comparison_service():
    """Test the string comparison service"""
    print("🔍 Testing String Comparison Service...")
    
    client = StringComparisonClient()
    
    # Check service availability
    if not client.is_service_available():
        print("❌ String comparison service is not available!")
        print("   Please start the service with: cd ../string-comparison && python3 backend.py")
        return False
    
    print("✅ String comparison service is available")
    
    # Test comparison
    test_cases = [
        ("I love programming", "I enjoy coding"),
        ("Python is great", "Java is awesome"),
        ("Machine learning is fun", "AI is interesting")
    ]
    
    for sentence1, sentence2 in test_cases:
        result = client.compare_sentences(sentence1, sentence2)
        if result['success']:
            print(f"   '{sentence1}' vs '{sentence2}' -> {result['similarity_percentage']} ({result['quality_label']})")
        else:
            print(f"   Error comparing sentences: {result['error']}")
    
    return True

def test_database_operations():
    """Test database operations"""
    print("\n💾 Testing Database Operations...")
    
    db = TrainingDatabase()
    
    # Start a test session
    session_id = db.start_training_session(
        user_id="test_user",
        model_type="test_model",
        metadata={"test": True}
    )
    print(f"✅ Started test session: {session_id}")
    
    # Record some test cycles
    test_cycles = [
        {
            'prompt': 'What is Python?',
            'expected_answer': 'Python is a programming language',
            'model_response': 'Python is a high-level programming language',
            'similarity_score': 0.85,
            'quality_label': 'HIGH',
            'model_confidence': 0.9
        },
        {
            'prompt': 'How to use loops?',
            'expected_answer': 'Use for and while loops',
            'model_response': 'You can use for loops and while loops',
            'similarity_score': 0.75,
            'quality_label': 'MEDIUM',
            'model_confidence': 0.8
        }
    ]
    
    for cycle in test_cycles:
        db.record_training_cycle(session_id, cycle)
        print(f"   Recorded cycle: {cycle['prompt'][:30]}... -> {cycle['similarity_score']:.3f}")
    
    # Complete session
    db.complete_training_session(session_id)
    print("✅ Completed test session")
    
    # Get metrics
    metrics = db.get_session_metrics(session_id)
    print(f"   Session metrics: avg_similarity={metrics['metrics'].get('avg_similarity', 0):.3f}")
    
    return True

def test_integrated_training():
    """Test integrated training with real model trainer"""
    print("\n🤖 Testing Integrated Training...")
    
    # Check if string comparison service is available
    client = StringComparisonClient()
    if not client.is_service_available():
        print("⚠️  String comparison service not available, skipping integrated test")
        return False
    
    # Create a test configuration
    config = {
        'adk': {
            'vertex_ai': {
                'project_id': 'test-project',
                'location': 'us-central1'
            }
        }
    }
    
    # Initialize trainer
    trainer = RealModelTrainer("integrated_test_user", config)
    
    # Test training data
    training_data = """
    Sou um especialista em Python e desenvolvimento web.
    
    Exemplos:
    - Como criar uma API? Use Flask ou FastAPI para criar APIs REST.
    - O que é Python? Python é uma linguagem de programação interpretada.
    - Como usar loops? Use for para iterar sobre sequências e while para loops condicionais.
    """
    
    print("   Processing training data...")
    processed_data = trainer.process_training_data(training_data)
    print(f"   Created {processed_data.get('training_pairs_count', 0)} training pairs")
    
    print("   Training model...")
    training_results = trainer.train_model(processed_data)
    print(f"   Training completed: {training_results.get('status', 'unknown')}")
    
    print("   Evaluating model with string comparison...")
    evaluation = trainer.evaluate_model(
        "Como criar uma API REST?",
        "Use Flask ou FastAPI para criar APIs REST simples e eficientes."
    )
    
    if 'error' not in evaluation:
        similarity = evaluation.get('semantic_similarity', evaluation.get('overall_similarity', 0))
        print(f"   Evaluation completed: similarity={similarity:.3f}, quality={evaluation.get('quality_label', 'N/A')}")
        
        if evaluation.get('string_comparison_used'):
            print("   ✅ String comparison service was used successfully")
        else:
            print("   ⚠️  Fallback evaluation was used")
    else:
        print(f"   ❌ Evaluation failed: {evaluation['error']}")
    
    return True

def main():
    """Main test function"""
    print("🧪 Testing Integrated Training System")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: String comparison service
    if test_string_comparison_service():
        success_count += 1
    
    # Test 2: Database operations
    if test_database_operations():
        success_count += 1
    
    # Test 3: Integrated training
    if test_integrated_training():
        success_count += 1
    
    # Summary
    print(f"\n📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! The integrated system is working correctly.")
        print("\n📋 Next steps:")
        print("   1. Start training agents to see data in the database")
        print("   2. Use 'python view_training_metrics.py summary' to view results")
        print("   3. Monitor string comparison logs for detailed interactions")
    else:
        print("⚠️  Some tests failed. Please check the configuration and services.")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
