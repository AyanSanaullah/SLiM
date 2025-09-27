"""
CUDA Processing Endpoints for GPU-based AI processing
Handles CUDA initialization, training, and testing operations
"""

from flask import Blueprint, request, jsonify
import subprocess
import os
import sys
import json
import traceback
from pathlib import Path

# Create blueprint for CUDA endpoints
cuda_bp = Blueprint('cuda', __name__, url_prefix='/cuda')

# Path to SLMInit directory
SLMINIT_PATH = os.path.join(os.path.dirname(__file__), '..', 'SLMInit')

@cuda_bp.route('/check', methods=['GET'])
def check_cuda():
    """Check if CUDA is available on the system"""
    try:
        # Try to import torch and check CUDA availability
        result = subprocess.run([
            sys.executable, '-c', 
            'import torch; print(f"CUDA_AVAILABLE:{torch.cuda.is_available()}"); print(f"CUDA_DEVICES:{torch.cuda.device_count()}"); print(f"CUDA_VERSION:{torch.version.cuda if torch.cuda.is_available() else None}")'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            output_lines = result.stdout.strip().split('\n')
            cuda_info = {}
            
            for line in output_lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    if key == 'CUDA_AVAILABLE':
                        cuda_info['available'] = value.lower() == 'true'
                    elif key == 'CUDA_DEVICES':
                        cuda_info['device_count'] = int(value) if value.isdigit() else 0
                    elif key == 'CUDA_VERSION':
                        cuda_info['version'] = value if value != 'None' else None
            
            return jsonify({
                'success': True,
                'cuda_info': cuda_info,
                'message': 'CUDA check completed successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'CUDA check failed: {result.stderr}',
                'message': 'CUDA not available or PyTorch not installed with CUDA support'
            }), 400
            
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': 'CUDA check timed out',
            'message': 'System took too long to respond'
        }), 408
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error checking CUDA availability'
        }), 500

@cuda_bp.route('/init', methods=['POST'])
def cuda_init():
    """Initialize CUDA training using cudaInit.py"""
    try:
        data = request.get_json() or {}
        
        # Get parameters with defaults
        base_model = data.get('base_model', 'gpt2')
        data_path = data.get('data_path', '../UserFacing/db/LLMData.json')
        output_dir = data.get('output_dir', './cuda_lora_out')
        
        # Validate data file exists
        full_data_path = os.path.join(SLMINIT_PATH, data_path)
        if not os.path.exists(full_data_path):
            return jsonify({
                'success': False,
                'error': f'Training data file not found: {full_data_path}',
                'message': 'Please ensure LLM data is available before training'
            }), 400
        
        # Change to SLMInit directory and run cudaInit.py
        cuda_init_path = os.path.join(SLMINIT_PATH, 'cudaInit.py')
        
        if not os.path.exists(cuda_init_path):
            return jsonify({
                'success': False,
                'error': f'cudaInit.py not found at {cuda_init_path}',
                'message': 'CUDA initialization script missing'
            }), 404
        
        # Set environment variables for the script
        env = os.environ.copy()
        env['PYTHONPATH'] = SLMINIT_PATH
        
        # Run the CUDA initialization script
        result = subprocess.run([
            sys.executable, cuda_init_path
        ], capture_output=True, text=True, cwd=SLMINIT_PATH, env=env, timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            return jsonify({
                'success': True,
                'message': 'CUDA initialization completed successfully',
                'output': result.stdout,
                'base_model': base_model,
                'output_dir': output_dir
            })
        else:
            return jsonify({
                'success': False,
                'error': result.stderr,
                'output': result.stdout,
                'message': 'CUDA initialization failed'
            }), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': 'CUDA initialization timed out',
            'message': 'Training took too long to complete (>30 minutes)'
        }), 408
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'message': 'Error during CUDA initialization'
        }), 500

@cuda_bp.route('/test', methods=['POST'])
def cuda_test():
    """Run CUDA test suite using testSuite.py"""
    try:
        data = request.get_json() or {}
        
        # Get parameters with defaults
        base_model = data.get('base_model', 'gpt2')
        lora_model_path = data.get('lora_model_path', './cuda_lora_out')
        
        # Validate LoRA model exists
        full_lora_path = os.path.join(SLMINIT_PATH, lora_model_path)
        if not os.path.exists(full_lora_path):
            return jsonify({
                'success': False,
                'error': f'LoRA model not found: {full_lora_path}',
                'message': 'Please run CUDA initialization first'
            }), 400
        
        # Change to SLMInit directory and run testSuite.py
        test_suite_path = os.path.join(SLMINIT_PATH, 'testSuite.py')
        
        if not os.path.exists(test_suite_path):
            return jsonify({
                'success': False,
                'error': f'testSuite.py not found at {test_suite_path}',
                'message': 'CUDA test suite script missing'
            }), 404
        
        # Set environment variables for the script
        env = os.environ.copy()
        env['PYTHONPATH'] = SLMINIT_PATH
        
        # Run the CUDA test suite
        result = subprocess.run([
            sys.executable, test_suite_path
        ], capture_output=True, text=True, cwd=SLMINIT_PATH, env=env, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            # Parse the output to extract the generated response
            output_lines = result.stdout.strip().split('\n')
            generated_response = ""
            
            # Look for the generated response in the output
            for i, line in enumerate(output_lines):
                if "Generated response:" in line:
                    # Capture everything after this line as the response
                    generated_response = '\n'.join(output_lines[i+1:])
                    break
            
            return jsonify({
                'success': True,
                'message': 'CUDA test suite completed successfully',
                'output': result.stdout,
                'generated_response': generated_response,
                'base_model': base_model,
                'lora_model_path': lora_model_path
            })
        else:
            return jsonify({
                'success': False,
                'error': result.stderr,
                'output': result.stdout,
                'message': 'CUDA test suite failed'
            }), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': 'CUDA test suite timed out',
            'message': 'Test took too long to complete (>5 minutes)'
        }), 408
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'message': 'Error during CUDA testing'
        }), 500

@cuda_bp.route('/status', methods=['GET'])
def cuda_status():
    """Get status of CUDA models and training"""
    try:
        status_info = {
            'cuda_available': False,
            'models': [],
            'training_data_available': False,
            'last_training': None
        }
        
        # Check CUDA availability
        try:
            result = subprocess.run([
                sys.executable, '-c', 
                'import torch; print(torch.cuda.is_available())'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                status_info['cuda_available'] = result.stdout.strip().lower() == 'true'
        except:
            pass
        
        # Check for trained models
        models_dir = os.path.join(SLMINIT_PATH, 'cuda_lora_out')
        if os.path.exists(models_dir):
            model_files = []
            for file in os.listdir(models_dir):
                if file.endswith(('.bin', '.safetensors', '.json')):
                    file_path = os.path.join(models_dir, file)
                    model_files.append({
                        'name': file,
                        'size': os.path.getsize(file_path),
                        'modified': os.path.getmtime(file_path)
                    })
            status_info['models'] = model_files
        
        # Check for training data
        data_path = os.path.join(SLMINIT_PATH, '../UserFacing/db/LLMData.json')
        status_info['training_data_available'] = os.path.exists(data_path)
        
        return jsonify({
            'success': True,
            'status': status_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error getting CUDA status'
        }), 500

# Error handlers for the blueprint
@cuda_bp.errorhandler(404)
def cuda_not_found(error):
    return jsonify({
        'success': False,
        'error': 'CUDA endpoint not found',
        'message': 'The requested CUDA operation is not available'
    }), 404

@cuda_bp.errorhandler(500)
def cuda_internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal CUDA processing error',
        'message': 'An unexpected error occurred during CUDA processing'
    }), 500
