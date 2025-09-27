"""
Main Flask application for hosting Google ADK Agents
Provides REST API endpoints for agent management and inference
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import yaml
from datetime import datetime
import json
from typing import Dict, Any, Optional
import traceback

# Import our ADK agents
from adk_agents.user_agent_manager import UserAgentManager
from adk_agents.vertex_client import VertexAIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
config_path = os.getenv('ADK_CONFIG_PATH', 'config/adk_config.yaml')
vertex_config_path = os.getenv('VERTEX_CONFIG_PATH', 'config/vertex_config.yaml')
project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'your-project-id')
location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')

# Initialize components
user_agent_manager = UserAgentManager(config_path)
vertex_client = None  # Will be initialized when needed

def get_vertex_client():
    """Get or create Vertex AI client"""
    global vertex_client
    if vertex_client is None:
        try:
            vertex_client = VertexAIClient(project_id, location, vertex_config_path)
        except Exception as e:
            logger.warning(f"Could not initialize Vertex AI client: {e}")
            return None
    return vertex_client

@app.route('/', methods=['GET'])
def home():
    """Home page with API information"""
    return jsonify({
        'service': 'ShellHacks ADK Agents - Sistema Real de Treinamento',
        'version': '2.0.0',
        'description': 'Sistema de treinamento real de modelos personalizados',
        'features': [
            'Treinamento real de modelos (não Gemini)',
            'Database de prompt/answer',
            'Avaliação com string comparison',
            'Métricas de confiança e similaridade'
        ],
        'endpoints': {
            'health': 'GET /health',
            'create_agent': 'POST /api/v1/agents',
            'create_advanced_agent': 'POST /api/v1/agents/advanced',
            'list_agents': 'GET /api/v1/agents',
            'agent_status': 'GET /api/v1/agents/{user_id}/status',
            'inference': 'POST /api/v1/agents/{user_id}/inference',
            'evaluate': 'POST /api/v1/agents/{user_id}/evaluate',
            'delete_agent': 'DELETE /api/v1/agents/{user_id}'
        },
        'example_usage': {
            'create_agent': {
                'method': 'POST',
                'url': '/api/v1/agents',
                'body': {
                    'user_id': 'python_expert',
                    'training_data': 'I am a Python expert...',
                    'base_model': 'distilbert-base-uncased'
                }
            },
            'inference': {
                'method': 'POST',
                'url': '/api/v1/agents/{user_id}/inference',
                'body': {
                    'prompt': 'Como criar uma API REST?'
                }
            }
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'shellhacks-adk-agents',
        'version': '2.0.0',
        'training_type': 'REAL_MODELS'
    })

@app.route('/api/v1/agents', methods=['POST'])
def create_user_agent():
    """
    Create a new personalized agent for a user
    
    Request body:
    {
        "user_id": "string",
        "training_data": "string",
        "base_model": "string" (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        user_id = data.get('user_id')
        training_data = data.get('training_data')
        base_model = data.get('base_model', 'distilbert-base-uncased')
        
        if not user_id or not training_data:
            return jsonify({
                'error': 'user_id and training_data are required'
            }), 400
        
        # Create user agent pipeline
        result = user_agent_manager.create_user_agent_pipeline(
            user_id=user_id,
            training_data=training_data,
            base_model=base_model
        )
        
        return jsonify({
            'message': result,
            'user_id': user_id,
            'status': 'created',
            'timestamp': datetime.now().isoformat()
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating user agent: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Failed to create user agent: {str(e)}'
        }), 500

@app.route('/api/v1/agents/<user_id>/status', methods=['GET'])
def get_user_agent_status(user_id: str):
    """Get status of a user's agent"""
    try:
        status = user_agent_manager.get_user_status(user_id)
        
        if status is None:
            return jsonify({
                'error': f'User {user_id} not found'
            }), 404
        
        return jsonify({
            'user_id': user_id,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting user agent status: {e}")
        return jsonify({
            'error': f'Failed to get user agent status: {str(e)}'
        }), 500

@app.route('/api/v1/agents/<user_id>/inference', methods=['POST'])
def make_inference(user_id: str):
    """
    Make inference using user's personalized model
    
    Request body:
    {
        "prompt": "string"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        prompt = data.get('prompt')
        
        if not prompt:
            return jsonify({
                'error': 'prompt is required'
            }), 400
        
        # Make inference
        result = user_agent_manager.make_inference(user_id, prompt)
        
        return jsonify({
            'user_id': user_id,
            'prompt': prompt,
            'response': result,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error making inference: {e}")
        return jsonify({
            'error': f'Failed to make inference: {str(e)}'
        }), 500

@app.route('/api/v1/agents/<user_id>/pipeline', methods=['GET'])
def get_pipeline_status(user_id: str):
    """Get detailed pipeline status for a user"""
    try:
        pipeline_status = user_agent_manager.get_pipeline_status(user_id)
        
        if pipeline_status is None:
            return jsonify({
                'error': f'User {user_id} not found'
            }), 404
        
        return jsonify({
            'user_id': user_id,
            'pipeline': pipeline_status,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        return jsonify({
            'error': f'Failed to get pipeline status: {str(e)}'
        }), 500

@app.route('/api/v1/agents/<user_id>/evaluate', methods=['POST'])
def evaluate_model(user_id: str):
    """
    Evaluate model performance using string comparison
    
    Request body:
    {
        "test_prompt": "string",
        "expected_answer": "string"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        test_prompt = data.get('test_prompt')
        expected_answer = data.get('expected_answer')
        
        if not test_prompt or not expected_answer:
            return jsonify({
                'error': 'test_prompt and expected_answer are required'
            }), 400
        
        # Evaluate model performance
        evaluation_results = user_agent_manager.evaluate_model_performance(
            user_id, test_prompt, expected_answer
        )
        
        if 'error' in evaluation_results:
            return jsonify(evaluation_results), 404
        
        return jsonify({
            'user_id': user_id,
            'evaluation': evaluation_results,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return jsonify({
            'error': f'Failed to evaluate model: {str(e)}'
        }), 500

@app.route('/api/v1/agents/advanced', methods=['POST'])
def create_advanced_agent():
    """
    Create a new advanced agent with JSON dataset and string comparison
    
    Request body:
    {
        "user_id": "string",
        "json_dataset": [{"prompt": "...", "answer": "..."}, ...],
        "base_model": "string" (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        user_id = data.get('user_id')
        json_dataset = data.get('json_dataset')
        base_model = data.get('base_model', 'distilbert-base-uncased')
        
        if not user_id or not json_dataset:
            return jsonify({
                'error': 'user_id and json_dataset are required'
            }), 400
        
        if not isinstance(json_dataset, list):
            return jsonify({
                'error': 'json_dataset must be a list of objects'
            }), 400
        
        # Validate JSON structure
        for i, item in enumerate(json_dataset):
            if not isinstance(item, dict) or 'prompt' not in item or 'answer' not in item:
                return jsonify({
                    'error': f'Invalid item at index {i}: must have "prompt" and "answer" fields'
                }), 400
        
        # Create advanced agent pipeline
        result = user_agent_manager.create_advanced_agent_pipeline(
            user_id=user_id,
            json_dataset=json_dataset,
            base_model=base_model
        )
        
        return jsonify({
            'message': result,
            'user_id': user_id,
            'status': 'created',
            'training_type': 'advanced_json_qlora',
            'dataset_size': len(json_dataset),
            'string_comparison_enabled': True,
            'timestamp': datetime.now().isoformat()
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating advanced agent: {e}")
        return jsonify({
            'error': f'Failed to create advanced agent: {str(e)}'
        }), 500

@app.route('/api/v1/agents', methods=['GET'])
def list_all_agents():
    """List all active user agents"""
    try:
        all_users = user_agent_manager.get_all_users()
        
        return jsonify({
            'users': all_users,
            'total_users': len(all_users),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        return jsonify({
            'error': f'Failed to list agents: {str(e)}'
        }), 500

@app.route('/api/v1/agents/<user_id>', methods=['DELETE'])
def delete_user_agent(user_id: str):
    """Delete a user's agent and clean up resources"""
    try:
        success = user_agent_manager.remove_user(user_id)
        
        if not success:
            return jsonify({
                'error': f'User {user_id} not found'
            }), 404
        
        return jsonify({
            'message': f'User {user_id} deleted successfully',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting user agent: {e}")
        return jsonify({
            'error': f'Failed to delete user agent: {str(e)}'
        }), 500

@app.route('/api/v1/vertex/training-jobs', methods=['POST'])
def create_training_job():
    """
    Create a custom training job in Vertex AI
    
    Request body:
    {
        "user_id": "string",
        "training_data": "string",
        "base_model": "string" (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        user_id = data.get('user_id')
        training_data = data.get('training_data')
        base_model = data.get('base_model', 'distilbert-base-uncased')
        
        if not user_id or not training_data:
            return jsonify({
                'error': 'user_id and training_data are required'
            }), 400
        
        # Create training job
        client = get_vertex_client()
        if not client:
            return jsonify({
                'error': 'Vertex AI client not available'
            }), 500
        
        job_info = client.create_custom_training_job(
            user_id=user_id,
            training_data=training_data,
            base_model=base_model
        )
        
        return jsonify({
            'message': 'Training job created successfully',
            'job_info': json.loads(job_info),
            'timestamp': datetime.now().isoformat()
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating training job: {e}")
        return jsonify({
            'error': f'Failed to create training job: {str(e)}'
        }), 500

@app.route('/api/v1/vertex/training-jobs/<job_id>/status', methods=['GET'])
def get_training_job_status(job_id: str):
    """Get status of a training job"""
    try:
        client = get_vertex_client()
        if not client:
            return jsonify({
                'error': 'Vertex AI client not available'
            }), 500
        
        status = client.get_training_job_status(job_id)
        
        return jsonify({
            'job_id': job_id,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting training job status: {e}")
        return jsonify({
            'error': f'Failed to get training job status: {str(e)}'
        }), 500

@app.route('/api/v1/vertex/models/<user_id>', methods=['GET'])
def list_user_models(user_id: str):
    """List all models for a specific user"""
    try:
        client = get_vertex_client()
        if not client:
            return jsonify({
                'error': 'Vertex AI client not available'
            }), 500
        
        models = client.list_user_models(user_id)
        
        return jsonify({
            'user_id': user_id,
            'models': models,
            'total_models': len(models),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing user models: {e}")
        return jsonify({
            'error': f'Failed to list user models: {str(e)}'
        }), 500

@app.route('/api/v1/vertex/models/<user_id>/<model_id>', methods=['DELETE'])
def delete_user_model(user_id: str, model_id: str):
    """Delete a user's model"""
    try:
        client = get_vertex_client()
        if not client:
            return jsonify({
                'error': 'Vertex AI client not available'
            }), 500
        
        success = client.delete_user_model(user_id, model_id)
        
        if not success:
            return jsonify({
                'error': f'Model {model_id} not found for user {user_id}'
            }), 404
        
        return jsonify({
            'message': f'Model {model_id} deleted successfully',
            'user_id': user_id,
            'model_id': model_id,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting user model: {e}")
        return jsonify({
            'error': f'Failed to delete user model: {str(e)}'
        }), 500

@app.route('/api/v1/vertex/inference', methods=['POST'])
def vertex_inference():
    """
    Make inference using Vertex AI endpoint
    
    Request body:
    {
        "user_id": "string",
        "prompt": "string",
        "endpoint_url": "string" (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        user_id = data.get('user_id')
        prompt = data.get('prompt')
        endpoint_url = data.get('endpoint_url')
        
        if not user_id or not prompt:
            return jsonify({
                'error': 'user_id and prompt are required'
            }), 400
        
        # Make inference
        client = get_vertex_client()
        if not client:
            return jsonify({
                'error': 'Vertex AI client not available'
            }), 500
        
        result = client.make_inference(user_id, prompt, endpoint_url)
        
        return jsonify({
            'user_id': user_id,
            'prompt': prompt,
            'response': result,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error making Vertex AI inference: {e}")
        return jsonify({
            'error': f'Failed to make inference: {str(e)}'
        }), 500

@app.route('/api/v1/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    try:
        config_info = {
            'project_id': project_id,
            'location': location,
            'config_path': config_path,
            'vertex_config_path': vertex_config_path,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(config_info), 200
        
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return jsonify({
            'error': f'Failed to get config: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting ShellHacks ADK Agents service on port {port}")
    logger.info(f"Project ID: {project_id}")
    logger.info(f"Location: {location}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
