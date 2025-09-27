from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import subprocess
import os
import json
import sys

# Add the api directory to the path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))
from inputLLM import chat_with_gemini
from slm_inference import slm_inference, generate_slm_stream

# Import CUDA endpoints and GPU monitor
from cuda_endpoints import cuda_bp
from gpu_monitor import gpu_monitor

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Register CUDA blueprint
app.register_blueprint(cuda_bp)

# Start GPU monitoring when app starts
gpu_monitor.start_monitoring()

@app.route('/')
def index():
    return open('../../frontend/index.html').read()

@app.route('/test')
def test():
    return "Backend is running!"

def generate_stream(prompt):
    """Generator function to stream LLM response"""
    try:
        # Import here to avoid circular imports
        import requests
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
        API_KEY = os.environ.get("GEMINI_API_KEY")
        
        if not API_KEY:
            yield f"data: {json.dumps({'error': 'API key not found'})}\n\n"
            return
            
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": API_KEY
        }
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }
        
        yield f"data: {json.dumps({'status': 'Sending request to Gemini API...'})}\n\n"
        
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent",
            headers=headers, 
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            yield f"data: {json.dumps({'status': 'Response received!'})}\n\n"
            
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    full_text = ""
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            text = part["text"]
                            # Stream word by word
                            words = text.split()
                            for i, word in enumerate(words):
                                full_text += word + " "
                                yield f"data: {json.dumps({'text': word + ' ', 'full_text': full_text.strip(), 'is_complete': i == len(words) - 1})}\n\n"
                    
                    # Write to both files
                    try:
                        # Create JSON data structure
                        json_data = {
                            "prompt": prompt,
                            "answer": full_text.strip()
                        }
                        
                        # Write to LLMCurrData.json (replace with most recent)
                        with open("db/LLMCurrData.json", "w", encoding="utf-8") as f:
                            json.dump(json_data, f, indent=2, ensure_ascii=False)
                        
                        # Append to LLMData.json (accumulate all messages)
                        # First, read existing data
                        existing_data = []
                        try:
                            with open("db/LLMData.json", "r", encoding="utf-8") as f:
                                existing_data = json.load(f)
                        except (FileNotFoundError, json.JSONDecodeError):
                            existing_data = []
                        
                        # Ensure existing_data is a list
                        if not isinstance(existing_data, list):
                            existing_data = []
                        
                        # Append new data
                        existing_data.append(json_data)
                        
                        # Write back to file
                        with open("db/LLMData.json", "w", encoding="utf-8") as f:
                            json.dump(existing_data, f, indent=2, ensure_ascii=False)
                        
                        yield f"data: {json.dumps({'status': 'Response saved to LLMCurrData.json and LLMData.json!'})}\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'status': f'Error saving to file: {str(e)}'})}\n\n"
                    
                    yield f"data: {json.dumps({'status': 'Response complete!'})}\n\n"
                else:
                    yield f"data: {json.dumps({'error': 'No text content found'})}\n\n"
            else:
                yield f"data: {json.dumps({'error': 'No candidates found'})}\n\n"
        else:
            yield f"data: {json.dumps({'error': f'HTTP {response.status_code}: {response.text}'})}\n\n"
            
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.route('/run_llm', methods=['POST'])
def run_llm():
    try:
        # Get prompt from request body
        data = request.get_json()
        prompt = data.get('prompt', 'write a 300 word story about a cat')
        
        return Response(generate_stream(prompt), 
                      mimetype='text/plain',
                      headers={'Cache-Control': 'no-cache'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ===== SLM (Small Language Model) Endpoints =====

@app.route('/slm/info', methods=['GET'])
def slm_info():
    """Get information about the SLM model"""
    try:
        info = slm_inference.get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/slm/load', methods=['POST'])
def slm_load():
    """Load the SLM model"""
    try:
        result = slm_inference.load_model()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/slm/unload', methods=['POST'])
def slm_unload():
    """Unload the SLM model to free memory"""
    try:
        result = slm_inference.unload_model()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/slm/generate', methods=['POST'])
def slm_generate():
    """Generate a response using the SLM model (non-streaming)"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', 'Hello, how are you?')
        max_new_tokens = data.get('max_new_tokens', 100)
        temperature = data.get('temperature', 0.7)
        do_sample = data.get('do_sample', True)
        
        result = slm_inference.generate_response(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/slm/stream', methods=['POST'])
def slm_stream():
    """Generate a streaming response using the SLM model"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', 'Hello, how are you?')
        max_new_tokens = data.get('max_new_tokens', 100)
        temperature = data.get('temperature', 0.7)
        
        return Response(generate_slm_stream(prompt, max_new_tokens, temperature), 
                      mimetype='text/plain',
                      headers={'Cache-Control': 'no-cache'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ===== Test Results Endpoints =====

@app.route('/test_results', methods=['POST'])
def receive_test_results():
    """Receive test results from evaluation scripts"""
    try:
        results = request.get_json()
        
        # Save results to database
        test_id = results.get('test_id', 'unknown')
        model_type = results.get('model_type', 'unknown')
        
        # Save to file
        results_filename = f"test_results_{model_type.lower()}_{test_id}.json"
        results_path = os.path.join("db", results_filename)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Also save as latest results for easy access
        latest_results_path = os.path.join("db", f"latest_test_results_{model_type.lower()}.json")
        with open(latest_results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            'success': True,
            'message': 'Test results received and saved',
            'test_id': test_id,
            'model_type': model_type,
            'results_file': results_filename
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test_results', methods=['GET'])
def get_test_results():
    """Get latest test results"""
    try:
        model_type = request.args.get('model_type', 'cuda').lower()
        
        # Try to get latest results
        latest_results_path = os.path.join("db", f"latest_test_results_{model_type}.json")
        
        if os.path.exists(latest_results_path):
            with open(latest_results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            return jsonify(results)
        else:
            return jsonify({
                'success': False,
                'error': f'No test results found for model type: {model_type}',
                'available_files': [f for f in os.listdir("db") if f.startswith("test_results_")]
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test_results/list', methods=['GET'])
def list_test_results():
    """List all available test result files"""
    try:
        db_path = "db"
        if not os.path.exists(db_path):
            return jsonify({'success': False, 'error': 'Database directory not found'})
        
        # Get all test result files
        test_files = []
        for filename in os.listdir(db_path):
            if filename.startswith("test_results_") and filename.endswith(".json"):
                file_path = os.path.join(db_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    file_info = {
                        'filename': filename,
                        'test_id': data.get('test_id', 'unknown'),
                        'model_type': data.get('model_type', 'unknown'),
                        'timestamp': data.get('timestamp', 'unknown'),
                        'total_examples': data.get('summary_metrics', {}).get('total_examples', 0),
                        'success_rate': data.get('summary_metrics', {}).get('successful_generations', 0)
                    }
                    test_files.append(file_info)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        
        # Sort by timestamp (newest first)
        test_files.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'success': True,
            'test_files': test_files,
            'total_files': len(test_files)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test_results/<test_id>', methods=['GET'])
def get_specific_test_results(test_id):
    """Get specific test results by test ID"""
    try:
        db_path = "db"
        
        # Find file with matching test_id
        for filename in os.listdir(db_path):
            if filename.startswith("test_results_") and test_id in filename:
                file_path = os.path.join(db_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                return jsonify(results)
        
        return jsonify({
            'success': False,
            'error': f'Test results not found for test_id: {test_id}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test_results/summary', methods=['GET'])
def get_test_summary():
    """Get a summary of all test results"""
    try:
        db_path = "db"
        summary = {
            'cuda_results': None,
            'cpu_results': None,
            'comparison': {}
        }
        
        # Get latest CUDA results
        cuda_path = os.path.join(db_path, "latest_test_results_cuda.json")
        if os.path.exists(cuda_path):
            with open(cuda_path, 'r', encoding='utf-8') as f:
                cuda_data = json.load(f)
                summary['cuda_results'] = cuda_data.get('summary_metrics', {})
                summary['cuda_results']['timestamp'] = cuda_data.get('timestamp')
        
        # Get latest CPU results
        cpu_path = os.path.join(db_path, "latest_test_results_cpu.json")
        if os.path.exists(cpu_path):
            with open(cpu_path, 'r', encoding='utf-8') as f:
                cpu_data = json.load(f)
                summary['cpu_results'] = cpu_data.get('summary_metrics', {})
                summary['cpu_results']['timestamp'] = cpu_data.get('timestamp')
        
        # Create comparison if both exist
        if summary['cuda_results'] and summary['cpu_results']:
            cuda_metrics = summary['cuda_results']
            cpu_metrics = summary['cpu_results']
            
            summary['comparison'] = {
                'speed_comparison': {
                    'cuda_avg_time': cuda_metrics.get('average_generation_time', 0),
                    'cpu_avg_time': cpu_metrics.get('average_generation_time', 0),
                    'speedup_factor': cpu_metrics.get('average_generation_time', 1) / cuda_metrics.get('average_generation_time', 1) if cuda_metrics.get('average_generation_time', 0) > 0 else 0
                },
                'accuracy_comparison': {
                    'cuda_success_rate': cuda_metrics.get('successful_generations', 0) / cuda_metrics.get('total_examples', 1) * 100 if cuda_metrics.get('total_examples', 0) > 0 else 0,
                    'cpu_success_rate': cpu_metrics.get('successful_generations', 0) / cpu_metrics.get('total_examples', 1) * 100 if cpu_metrics.get('total_examples', 0) > 0 else 0
                }
            }
        
        return jsonify({
            'success': True,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ===== GPU Monitoring Endpoints =====

@app.route('/gpu/current', methods=['GET'])
def get_current_gpu_data():
    """Get current GPU usage data"""
    try:
        current_data = gpu_monitor.get_current_data()
        return jsonify({
            'success': True,
            'data': current_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/gpu/graph', methods=['GET'])
def get_gpu_graph_data():
    """Get GPU data formatted for dashboard graph"""
    try:
        graph_data = gpu_monitor.get_graph_data()
        return jsonify({
            'success': True,
            'data': graph_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/gpu/history', methods=['GET'])
def get_gpu_history():
    """Get historical GPU data"""
    try:
        history = gpu_monitor.get_historical_data()
        return jsonify({
            'success': True,
            'data': history
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/gpu/start', methods=['POST'])
def start_gpu_monitoring():
    """Start GPU monitoring"""
    try:
        gpu_monitor.start_monitoring()
        return jsonify({
            'success': True,
            'message': 'GPU monitoring started'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/gpu/stop', methods=['POST'])
def stop_gpu_monitoring():
    """Stop GPU monitoring"""
    try:
        gpu_monitor.stop_monitoring()
        return jsonify({
            'success': True,
            'message': 'GPU monitoring stopped'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
