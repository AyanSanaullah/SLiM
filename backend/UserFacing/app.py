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

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
