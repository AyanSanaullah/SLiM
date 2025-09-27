from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import subprocess
import os
import json
import sys

# Add the api directory to the path so we can import inputLLM
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))
from inputLLM import chat_with_gemini

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return open('../frontend/index.html').read()

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
                        # Write to LLMCurrData.txt (replace with most recent)
                        with open("db/LLMCurrData.txt", "w", encoding="utf-8") as f:
                            f.write(prompt + "\n")
                            f.write(full_text.strip() + "\n")
                        
                        # Append to LLMData.txt (accumulate all messages)
                        with open("db/LLMData.txt", "a", encoding="utf-8") as f:
                            f.write("---\n")
                            f.write(prompt + "\n")
                            f.write(full_text.strip() + "\n")
                            f.write("\n")
                        
                        yield f"data: {json.dumps({'status': 'Response saved to LLMCurrData.txt and LLMData.txt!'})}\n\n"
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
