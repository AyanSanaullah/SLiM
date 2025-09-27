# SLM API Documentation

This API allows you to serve your trained Small Language Model (SLM) via REST endpoints. After training your model using the LoRA fine-tuning scripts, you can deploy it as a web service.

## Quick Start

### 1. Train Your Model First
Before using the API, you need to train a model:

```bash
# For CUDA (GPU)
cd ../SLMInit
python cudaInit.py

# For CPU
cd ../SLMInit
python cudaInit_cpu.py
```

### 2. Start the API Server
```bash
# Simple start
python start_slm_api.py

# Custom port and host
python start_slm_api.py --port 8080 --host 0.0.0.0

# With debug mode
python start_slm_api.py --debug
```

### 3. Test the API
```bash
# Run test suite
python test_slm_api.py

# Test with custom prompt
python test_slm_api.py --prompt "Tell me about artificial intelligence"
```

## API Endpoints

### Model Information
**GET** `/slm/info`

Get information about the model status and capabilities.

**Response:**
```json
{
  "model_loaded": false,
  "device": "cuda",
  "base_model": "gpt2",
  "lora_path": "../SLMInit/cuda_lora_out",
  "cuda_available": true,
  "mps_available": false
}
```

### Load Model
**POST** `/slm/load`

Load the trained LoRA model into memory.

**Response:**
```json
{
  "success": true,
  "message": "Model loaded successfully on cuda",
  "device": "cuda",
  "model_path": "../SLMInit/cuda_lora_out"
}
```

### Generate Response (Non-Streaming)
**POST** `/slm/generate`

Generate a complete response at once.

**Request Body:**
```json
{
  "prompt": "Explain what machine learning is",
  "max_new_tokens": 100,
  "temperature": 0.7,
  "do_sample": true
}
```

**Response:**
```json
{
  "success": true,
  "prompt": "Explain what machine learning is",
  "response": "Machine learning is a subset of artificial intelligence...",
  "full_response": "### Instruction:\nExplain what machine learning is\n\n### Response:\nMachine learning is a subset of artificial intelligence...",
  "device": "cuda",
  "settings": {
    "max_new_tokens": 100,
    "temperature": 0.7,
    "do_sample": true
  }
}
```

### Generate Response (Streaming)
**POST** `/slm/stream`

Generate a response with real-time streaming.

**Request Body:**
```json
{
  "prompt": "Write a short story about a robot",
  "max_new_tokens": 150,
  "temperature": 0.8
}
```

**Response:** Server-Sent Events (SSE) stream
```
data: {"status": "Loading SLM model..."}

data: {"status": "Model loaded on cuda"}

data: {"status": "Generating response..."}

data: {"text": "Once ", "full_text": "Once", "is_complete": false}

data: {"text": "upon ", "full_text": "Once upon", "is_complete": false}

data: {"status": "Response complete!", "device_used": "cuda"}
```

### Unload Model
**POST** `/slm/unload`

Unload the model from memory to free resources.

**Response:**
```json
{
  "success": true,
  "message": "Model unloaded successfully"
}
```

## Usage Examples

### Python Client
```python
import requests
import json

# Base URL
base_url = "http://localhost:5000"

# Load model
response = requests.post(f"{base_url}/slm/load")
print(response.json())

# Generate response
payload = {
    "prompt": "What is the capital of France?",
    "max_new_tokens": 50,
    "temperature": 0.7
}
response = requests.post(f"{base_url}/slm/generate", json=payload)
result = response.json()
print(result["response"])
```

### JavaScript/Node.js Client
```javascript
const axios = require('axios');

const baseUrl = 'http://localhost:5000';

// Load model
async function loadModel() {
    const response = await axios.post(`${baseUrl}/slm/load`);
    console.log(response.data);
}

// Generate response
async function generateResponse(prompt) {
    const response = await axios.post(`${baseUrl}/slm/generate`, {
        prompt: prompt,
        max_new_tokens: 100,
        temperature: 0.7
    });
    return response.data.response;
}

// Usage
loadModel().then(() => {
    generateResponse("Tell me a joke").then(response => {
        console.log(response);
    });
});
```

### cURL Examples
```bash
# Load model
curl -X POST http://localhost:5000/slm/load

# Generate response
curl -X POST http://localhost:5000/slm/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Python?", "max_new_tokens": 80}'

# Get model info
curl http://localhost:5000/slm/info
```

## Configuration

### Model Parameters
- **max_new_tokens**: Maximum number of tokens to generate (default: 100)
- **temperature**: Controls randomness (0.0 = deterministic, 1.0 = very random)
- **do_sample**: Whether to use sampling (true) or greedy decoding (false)

### Performance Tips
- **CUDA**: Use for fastest inference if you have a compatible GPU
- **CPU**: Slower but works on any machine
- **MPS**: Good performance on Apple Silicon Macs

### Memory Management
- The model stays loaded in memory until explicitly unloaded
- Use `/slm/unload` to free memory when not in use
- Loading takes time but inference is fast once loaded

## Troubleshooting

### Common Issues

**"LoRA model not found"**
- Train your model first using `cudaInit.py` or `cudaInit_cpu.py`
- Check that the output directory exists

**"CUDA out of memory"**
- Reduce `max_new_tokens`
- Use CPU version instead
- Unload other models/applications using GPU memory

**"Module not found"**
- Install dependencies: `pip install -r ../SLMInit/requirements.txt`
- Ensure you're in the correct directory

**API server won't start**
- Check if port is already in use
- Verify Python dependencies are installed
- Check for any error messages in the console

### Debug Mode
Run with debug mode for detailed error messages:
```bash
python start_slm_api.py --debug
```

## Integration Examples

### Web Application
```html
<!DOCTYPE html>
<html>
<head>
    <title>SLM Chat</title>
</head>
<body>
    <div id="chat"></div>
    <input type="text" id="prompt" placeholder="Enter your message...">
    <button onclick="sendMessage()">Send</button>

    <script>
    async function sendMessage() {
        const prompt = document.getElementById('prompt').value;
        const response = await fetch('http://localhost:5000/slm/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({prompt: prompt, max_new_tokens: 100})
        });
        const data = await response.json();
        document.getElementById('chat').innerHTML += 
            `<p><strong>You:</strong> ${prompt}</p>
             <p><strong>AI:</strong> ${data.response}</p>`;
        document.getElementById('prompt').value = '';
    }
    </script>
</body>
</html>
```

### Discord Bot
```python
import discord
import requests

class SLMBot(discord.Client):
    def __init__(self):
        super().__init__()
        self.slm_url = "http://localhost:5000"
        
    async def on_message(self, message):
        if message.author == self.user:
            return
            
        if message.content.startswith('!ask '):
            prompt = message.content[5:]  # Remove '!ask '
            
            response = requests.post(f"{self.slm_url}/slm/generate", 
                json={"prompt": prompt, "max_new_tokens": 150})
            
            if response.status_code == 200:
                data = response.json()
                await message.channel.send(data["response"])
            else:
                await message.channel.send("Sorry, I couldn't generate a response.")

# Run bot
bot = SLMBot()
bot.run('YOUR_BOT_TOKEN')
```

## Advanced Usage

### Custom Model Loading
You can modify `slm_inference.py` to use different base models:
```python
# In slm_inference.py, change:
self.base_model_name = "distilgpt2"  # Smaller, faster
# or
self.base_model_name = "microsoft/DialoGPT-medium"  # Conversational
```

### Batch Processing
```python
prompts = [
    "What is AI?",
    "Explain quantum computing",
    "Write a haiku about programming"
]

for prompt in prompts:
    response = requests.post("http://localhost:5000/slm/generate", 
        json={"prompt": prompt})
    print(f"Q: {prompt}")
    print(f"A: {response.json()['response']}\n")
```

### Production Deployment
For production use, consider:
- Using a production WSGI server like Gunicorn
- Adding authentication/rate limiting
- Implementing proper logging
- Using a reverse proxy like Nginx
- Adding health checks and monitoring

```bash
# Example production deployment
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify your model was trained successfully
3. Ensure all dependencies are installed
4. Test with the provided test script
5. Check the troubleshooting section above

Happy coding! 🚀
