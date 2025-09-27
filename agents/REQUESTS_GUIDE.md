# Complete Guide: How to Make Requests to Trained Models

This guide explains how to make requests to receive responses from personalized models that are being trained in the agents system.

## 🚀 System Overview

The system works as follows:

1. **Agent Creation**: You create a personalized agent for a user
2. **Training**: The system processes the data and trains the model
3. **Deploy**: The model is deployed to Vertex AI
4. **Inference**: You can make requests to get responses from the trained model

## 📋 Available Endpoints

### 1. Health Check
```bash
GET /health
```
Checks if the service is running.

### 2. Create Personalized Agent
```bash
POST /api/v1/agents
```
Creates a new personalized agent for a user.

### 3. Check Agent Status
```bash
GET /api/v1/agents/{user_id}/status
```
Checks the model training status.

### 4. **Make Inference (Get Model Response)**
```bash
POST /api/v1/agents/{user_id}/inference
```
This is the main endpoint to receive responses from trained models.

### 5. List All Agents
```bash
GET /api/v1/agents
```
Lists all active agents.

## 🔧 Initial Setup

### 1. Start the Server
```bash
cd agents
python app.py
```

The server will be available at `http://localhost:8080`

### 2. Configure Environment Variables
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GEMINI_API_KEY="your-gemini-key"
export GOOGLE_APPLICATION_CREDENTIALS="credentials/service-account.json"
```

## 📝 Practical Examples

### Example 1: Complete Flow

#### Step 1: Create a Personalized Agent
```bash
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "training_data": "I am a Python programming and web development expert. I have experience with Flask, Django, FastAPI and other technologies. I can help with questions about algorithms, data structures and programming best practices.",
    "base_model": "distilbert-base-uncased"
  }'
```

**Expected response:**
```json
{
  "message": "Agent pipeline created and started for user user123",
  "user_id": "user123",
  "status": "created",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### Step 2: Check Training Status
```bash
curl http://localhost:8080/api/v1/agents/user123/status
```

**Expected response:**
```json
{
  "user_id": "user123",
  "status": "ready",
  "created_at": "2024-01-15T10:30:00.000Z",
  "pipeline_id": "abc123-def456-ghi789",
  "base_model": "distilbert-base-uncased",
  "training_data_size": 245,
  "model_ready_at": "2024-01-15T10:35:00.000Z",
  "endpoint_url": "https://us-central1-aiplatform.googleapis.com/v1/projects/your-project/locations/us-central1/endpoints/endpoint-user123-xyz789"
}
```

#### Step 3: Make Inference (Get Model Response)
```bash
curl -X POST http://localhost:8080/api/v1/agents/user123/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How to create a REST API with Flask?"
  }'
```

**Expected response:**
```json
{
  "user_id": "user123",
  "prompt": "How to create a REST API with Flask?",
  "response": "To create a REST API with Flask, you can follow these steps:\n\n1. Install Flask:\n```bash\npip install flask\n```\n\n2. Create an app.py file:\n```python\nfrom flask import Flask, jsonify, request\n\napp = Flask(__name__)\n\n@app.route('/api/hello', methods=['GET'])\ndef hello():\n    return jsonify({'message': 'Hello World!'})\n\n@app.route('/api/users', methods=['POST'])\ndef create_user():\n    data = request.get_json()\n    return jsonify({'user': data}), 201\n\nif __name__ == '__main__':\n    app.run(debug=True)\n```\n\n3. Run the application:\n```bash\npython app.py\n```\n\nThe API will be available at http://localhost:5000",
  "timestamp": "2024-01-15T10:40:00.000Z"
}
```

### Example 2: Using JavaScript/Fetch

```javascript
// Create agent
async function createAgent(userId, trainingData) {
  const response = await fetch('http://localhost:8080/api/v1/agents', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      user_id: userId,
      training_data: trainingData,
      base_model: 'distilbert-base-uncased'
    })
  });
  
  return await response.json();
}

// Make inference
async function getModelResponse(userId, prompt) {
  const response = await fetch(`http://localhost:8080/api/v1/agents/${userId}/inference`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prompt: prompt
    })
  });
  
  return await response.json();
}

// Usage example
async function example() {
  // Create agent
  const agent = await createAgent('user456', 'I am a machine learning and deep learning expert...');
  console.log('Agent created:', agent);
  
  // Wait a bit for training
  setTimeout(async () => {
    // Ask question
    const answer = await getModelResponse('user456', 'Explain what deep learning is');
    console.log('Model response:', answer.response);
  }, 10000); // 10 seconds
}
```

### Example 3: Using Python

```python
import requests
import time

class AgentClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
    
    def create_agent(self, user_id, training_data, base_model="distilbert-base-uncased"):
        """Create a personalized agent"""
        url = f"{self.base_url}/api/v1/agents"
        data = {
            "user_id": user_id,
            "training_data": training_data,
            "base_model": base_model
        }
        response = requests.post(url, json=data)
        return response.json()
    
    def get_status(self, user_id):
        """Check agent status"""
        url = f"{self.base_url}/api/v1/agents/{user_id}/status"
        response = requests.get(url)
        return response.json()
    
    def make_inference(self, user_id, prompt):
        """Make inference with the trained model"""
        url = f"{self.base_url}/api/v1/agents/{user_id}/inference"
        data = {"prompt": prompt}
        response = requests.post(url, json=data)
        return response.json()
    
    def wait_for_ready(self, user_id, max_wait=300):
        """Wait until the model is ready"""
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status = self.get_status(user_id)
            if status.get('status') == 'ready':
                return True
            elif status.get('status') == 'error':
                raise Exception(f"Training error: {status.get('error')}")
            time.sleep(5)
        return False

# Usage example
client = AgentClient()

# Create agent
agent = client.create_agent(
    user_id="python_user",
    training_data="I am a Python, data science and machine learning expert..."
)
print("Agent created:", agent)

# Wait for model to be ready
if client.wait_for_ready("python_user"):
    # Ask question
    answer = client.make_inference(
        "python_user", 
        "How to implement a neural network in Python?"
    )
    print("Answer:", answer['response'])
else:
    print("Timeout waiting for model to be ready")
```

## 🔍 Agent Status

Possible status values:

- **`initializing`**: Agent being created
- **`processing`**: Data being processed and model being trained
- **`ready`**: Model ready for inference
- **`error`**: Error during the process

## ⚠️ Important Points

### 1. Wait for Training
The model needs to be trained before it can answer questions. Check the status before making inference.

### 2. Training Data
Training data is used to personalize the model. The more specific and relevant, the better the response.

### 3. Limitations
- Each user can have only one active agent
- Training may take a few minutes
- The model uses Gemini as base for inference

### 4. Monitoring
Use the `/api/v1/agents` endpoint to list all agents and their status.

## 🚨 Troubleshooting

### Error: "Model not ready"
```bash
# Check status
curl http://localhost:8080/api/v1/agents/user123/status

# Wait and try again
```

### Error: "User not found"
```bash
# Check if agent was created
curl http://localhost:8080/api/v1/agents

# Create agent if needed
```

### Connection Error
```bash
# Check if server is running
curl http://localhost:8080/health
```

## 📊 Complete Integration Example

```python
import requests
import json
from typing import Dict, Any

class ShellHacksAgentAPI:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
    
    def create_personalized_agent(self, user_id: str, expertise: str) -> Dict[str, Any]:
        """Create personalized agent based on user expertise"""
        training_data = f"""
        I am an expert in {expertise}. 
        I have extensive experience and deep knowledge in this area.
        I can help with questions, explanations and guidance about {expertise}.
        """
        
        response = requests.post(
            f"{self.base_url}/api/v1/agents",
            json={
                "user_id": user_id,
                "training_data": training_data,
                "base_model": "distilbert-base-uncased"
            }
        )
        return response.json()
    
    def ask_question(self, user_id: str, question: str) -> str:
        """Ask question to personalized model"""
        response = requests.post(
            f"{self.base_url}/api/v1/agents/{user_id}/inference",
            json={"prompt": question}
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: {response.json().get('error', 'Unknown error')}"
    
    def get_agent_info(self, user_id: str) -> Dict[str, Any]:
        """Get agent information"""
        response = requests.get(f"{self.base_url}/api/v1/agents/{user_id}/status")
        return response.json()

# Usage example
api = ShellHacksAgentAPI()

# Create agent for Python expert
agent = api.create_personalized_agent("dev_python", "Python programming and web development")
print("Agent created:", agent)

# Wait a bit for training
import time
time.sleep(30)

# Ask questions
questions = [
    "How to implement JWT authentication in Flask?",
    "What's the difference between list comprehension and generator expression?",
    "How to optimize Django query performance?"
]

for question in questions:
    answer = api.ask_question("dev_python", question)
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")
```

## 🎯 Next Steps

1. **Implement Cache**: For faster responses
2. **Add Authentication**: For security
3. **Implement Rate Limiting**: For usage control
4. **Add Metrics**: For monitoring
5. **Implement Webhooks**: For notifications when model is ready

---

This guide provides everything you need to make requests and receive responses from models trained in the agents system!
