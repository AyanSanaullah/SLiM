# How to Use the Agents System - Practical Guide

## 🎯 Quick Summary

To make requests and receive responses from trained models, you need to follow these steps:

1. **Start the server**: `python app.py`
2. **Create an agent**: POST `/api/v1/agents`
3. **Wait for training**: GET `/api/v1/agents/{user_id}/status`
4. **Ask questions**: POST `/api/v1/agents/{user_id}/inference`

## 🚀 Quick Start

### 1. Setup and Start

```bash
# Navigate to the directory
cd agents

# Install dependencies (if needed)
pip install -r requirements.txt

# Configure environment variables
export GOOGLE_CLOUD_PROJECT="arctic-keyword-473423-g6"

# Start the server
python app.py
```

The server will be available at `http://localhost:8080`

### 2. Test with Example Script

```bash
# Run the test script
python test_agent_api.py
```

This script will automatically create 3 specialized agents and ask questions to each one.

## 📋 Usage Examples

### Example 1: Python Expert Agent

```bash
# 1. Create agent
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "python_expert",
    "training_data": "I am a Python expert specializing in Flask, Django, data science and machine learning. I can help with programming questions, algorithms and best practices.",
    "base_model": "distilbert-base-uncased"
  }'

# 2. Check status
curl http://localhost:8080/api/v1/agents/python_expert/status

# 3. Ask question
curl -X POST http://localhost:8080/api/v1/agents/python_expert/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How to create a REST API with Flask?"
  }'
```

### Example 2: Machine Learning Expert Agent

```bash
# 1. Create agent
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "ml_expert",
    "training_data": "I am a machine learning expert specializing in deep learning, NLP, computer vision and MLOps. I can help with algorithm implementation and data analysis.",
    "base_model": "distilbert-base-uncased"
  }'

# 2. Wait and ask question
curl -X POST http://localhost:8080/api/v1/agents/ml_expert/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How to choose between Random Forest and XGBoost?"
  }'
```

## 🔧 Programmatic Usage

### Python

```python
import requests

# Create agent
def create_agent(user_id, expertise):
    training_data = f"I am an expert in {expertise}. I can help with questions and explanations about this subject."
    
    response = requests.post(
        "http://localhost:8080/api/v1/agents",
        json={
            "user_id": user_id,
            "training_data": training_data
        }
    )
    return response.json()

# Ask question
def ask_question(user_id, question):
    response = requests.post(
        f"http://localhost:8080/api/v1/agents/{user_id}/inference",
        json={"prompt": question}
    )
    return response.json()['response']

# Usage example
agent = create_agent("dev_expert", "web development and programming")
answer = ask_question("dev_expert", "How to implement JWT authentication?")
print(answer)
```

### JavaScript/Node.js

```javascript
// Create agent
async function createAgent(userId, expertise) {
  const trainingData = `I am an expert in ${expertise}. I can help with questions and explanations about this subject.`;
  
  const response = await fetch('http://localhost:8080/api/v1/agents', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      training_data: trainingData
    })
  });
  
  return await response.json();
}

// Ask question
async function askQuestion(userId, question) {
  const response = await fetch(`http://localhost:8080/api/v1/agents/${userId}/inference`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: question })
  });
  
  const result = await response.json();
  return result.response;
}

// Usage example
async function example() {
  const agent = await createAgent('js_expert', 'JavaScript and Node.js');
  const answer = await askQuestion('js_expert', 'How to implement async/await?');
  console.log(answer);
}
```

## 📊 Agent Status

Possible status values:

- **`initializing`**: Agent being created
- **`processing`**: Data being processed and model being trained  
- **`ready`**: Model ready for inference
- **`error`**: Error during the process

## ⚠️ Important Points

1. **Wait for Training**: The model needs to be trained before answering questions
2. **Training Data**: Be specific in the expertise description for better results
3. **One Agent per User**: Each `user_id` can have only one active agent
4. **Timeout**: Training may take a few minutes

## 🔍 Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check if the service is running |
| POST | `/api/v1/agents` | Create a new agent |
| GET | `/api/v1/agents/{user_id}/status` | Check agent status |
| POST | `/api/v1/agents/{user_id}/inference` | **Ask question to the model** |
| GET | `/api/v1/agents` | List all agents |
| DELETE | `/api/v1/agents/{user_id}` | Delete an agent |

## 🚨 Troubleshooting

### Problem: "Connection refused"
**Solution**: Check if the server is running
```bash
curl http://localhost:8080/health
```

### Problem: "Model not ready"
**Solution**: Wait for training to complete
```bash
curl http://localhost:8080/api/v1/agents/user123/status
```

### Problem: "User not found"
**Solution**: Create the agent first
```bash
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "training_data": "..."}'
```

## 🎮 Interactive Mode

Use the test script for interactive mode:

```bash
python test_agent_api.py
```

Available commands:
- `status user_id`: Check status
- `ask user_id question`: Ask question
- `list`: List agents
- `delete user_id`: Delete agent

## 📚 Complete Examples

See the files:
- `REQUESTS_GUIDE.md`: Complete detailed guide
- `test_agent_api.py`: Test script with examples
- `README.md`: General project documentation

## 🎯 Next Steps

1. Experiment with different types of expertise
2. Test with specific questions from your domain
3. Integrate with your application
4. Monitor performance and adjust as needed

---

**Summary**: To receive responses from models, create an agent with `POST /api/v1/agents`, wait for `ready` status, and then ask questions with `POST /api/v1/agents/{user_id}/inference`.
