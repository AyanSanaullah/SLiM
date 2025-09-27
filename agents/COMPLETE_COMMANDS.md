# 📋 Complete Commands Guide - ShellHacks Agents

This document contains all available commands to use the agents system with real model training.

## 🚀 Initialization Commands

### 1. Start Main Server
```bash
cd agents
python3 app.py
```
**What it does:** Starts the Flask server on port 8080 with all API endpoints

### 2. Start String Comparison Service
```bash
cd string-comparison
python3 backend.py
```
**What it does:** Starts the semantic comparison service on port 8000

## 📡 API Commands (cURL)

### Basic Commands

#### Health Check
```bash
curl http://localhost:8080/health
```
**Expected response:**
```json
{
  "status": "healthy",
  "service": "shellhacks-adk-agents",
  "version": "2.0.0",
  "training_type": "REAL_MODELS"
}
```

#### Homepage (API Documentation)
```bash
curl http://localhost:8080/
```

#### List All Agents
```bash
curl http://localhost:8080/api/v1/agents
```

### Agent Creation Commands

#### Create Basic Agent
```bash
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "python_expert",
    "training_data": "I am a Python expert with experience in Flask, Django and web development. I can help with programming questions and best practices.",
    "base_model": "distilbert-base-uncased"
  }'
```

#### Create Advanced Agent (JSON Dataset + QLoRA + String Comparison)
```bash
curl -X POST http://localhost:8080/api/v1/agents/advanced \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "advanced_expert",
    "json_dataset": [
      {"prompt": "How to create a REST API?", "answer": "Use Flask with @app.route()"},
      {"prompt": "What is machine learning?", "answer": "ML is AI that learns from data"},
      {"prompt": "How to use Docker?", "answer": "Docker creates portable containers"}
    ],
    "base_model": "distilbert-base-uncased"
  }'
```

### Monitoring Commands

#### Check Agent Status
```bash
curl http://localhost:8080/api/v1/agents/python_expert/status
```

#### Check Detailed Pipeline
```bash
curl http://localhost:8080/api/v1/agents/python_expert/pipeline
```

### Inference Commands

#### Ask Question to Trained Model
```bash
curl -X POST http://localhost:8080/api/v1/agents/python_expert/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How to create a REST API with Flask?"
  }'
```

#### Evaluate Model with String Comparison
```bash
curl -X POST http://localhost:8080/api/v1/agents/python_expert/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "test_prompt": "What is Flask?",
    "expected_answer": "Flask is a Python web framework"
  }'
```

### Management Commands

#### Delete Agent
```bash
curl -X DELETE http://localhost:8080/api/v1/agents/python_expert
```

#### Get Current Configuration
```bash
curl http://localhost:8080/api/v1/config
```

## 🐍 Available Python Scripts

### Automated Test Scripts

#### Basic System Test
```bash
python3 test_agent_api.py
```
**What it does:**
- Creates 3 specialized agents
- Tests inference with each one
- Interactive mode available

#### Advanced System Test
```bash
python3 test_advanced_system.py
```
**What it does:**
- Creates advanced agent with JSON dataset
- Tests QLoRA + CUDA + String Comparison
- Shows detailed metrics

#### List Existing Agents
```bash
python3 list_agents.py
```
**What it does:**
- Lists all active agents
- Shows status and information

### Utility Scripts

#### Initial Setup Script
```bash
./setup.sh
```
**What it does:**
- Installs dependencies
- Configures virtual environment
- Prepares directories

#### Deploy Script
```bash
./deploy.sh cloud-run
```
**What it does:**
- Deploy to Google Cloud Run
- Configure Vertex AI
- Setup monitoring

## 🔧 Development Commands

### Dependency Installation
```bash
pip3 install -r requirements.txt
```

### Specific Dependencies
```bash
# Machine Learning
pip3 install torch transformers peft accelerate

# String Comparison Service
pip3 install fastapi sentence-transformers spacy nltk

# Google Cloud
pip3 install google-cloud-aiplatform google-adk
```

### Debug Commands

#### Check Real-time Logs
```bash
tail -f logs/user_logs/*/training.log
```

#### Check Trained Models
```bash
ls -la models/user_models/
```

#### Check Processed Data
```bash
ls -la data/user_data/
```

## 🎯 Complete Usage Examples

### Example 1: Python Expert Agent

```bash
# 1. Create agent
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "python_guru",
    "training_data": "I am a Python expert with 10 years of experience. I master Flask, Django, FastAPI, pandas, numpy, scikit-learn. I can help with web development, data analysis, machine learning and automation.",
    "base_model": "distilbert-base-uncased"
  }'

# 2. Wait for training (check status)
curl http://localhost:8080/api/v1/agents/python_guru/status

# 3. Ask questions
curl -X POST http://localhost:8080/api/v1/agents/python_guru/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How to optimize Flask application performance?"}'

# 4. Evaluate response
curl -X POST http://localhost:8080/api/v1/agents/python_guru/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "test_prompt": "How to optimize Flask?",
    "expected_answer": "Use cache, optimize queries, configure gunicorn"
  }'
```

### Example 2: Advanced Agent with JSON Dataset

```bash
# 1. Create advanced agent
curl -X POST http://localhost:8080/api/v1/agents/advanced \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "devops_expert",
    "json_dataset": [
      {
        "prompt": "How to setup CI/CD?",
        "answer": "Configure GitHub Actions or GitLab CI with build, test and deploy stages. Use Docker containers and separate environments."
      },
      {
        "prompt": "What is Kubernetes?",
        "answer": "Kubernetes is a container orchestrator that automates deployment, scaling and management of containerized applications."
      },
      {
        "prompt": "How to monitor applications?",
        "answer": "Use Prometheus for metrics, Grafana for dashboards, ELK Stack for logs and alerts via PagerDuty or Slack."
      }
    ]
  }'

# 2. Follow training with detailed logs
curl http://localhost:8080/api/v1/agents/devops_expert/status

# 3. Test inference
curl -X POST http://localhost:8080/api/v1/agents/devops_expert/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How to do secure deployment?"}'
```

## 🔍 Monitoring and Debug Commands

### Check Service Status
```bash
# Check if agents is running
curl http://localhost:8080/health

# Check if string-comparison is running
curl http://localhost:8000/health

# List active Python processes
ps aux | grep python
```

### Logs and Debugging
```bash
# View server logs
tail -f logs/app.log

# View specific training logs
tail -f logs/user_logs/python_expert/training.log

# View performance logs
cat logs/user_logs/python_expert/latest_performance.json
```

### Cleanup and Reset
```bash
# Stop all services
killall Python

# Clean training data
rm -rf data/user_data/*
rm -rf models/user_models/*
rm -rf logs/user_logs/*

# Restart server
python3 app.py
```

## 🌐 Deploy and Production Commands

### Local Deploy with Docker
```bash
# Build image
docker build -t shellhacks-agents .

# Run container
docker run -p 8080:8080 shellhacks-agents
```

### Deploy to Google Cloud
```bash
# Configure project
export GOOGLE_CLOUD_PROJECT="arctic-keyword-473423-g6"
gcloud config set project $GOOGLE_CLOUD_PROJECT

# Deploy to Cloud Run
./deploy.sh cloud-run

# Check deploy
gcloud run services list
```

### Vertex AI Commands
```bash
# List training jobs
gcloud ai custom-jobs list

# List endpoints
gcloud ai endpoints list

# Check models
gcloud ai models list
```

## 📊 Metrics and Analysis Commands

### Performance Analysis
```bash
# View statistics for all agents
python3 -c "
import requests
r = requests.get('http://localhost:8080/api/v1/agents')
for user_id, info in r.json()['users'].items():
    print(f'{user_id}: {info.get(\"status\", \"unknown\")}')
"
```

### String Comparison Testing
```bash
# Direct service test
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "sentence1": "How to create a REST API?",
    "sentence2": "Use Flask with @app.route decorators"
  }'
```

## 🆘 Troubleshooting Commands

### Common Problems

#### Server won't start
```bash
# Check if port is occupied
lsof -i :8080

# Kill process on port
kill -9 $(lsof -ti :8080)

# Reinstall dependencies
pip3 install -r requirements.txt --force-reinstall
```

#### Google Cloud permissions error
```bash
# Check authentication
gcloud auth list

# Renew credentials
gcloud auth login

# Check service account
gcloud iam service-accounts list
```

#### String comparison not working
```bash
# Check if it's running
curl http://localhost:8000/health

# Start if needed
cd string-comparison
python3 backend.py
```

## 📚 Documentation Commands

### Generate API Documentation
```bash
# View interactive documentation
curl http://localhost:8080/ | jq .

# Export API schema
curl http://localhost:8080/openapi.json > api_schema.json
```

### View Performance Logs
```bash
# Formatted logs
python3 -c "
import json
with open('logs/user_logs/python_expert/latest_performance.json') as f:
    print(json.dumps(json.load(f), indent=2))
"
```

---

## 🎉 Main Commands Summary

| Action | Command |
|--------|---------|
| **Start server** | `python3 app.py` |
| **Create basic agent** | `curl -X POST localhost:8080/api/v1/agents -d '{...}'` |
| **Create advanced agent** | `curl -X POST localhost:8080/api/v1/agents/advanced -d '{...}'` |
| **Check status** | `curl localhost:8080/api/v1/agents/USER_ID/status` |
| **Ask question** | `curl -X POST localhost:8080/api/v1/agents/USER_ID/inference -d '{...}'` |
| **Evaluate model** | `curl -X POST localhost:8080/api/v1/agents/USER_ID/evaluate -d '{...}'` |
| **List agents** | `curl localhost:8080/api/v1/agents` |
| **Automated test** | `python3 test_advanced_system.py` |
| **String comparison** | `curl -X POST localhost:8000/compare -d '{...}'` |

This system offers real model training with QLoRA, CUDA, string comparison evaluation and detailed performance logging! 🚀
