# ⚡ Quick Commands - ShellHacks Agents

## 🚀 Quick Start

```bash
# 1. Start services
cd agents && python3 app.py &
cd string-comparison && python3 backend.py &

# 2. Quick test
python3 test_advanced_system.py
```

## 📋 Essential Commands

### Create Basic Agent
```bash
curl -X POST localhost:8080/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{"user_id": "expert", "training_data": "I am an expert in..."}'
```

### Create Advanced Agent (JSON + String Comparison)
```bash
curl -X POST localhost:8080/api/v1/agents/advanced \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "advanced_expert",
    "json_dataset": [
      {"prompt": "How to do X?", "answer": "To do X, you..."},
      {"prompt": "What is Y?", "answer": "Y is a concept that..."}
    ]
  }'
```

### Check Status
```bash
curl localhost:8080/api/v1/agents/expert/status
```

### Ask Question
```bash
curl -X POST localhost:8080/api/v1/agents/expert/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How to solve this problem?"}'
```

### List Agents
```bash
curl localhost:8080/api/v1/agents
```

## 🔍 Health Checks

```bash
# Main service
curl localhost:8080/health

# String comparison
curl localhost:8000/health

# List agents
curl localhost:8080/api/v1/agents | jq '.users | keys'
```

## 🧪 Automated Tests

```bash
# Complete system test
python3 test_advanced_system.py

# Basic test
python3 test_agent_api.py

# List existing agents
python3 list_agents.py
```

## 🛠️ Debug and Cleanup

```bash
# Stop services
killall Python

# View processes
ps aux | grep python

# Clean data
rm -rf data/user_data/* models/user_models/* logs/user_logs/*
```

## 📊 JSON Dataset Examples

### Python Expert
```json
{
  "user_id": "python_expert",
  "json_dataset": [
    {"prompt": "How to create REST API?", "answer": "Use Flask with @app.route()"},
    {"prompt": "What is Django?", "answer": "Django is a Python web framework"},
    {"prompt": "How to use pandas?", "answer": "Pandas is for data analysis"}
  ]
}
```

### DevOps Expert
```json
{
  "user_id": "devops_expert", 
  "json_dataset": [
    {"prompt": "How to setup CI/CD?", "answer": "Use GitHub Actions or GitLab CI"},
    {"prompt": "What is Docker?", "answer": "Docker creates portable containers"},
    {"prompt": "How to use Kubernetes?", "answer": "K8s orchestrates containers"}
  ]
}
```

## 🎯 Commands by Functionality

### Training
| Action | Command |
|--------|---------|
| Create basic agent | `POST /api/v1/agents` |
| Create advanced agent | `POST /api/v1/agents/advanced` |
| Check status | `GET /api/v1/agents/{id}/status` |

### Inference
| Action | Command |
|--------|---------|
| Ask question | `POST /api/v1/agents/{id}/inference` |
| Evaluate response | `POST /api/v1/agents/{id}/evaluate` |
| View pipeline | `GET /api/v1/agents/{id}/pipeline` |

### Management
| Action | Command |
|--------|---------|
| List all | `GET /api/v1/agents` |
| Delete agent | `DELETE /api/v1/agents/{id}` |
| View config | `GET /api/v1/config` |

## 🔥 Performance Logging

The system automatically logs to terminal:
- ✅ Percentage of each string comparison test
- 📊 Detailed similarity metrics
- 🎯 Overall performance score
- 🟢🟡🔴 Quality classification

Example output:
```
🎯🎯🎯🎯 AGENT PERFORMANCE REPORT - ADVANCED_EXPERT 🎯🎯🎯🎯
📅 Completed at: 2024-01-15 14:30:22
🤖 Agent Type: advanced_qlora

🔍 STRING COMPARISON EVALUATION
   Total Tests: 8
   Successful: 8 (100.0%)
   Failed: 0 (0.0%)

📈 SIMILARITY METRICS
   Average String Similarity: 85.32%
   Average Model Confidence: 91.45%
   Best Similarity: 95.67%
   Worst Similarity: 72.18%

🎯 PERFORMANCE DISTRIBUTION
   🟢 High Quality (>80%): 6 (75.0%)
   🟡 Medium Quality (50-80%): 2 (25.0%)
   🔴 Low Quality (<50%): 0 (0.0%)

🏆 OVERALL PERFORMANCE SCORE: 87.2%
🏆 EXCELLENT PERFORMANCE!
```
