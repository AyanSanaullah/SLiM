# Multi-Agent Training System

Comprehensive system for training 5 specialized AI agents simultaneously with extensive datasets and advanced evaluation.

## 🎯 Overview

This system creates and trains 5 expert agents with deep domain knowledge:

1. **Python Expert** - Web development, frameworks, best practices
2. **ML Expert** - Machine learning, deep learning, NLP, MLOps  
3. **DevOps Expert** - CI/CD, containerization, cloud platforms
4. **Data Science Expert** - Statistics, analysis, visualization
5. **Cybersecurity Expert** - Security protocols, threat detection

## 📊 Training Datasets

Each agent is trained with comprehensive datasets containing 20 detailed Q&A pairs covering:

### Python Expert Dataset
- Web development with Flask, Django, FastAPI
- Advanced Python concepts (decorators, generators, async/await)
- Database operations and ORMs
- Testing and debugging
- Performance optimization
- Package creation and deployment

### ML Expert Dataset  
- Algorithm selection and comparison
- Model evaluation and validation
- Feature engineering techniques
- Deep learning implementations
- NLP and computer vision
- MLOps and deployment strategies

### DevOps Expert Dataset
- CI/CD pipeline setup
- Containerization with Docker and Kubernetes
- Infrastructure as Code
- Monitoring and observability
- Security best practices
- Cloud platform management

### Data Science Expert Dataset
- Exploratory data analysis
- Statistical concepts and hypothesis testing
- Data preprocessing and cleaning
- Visualization techniques
- Time series analysis
- Business intelligence

### Cybersecurity Expert Dataset
- Security fundamentals (CIA Triad)
- Authentication and authorization
- Penetration testing methodologies
- Incident response procedures
- Threat detection and prevention
- Compliance and risk management

## 🚀 Quick Start

### Option 1: Automated Quick Start
```bash
python3 quick_start_multiple_agents.py
```

### Option 2: Manual Setup
```bash
# 1. Start services
python3 app.py &
cd ../string-comparison && python3 backend.py &

# 2. Return to agents directory
cd ../agents

# 3. Train all agents
python3 train_multiple_agents.py
```

## 📋 Training Process

The training system performs the following steps:

1. **Service Verification** - Checks if required services are running
2. **Dataset Loading** - Loads comprehensive training datasets
3. **Agent Creation** - Creates advanced agents with JSON datasets
4. **Parallel Training** - Trains up to 3 agents simultaneously
5. **Progress Monitoring** - Tracks training status in real-time
6. **Inference Testing** - Tests each agent with domain-specific questions
7. **Report Generation** - Creates detailed training reports

## 🧪 Testing Framework

Each agent is tested with 5 specialized questions covering:

- **Python Expert**: Flask APIs, framework comparison, optimization, decorators, exception handling
- **ML Expert**: Algorithm selection, overfitting prevention, imbalanced data, feature engineering, model evaluation
- **DevOps Expert**: CI/CD setup, containerization, monitoring, security, deployment strategies
- **Data Science Expert**: EDA, statistics, missing data, outliers, feature engineering
- **Cybersecurity Expert**: Security principles, authentication, penetration testing, incident response, attack vectors

## 📊 Performance Metrics

The system tracks and reports:

- **Training Success Rate** - Percentage of successfully trained agents
- **Inference Success Rate** - Percentage of successful question responses
- **Response Quality** - Length and relevance of generated responses
- **Training Time** - Time taken for each agent to complete training
- **Resource Utilization** - System resource usage during training

## 📁 File Structure

```
agents/
├── training_datasets/
│   ├── python_expert_dataset.json
│   ├── ml_expert_dataset.json
│   ├── devops_expert_dataset.json
│   ├── data_science_expert_dataset.json
│   └── cybersecurity_expert_dataset.json
├── train_multiple_agents.py
├── quick_start_multiple_agents.py
└── MULTI_AGENT_TRAINING.md
```

## 🔧 Configuration

### Training Parameters
- **Base Model**: distilbert-base-uncased
- **Max Training Time**: 10 minutes per agent
- **Parallel Workers**: 3 simultaneous training processes
- **Test Questions**: 5 questions per agent
- **Response Timeout**: 30 seconds per inference

### Dataset Format
Each dataset follows this structure:
```json
{
  "user_id": "expert_name",
  "description": "Expert description",
  "training_data": [
    {
      "prompt": "Question text",
      "answer": "Comprehensive answer",
      "category": "category_name"
    }
  ]
}
```

## 📈 Expected Results

After successful training, you should see:

- **5 Specialized Agents** ready for inference
- **High Success Rates** (>90% for training and inference)
- **Detailed Reports** with performance metrics
- **Domain-Specific Responses** showing deep knowledge

## 🛠️ Troubleshooting

### Common Issues

1. **Services Not Running**
   ```bash
   # Check service status
   curl http://localhost:8080/health
   curl http://localhost:8000/health
   
   # Start services manually
   python3 app.py &
   cd ../string-comparison && python3 backend.py &
   ```

2. **Training Timeout**
   - Increase max_wait parameter in training script
   - Check system resources (CPU, memory)
   - Ensure stable network connection

3. **Dataset Loading Errors**
   - Verify JSON file format
   - Check file paths and permissions
   - Ensure UTF-8 encoding

4. **Inference Failures**
   - Wait for training to complete
   - Check agent status: `curl localhost:8080/api/v1/agents/agent_id/status`
   - Verify string comparison service is running

## 📊 Monitoring and Reports

The system generates comprehensive reports including:

- **Training Summary** - Overall success metrics
- **Agent Details** - Individual performance data
- **Test Results** - Question-answer pairs and success rates
- **Timestamps** - Training and testing times
- **JSON Export** - Detailed data for further analysis

## 🎯 Advanced Usage

### Custom Datasets
To add new expert agents:

1. Create new dataset file in `training_datasets/`
2. Follow the JSON format structure
3. Add agent configuration to `train_multiple_agents.py`
4. Include test questions for evaluation

### Parallel Training
The system uses ThreadPoolExecutor for parallel training:
- Maximum 3 concurrent training processes
- Automatic load balancing
- Progress monitoring for each agent
- Error handling and recovery

### Custom Testing
Modify test questions in the agent configurations:
```python
'test_questions': [
    "Your custom question 1",
    "Your custom question 2",
    # ... more questions
]
```

## 🔍 Verification

After training completion, verify agents are working:

```bash
# List all agents
curl localhost:8080/api/v1/agents

# Test specific agent
curl -X POST localhost:8080/api/v1/agents/python_expert/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How to create a REST API with Flask?"}'
```

## 📚 Related Documentation

- [HOW_TO_USE.md](HOW_TO_USE.md) - Basic usage guide
- [COMPLETE_COMMANDS.md](COMPLETE_COMMANDS.md) - All available commands
- [REQUESTS_GUIDE.md](REQUESTS_GUIDE.md) - API usage examples
- [README.md](README.md) - General project documentation

---

**Note**: This system requires significant computational resources. Ensure adequate CPU, memory, and storage space for optimal performance.
