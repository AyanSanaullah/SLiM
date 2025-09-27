# Model Evaluation System

This enhanced evaluation system automatically splits your training data, evaluates model performance, and sends results to the frontend via API.

## 🔄 **What Changed**

### Training Scripts (`cudaInit.py` & `cudaInit_cpu.py`)
- **Data Split**: Automatically splits data into 80% training / 20% testing
- **Test Data Storage**: Saves test data for later evaluation
- **Consistent Splits**: Uses `seed=42` for reproducible splits

### Enhanced Test Suites
- **`testSuite_enhanced.py`**: CUDA version with advanced metrics (ROUGE, semantic similarity)
- **`testSuite_enhanced_cpu.py`**: CPU version with basic metrics (optimized for performance)
- **Automatic Evaluation**: Tests model on held-out data and calculates performance metrics
- **API Integration**: Sends results to frontend automatically

### New API Endpoints
- **`POST /test_results`**: Receive evaluation results
- **`GET /test_results`**: Get latest results by model type
- **`GET /test_results/list`**: List all evaluation runs
- **`GET /test_results/summary`**: Compare CUDA vs CPU performance

## 🚀 **How to Use**

### 1. Train Your Models (Now with Data Splitting)
```bash
cd backend/SLMInit

# For CUDA (creates test data at ../UserFacing/db/LLMTestData.json)
python cudaInit.py

# For CPU (creates test data at ../UserFacing/db/LLMTestData_CPU.json)  
python cudaInit_cpu.py
```

### 2. Start the API Server
```bash
cd ../UserFacing
python start_slm_api.py
```

### 3. Run Evaluations

#### Option A: Run Individual Evaluations
```bash
cd ../SLMInit

# CUDA evaluation (advanced metrics)
python testSuite_enhanced.py

# CPU evaluation (basic metrics)
python testSuite_enhanced_cpu.py
```

#### Option B: Run All Evaluations
```bash
# Run both CUDA and CPU evaluations
python run_evaluation.py

# Run specific model type
python run_evaluation.py --model-type cuda
python run_evaluation.py --model-type cpu
```

### 4. View Results via API

#### Get Latest Results
```bash
# CUDA results
curl "http://localhost:5000/test_results?model_type=cuda"

# CPU results  
curl "http://localhost:5000/test_results?model_type=cpu"
```

#### Get Summary Comparison
```bash
curl "http://localhost:5000/test_results/summary"
```

#### List All Test Runs
```bash
curl "http://localhost:5000/test_results/list"
```

## 📊 **Evaluation Metrics**

### CUDA Version (Advanced Metrics)
- **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L for text similarity
- **Semantic Similarity**: Cosine similarity using sentence embeddings
- **Generation Speed**: Time per response
- **Length Analysis**: Generated vs reference text length ratios

### CPU Version (Basic Metrics)
- **Word Overlap**: Percentage of shared words with reference
- **Character Similarity**: Length-based similarity measure
- **Generation Speed**: Time per response
- **Length Analysis**: Generated vs reference text length ratios

## 📁 **File Structure**

```
backend/SLMInit/
├── cudaInit.py                    # CUDA training (now with data splitting)
├── cudaInit_cpu.py                # CPU training (now with data splitting)
├── testSuite_enhanced.py          # CUDA evaluation with advanced metrics
├── testSuite_enhanced_cpu.py      # CPU evaluation with basic metrics
├── run_evaluation.py              # Evaluation runner script
├── requirements_evaluation.txt    # Additional packages for evaluation
└── EVALUATION_README.md           # This file

backend/UserFacing/
├── app.py                         # Flask app (now with test result endpoints)
├── db/
│   ├── LLMData.json              # Original training data
│   ├── LLMTestData.json          # CUDA test data (20% split)
│   ├── LLMTestData_CPU.json      # CPU test data (20% split)
│   ├── test_results_cuda_*.json  # CUDA evaluation results
│   ├── test_results_cpu_*.json   # CPU evaluation results
│   ├── latest_test_results_cuda.json  # Latest CUDA results
│   └── latest_test_results_cpu.json   # Latest CPU results
```

## 🔧 **Installation**

### Basic Requirements (Already Installed)
```bash
pip install torch transformers peft datasets flask flask-cors
```

### Enhanced Evaluation Tools (Optional)
```bash
pip install -r requirements_evaluation.txt
```

This adds:
- `rouge-score`: For ROUGE metrics
- `sentence-transformers`: For semantic similarity
- `scikit-learn`: For cosine similarity calculations

## 📈 **Example API Responses**

### Test Results Summary
```json
{
  "success": true,
  "summary": {
    "cuda_results": {
      "total_examples": 5,
      "successful_generations": 5,
      "average_generation_time": 0.234,
      "average_rouge1": 0.456,
      "average_semantic_similarity": 0.678
    },
    "cpu_results": {
      "total_examples": 5,
      "successful_generations": 5,
      "average_generation_time": 2.145,
      "average_word_overlap": 0.543,
      "average_char_similarity": 0.721
    },
    "comparison": {
      "speed_comparison": {
        "cuda_avg_time": 0.234,
        "cpu_avg_time": 2.145,
        "speedup_factor": 9.17
      },
      "accuracy_comparison": {
        "cuda_success_rate": 100.0,
        "cpu_success_rate": 100.0
      }
    }
  }
}
```

### Individual Test Result
```json
{
  "test_id": "20241227_143022",
  "model_type": "CUDA",
  "summary_metrics": {
    "total_examples": 5,
    "successful_generations": 5,
    "average_generation_time": 0.234,
    "average_rouge1": 0.456
  },
  "test_results": [
    {
      "example_id": 1,
      "prompt": "write a short story about a car",
      "reference_answer": "The '57 Chevy, nicknamed \"Betsy,\"...",
      "generated_answer": "There once was a car named Lightning...",
      "generation_time": 0.245,
      "metrics": {
        "rouge1_f": 0.423,
        "semantic_similarity": 0.678,
        "length_ratio": 0.89
      }
    }
  ]
}
```

## 🎯 **Integration with Frontend**

The evaluation results are automatically sent to your Flask API and can be consumed by any frontend:

### JavaScript Example
```javascript
// Get latest test summary
fetch('http://localhost:5000/test_results/summary')
  .then(response => response.json())
  .then(data => {
    console.log('CUDA vs CPU Performance:', data.summary.comparison);
  });

// Get detailed results for a specific model
fetch('http://localhost:5000/test_results?model_type=cuda')
  .then(response => response.json())
  .then(data => {
    console.log('CUDA Test Results:', data.test_results);
  });
```

### React Component Example
```jsx
function ModelPerformance() {
  const [summary, setSummary] = useState(null);
  
  useEffect(() => {
    fetch('/test_results/summary')
      .then(res => res.json())
      .then(data => setSummary(data.summary));
  }, []);
  
  if (!summary) return <div>Loading...</div>;
  
  return (
    <div>
      <h2>Model Performance Comparison</h2>
      <p>CUDA Speed: {summary.cuda_results?.average_generation_time}s</p>
      <p>CPU Speed: {summary.cpu_results?.average_generation_time}s</p>
      <p>Speedup: {summary.comparison?.speed_comparison?.speedup_factor}x</p>
    </div>
  );
}
```

## 🔍 **Troubleshooting**

### "Test data not found"
- Make sure you've run the training scripts (`cudaInit.py` or `cudaInit_cpu.py`) first
- The training scripts now automatically create test data files

### "Model not found"
- Train your model first using the appropriate training script
- Check that the output directories (`cuda_lora_out` or `cpu_lora_out`) exist

### "API connection failed"
- Make sure the Flask server is running: `python start_slm_api.py`
- Check that the server is accessible at `http://localhost:5000`

### "Evaluation packages missing"
- Install evaluation requirements: `pip install -r requirements_evaluation.txt`
- The CPU version works without additional packages

## 🎉 **Benefits**

1. **Automated Testing**: No manual intervention needed
2. **Consistent Evaluation**: Same test data across runs
3. **Performance Tracking**: Compare models over time
4. **Frontend Integration**: Results automatically available via API
5. **Comprehensive Metrics**: Multiple evaluation approaches
6. **Easy Comparison**: CUDA vs CPU performance analysis

Your SLM training pipeline now includes comprehensive evaluation and reporting! 🚀
