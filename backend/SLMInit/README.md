# LoRA Fine-Tuning Setup Guide

This guide will help you set up and use LoRA (Low-Rank Adaptation) fine-tuning for GPT-2 models on your NVIDIA RTX 4060 GPU.

## Prerequisites

- **NVIDIA GPU**: RTX 4060 (or any CUDA-compatible GPU)
- **Python**: 3.8 or higher
- **CUDA Drivers**: Latest NVIDIA drivers installed
- **Memory**: At least 8GB RAM (16GB recommended)

## Quick Start

### 1. Install Dependencies

#### Option A: Automated Installation (Recommended)
```powershell
# Navigate to the SLMInit directory
cd backend/SLMInit

# Run the installation script
.\install_cuda_pytorch.bat
```

#### Option B: Manual Installation
```powershell
# Uninstall existing PyTorch (if any)
pip uninstall torch torchvision torchaudio -y

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 2. Verify CUDA Installation
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

### 3. Prepare Your Data
Ensure your training data is in the correct format at `../UserFacing/db/LLMData.json`:

```json
[
  {
    "prompt": "write a 100 word story about a car",
    "answer": "The '57 Chevy, nicknamed \"Betsy,\" was more than just a car..."
  },
  {
    "prompt": "write a short story about a dog",
    "answer": "Barnaby, a scruffy terrier mix with one perpetually floppy ear..."
  }
]
```

## Usage

### Training a LoRA Model

Run the training script:
```powershell
python cudaInit.py
```

**What happens during training:**
- Loads GPT-2 model with 8-bit quantization (if available)
- Applies LoRA adapters for efficient fine-tuning
- Trains on your custom dataset
- Saves the trained model to `./cuda_lora_out/`

**Training parameters (configurable in `cudaInit.py`):**
- **Base Model**: GPT-2 (can be changed to other models)
- **LoRA Rank**: 8 (controls adaptation strength)
- **Learning Rate**: 2e-4
- **Batch Size**: 2
- **Epochs**: 1
- **Max Length**: 512 tokens

### Testing the Trained Model

Run the inference script:
```powershell
python testSuite.py
```

**What happens during testing:**
- Loads the base GPT-2 model
- Applies your trained LoRA adapters
- Generates responses to test prompts
- Displays the generated text

## Configuration

### Model Settings (`cudaInit.py`)

```python
# ----- CONFIG -----
base_model = "gpt2"         # Base model to fine-tune
data_path = "../UserFacing/db/LLMData.json"   # Training data path
output_dir = "./cuda_lora_out"                # Where to save the model
# ------------------
```

### LoRA Settings

```python
lora_config = LoraConfig(
    r=8,                    # Rank (higher = more parameters)
    lora_alpha=32,          # Scaling factor
    target_modules=["c_attn"],  # Which layers to adapt
    lora_dropout=0.05,      # Dropout rate
    bias="none",            # Bias handling
    task_type="CAUSAL_LM",  # Task type
)
```

### Training Settings

```python
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,    # Batch size per GPU
    num_train_epochs=1,               # Number of training epochs
    learning_rate=2e-4,               # Learning rate
    fp16=True,                        # Mixed precision training
    logging_steps=10,                 # Log every N steps
    save_strategy="epoch",            # When to save checkpoints
)
```

## Troubleshooting

### CUDA Not Available Error

**Problem**: `RuntimeError: CUDA is not available`

**Solutions**:
1. **Install CUDA-enabled PyTorch**:
   ```powershell
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Check NVIDIA drivers**:
   - Download latest drivers from [NVIDIA website](https://www.nvidia.com/drivers/)
   - Restart your computer after installation

3. **Verify CUDA installation**:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

### Out of Memory Error

**Problem**: `CUDA out of memory`

**Solutions**:
1. **Reduce batch size**:
   ```python
   per_device_train_batch_size=1  # Instead of 2
   ```

2. **Use gradient accumulation**:
   ```python
   gradient_accumulation_steps=2
   ```

3. **Reduce sequence length**:
   ```python
   max_length=256  # Instead of 512
   ```

### Model Loading Errors

**Problem**: `FileNotFoundError: LoRA model not found`

**Solution**: Train the model first using `cudaInit.py` before running `testSuite.py`

### Data Format Errors

**Problem**: Dataset loading fails

**Solution**: Ensure your JSON file has the correct format:
- Array of objects
- Each object has "prompt" and "answer" keys
- Valid JSON syntax

## File Structure

```
backend/SLMInit/
├── cudaInit.py              # Training script
├── testSuite.py             # Inference/testing script
├── requirements.txt         # Python dependencies
├── install_cuda_pytorch.bat # Windows installation script
├── install_cuda_pytorch.sh  # Linux/Mac installation script
└── README.md               # This file
```

## Performance Tips

### For RTX 4060 (8GB VRAM):
- **Batch Size**: 1-2
- **Sequence Length**: 256-512 tokens
- **LoRA Rank**: 8-16
- **Use 8-bit quantization**: Enabled by default

### For Better Results:
- **More Training Data**: 100+ examples minimum
- **Longer Training**: Increase epochs to 2-3
- **Better Prompts**: Use consistent, clear instruction format
- **Validation Split**: Add validation data for monitoring

## Advanced Usage

### Custom Base Models
Replace `"gpt2"` with other models:
```python
base_model = "microsoft/DialoGPT-medium"  # For conversational AI
base_model = "distilgpt2"                 # Smaller, faster model
```

### Custom LoRA Targets
For different model architectures:
```python
# For GPT-2
target_modules=["c_attn", "c_proj"]

# For LLaMA
target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]

# For T5
target_modules=["q", "v", "k", "o"]
```

### Monitoring Training
Add Weights & Biases logging:
```python
training_args = TrainingArguments(
    # ... other args ...
    report_to="wandb",
    run_name="lora-gpt2-finetune",
)
```

## Support

If you encounter issues:
1. Check the error messages carefully
2. Verify CUDA installation
3. Ensure data format is correct
4. Check available GPU memory
5. Review the troubleshooting section above

## Next Steps

After successful training:
1. Test your model with various prompts
2. Fine-tune hyperparameters for better results
3. Integrate the model into your application
4. Consider deploying to production with proper optimization

Happy fine-tuning! 🚀
