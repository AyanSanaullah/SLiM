# LoRA Fine-Tuning Setup Guide - CPU & Mac

This guide covers setting up LoRA fine-tuning for systems **without NVIDIA GPUs**, including:
- **Mac M1/M2/M3** (Apple Silicon)
- **Intel Macs** (without eGPU)
- **Windows/Linux** systems without NVIDIA GPUs
- **Cloud instances** with CPU-only

## Prerequisites

- **Python**: 3.8 or higher
- **Memory**: At least 8GB RAM (16GB+ recommended)
- **Storage**: 5GB+ free space for models and data
- **Mac M1/M2/M3**: Use Apple Silicon optimized PyTorch
- **Intel Macs**: Use Intel-optimized PyTorch

## Quick Start

### 1. Install Dependencies

#### Option A: Automated Installation
```bash
# Navigate to the SLMInit directory
cd backend/SLMInit

# Run the installation script
chmod +x install_cpu_pytorch.sh
./install_cpu_pytorch.sh
```

#### Option B: Manual Installation

**For Mac M1/M2/M3 (Apple Silicon):**
```bash
# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio -y

# Install Apple Silicon optimized PyTorch
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements_cpu.txt
```

**For Intel Macs and CPU-only systems:**
```bash
# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio -y

# Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements_cpu.txt
```

### 2. Verify Installation
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CPU available: {torch.cuda.is_available() == False}")  # Should be True for CPU
print(f"Device: {'MPS' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'CPU'}")
```

## CPU-Optimized Files

### Modified Training Script (`cudaInit_cpu.py`)

```python
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ----- CONFIG -----
base_model = "gpt2"         # Use smaller models for CPU
data_path = "../UserFacing/db/LLMData.json"
output_dir = "./cpu_lora_out"
# ------------------

# Check if we're on Mac with MPS support
device = "cpu"
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Metal Performance Shaders (MPS)")
else:
    print("Using CPU")

# Check if data file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

print(f"Device: {device}")
print(f"Data file exists: {os.path.exists(data_path)}")

# Load tokenizer & model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
# Use smaller model for CPU training
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float32,  # Use float32 for better CPU compatibility
    device_map=None  # Don't use device_map for CPU
)

# Move model to appropriate device
model = model.to(device)

# LoRA config (smaller rank for CPU)
lora_config = LoraConfig(
    r=4,  # Reduced from 8
    lora_alpha=16,  # Reduced from 32
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Dataset
print("Loading dataset...")
ds = load_dataset("json", data_files=data_path, split="train")
print(f"Dataset loaded with {len(ds)} examples")

def preprocess(batch):
    prompt = f"### Instruction:\n{batch['prompt']}\n\n### Response:\n"
    full = prompt + batch["answer"]

    tokenized = tokenizer(full, truncation=True, max_length=256, padding="max_length")  # Reduced from 512
    labels = tokenized["input_ids"][:]
    # mask out prompt tokens
    prompt_len = len(tokenizer(prompt)["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels
    return tokenized

ds = ds.map(preprocess, remove_columns=ds.column_names)

# Training (CPU-optimized settings)
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,  # Reduced for CPU
    gradient_accumulation_steps=4,  # Compensate for smaller batch size
    num_train_epochs=1,
    learning_rate=1e-4,  # Slightly lower learning rate
    fp16=False,  # Disable mixed precision for CPU
    logging_steps=10,
    save_strategy="epoch",
    dataloader_num_workers=0,  # Disable multiprocessing for stability
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
)

print("Starting training...")
trainer.train()
print("Training completed!")

print(f"Saving model to {output_dir}...")
trainer.save_model(output_dir)
print("Model saved successfully!")
```

### Modified Testing Script (`testSuite_cpu.py`)

```python
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
base_model = "gpt2"
lora_model_path = "./cpu_lora_out"

# Check device
device = "cpu"
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Metal Performance Shaders (MPS)")
else:
    print("Using CPU")

# Check if LoRA model exists
if not os.path.exists(lora_model_path):
    raise FileNotFoundError(f"LoRA model not found at {lora_model_path}. Please train the model first using cudaInit_cpu.py")

print(f"Device: {device}")

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the base GPT-2 model
print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(base_model).to(device)

# Load the trained LoRA adapters
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base, lora_model_path).to(device)

# Build a prompt
prompt = "### Instruction:\nExplain what machine learning is.\n\n### Response:\n"

print(f"Prompt: {prompt}")

# Tokenize and move to device
print("Tokenizing input...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate an answer
print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,  # Reduced for CPU
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        num_beams=1,  # Disable beam search for speed
    )

# Decode back to text
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nGenerated response:\n{response}")
```

## CPU-Optimized Requirements

Create `requirements_cpu.txt`:

```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
peft>=0.4.0
accelerate>=0.20.0
numpy>=1.24.0
```

## Performance Optimizations

### For Mac M1/M2/M3:
- **Use MPS**: Automatic GPU acceleration on Apple Silicon
- **Batch Size**: 1-2
- **Sequence Length**: 128-256 tokens
- **LoRA Rank**: 4-8

### For CPU-Only Systems:
- **Batch Size**: 1
- **Sequence Length**: 128-256 tokens
- **LoRA Rank**: 2-4
- **Gradient Accumulation**: 4-8 steps
- **Use smaller models**: `distilgpt2` instead of `gpt2`

## Installation Scripts

### `install_cpu_pytorch.sh` (Mac/Linux)
```bash
#!/bin/bash
echo "Installing CPU-optimized PyTorch..."
echo

# Detect if we're on Apple Silicon
if [[ $(uname -m) == "arm64" ]]; then
    echo "Detected Apple Silicon (M1/M2/M3)"
    echo "Installing Apple Silicon optimized PyTorch..."
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio
else
    echo "Detected Intel/AMD processor"
    echo "Installing CPU-only PyTorch..."
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo
echo "Installing other dependencies..."
pip install -r requirements_cpu.txt

echo
echo "Installation complete! Testing installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'Device: {\"MPS\" if hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available() else \"CPU\"}')"
```

### `install_cpu_pytorch.bat` (Windows)
```batch
@echo off
echo Installing CPU-optimized PyTorch...
echo.

echo Uninstalling existing PyTorch...
pip uninstall torch torchvision torchaudio -y

echo.
echo Installing CPU-only PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Installing other dependencies...
pip install -r requirements_cpu.txt

echo.
echo Installation complete! Testing installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print('Device: CPU')"

pause
```

## Usage

### Training (CPU/Mac)
```bash
python cudaInit_cpu.py
```

### Testing (CPU/Mac)
```bash
python testSuite_cpu.py
```

## Performance Expectations

### Mac M1/M2/M3:
- **Training Time**: 2-5x slower than RTX 4060
- **Memory Usage**: 4-8GB RAM
- **Inference Speed**: 1-3 seconds per response

### CPU-Only Systems:
- **Training Time**: 5-10x slower than RTX 4060
- **Memory Usage**: 2-6GB RAM
- **Inference Speed**: 3-10 seconds per response

## Troubleshooting

### MPS Not Available (Mac)
**Problem**: MPS backend not detected

**Solution**:
```python
# Check MPS availability
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

### Out of Memory (CPU)
**Problem**: System runs out of RAM

**Solutions**:
1. **Reduce batch size to 1**
2. **Use gradient accumulation**
3. **Reduce sequence length to 128**
4. **Use smaller model**: `distilgpt2`

### Slow Training
**Problem**: Training is very slow

**Solutions**:
1. **Use smaller models**: `distilgpt2` instead of `gpt2`
2. **Reduce LoRA rank**: 2-4 instead of 8
3. **Shorter sequences**: 128-256 tokens
4. **Fewer training examples**: 50-100 instead of 500+

## Alternative: Cloud Solutions

For better performance without NVIDIA GPU:

### Google Colab (Free)
- Free GPU access (T4)
- Pre-installed PyTorch with CUDA
- Use the original `cudaInit.py` script

### AWS/Azure/GCP
- Rent GPU instances
- Use the original CUDA scripts
- Pay per hour usage

### Hugging Face Spaces
- Deploy models for free
- Use CPU-optimized versions
- Share with teammates

## File Structure

```
backend/SLMInit/
├── cudaInit.py              # Original CUDA training script
├── testSuite.py             # Original CUDA testing script
├── cudaInit_cpu.py          # CPU/Mac training script
├── testSuite_cpu.py         # CPU/Mac testing script
├── requirements.txt         # CUDA dependencies
├── requirements_cpu.txt     # CPU dependencies
├── install_cpu_pytorch.sh   # Mac/Linux CPU installation
├── install_cpu_pytorch.bat  # Windows CPU installation
├── README.md               # CUDA setup guide
└── README_CPU_MAC.md       # This file
```

## Next Steps

1. **Choose the right script** for your system
2. **Install dependencies** using the appropriate method
3. **Start with small datasets** to test performance
4. **Consider cloud solutions** for production use
5. **Optimize settings** based on your hardware

Happy fine-tuning on any system! 🚀
