import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
base_model = "gpt2"
lora_model_path = "./cuda_lora_out"

# Check CUDA availability
if not torch.cuda.is_available():
    print("CUDA is not available. This could be due to:")
    print("1. PyTorch was installed without CUDA support")
    print("2. CUDA drivers are not installed")
    print("3. CUDA version mismatch")
    print("\nTo fix this:")
    print("1. Uninstall current PyTorch: pip uninstall torch")
    print("2. Install CUDA-enabled PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("3. Or use the requirements.txt: pip install -r requirements.txt")
    raise RuntimeError("CUDA is not available. This script requires CUDA for inference.")

# Check if LoRA model exists
if not os.path.exists(lora_model_path):
    raise FileNotFoundError(f"LoRA model not found at {lora_model_path}. Please train the model first using cudaInit.py")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name()}")

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the base GPT-2 model onto CUDA
print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(base_model).to("cuda")

# Load the trained LoRA adapters on top of the base model
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base, lora_model_path).to("cuda")

# Build a prompt in the same style you trained on
prompt = "### Instruction:\nExplain what CUDA is.\n\n### Response:\n"

print(f"Prompt: {prompt}")

# Tokenize and move to CUDA
print("Tokenizing input...")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate an answer
print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,   # how many new tokens to generate
        do_sample=True,       # stochastic decoding
        temperature=0.7,      # controls randomness
        pad_token_id=tokenizer.eos_token_id
    )

# Decode back to text
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nGenerated response:\n{response}")
