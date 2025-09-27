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
