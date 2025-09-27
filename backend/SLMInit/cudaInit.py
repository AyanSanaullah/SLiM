import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ----- CONFIG -----
base_model = "gpt2"         # replace with your small base model (local or HF)
data_path = "../UserFacing/db/LLMData.json"   # JSONL: {"prompt": "...", "answer": "..."}
output_dir = "./cuda_lora_out"
# ------------------

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
    raise RuntimeError("CUDA is not available. This script requires CUDA for training.")

# Check if data file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"Data file exists: {os.path.exists(data_path)}")

# Load tokenizer & model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,    # requires bitsandbytes
        device_map="auto"     # will push to CUDA automatically
    )
    model = prepare_model_for_kbit_training(model)
except Exception as e:
    print(f"Error loading model with 8-bit quantization: {e}")
    print("Falling back to full precision...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto"
    )

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],  # works for GPT-2; adjust for other models
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Dataset
print("Loading dataset...")
ds = load_dataset("json", data_files=data_path, split="train")
print(f"Dataset loaded with {len(ds)} examples")

# Split dataset into 80% training and 20% testing
print("Splitting dataset into training (80%) and testing (20%)...")
split_ds = ds.train_test_split(test_size=0.2, seed=42)
train_ds = split_ds["train"]
test_ds = split_ds["test"]

print(f"Training examples: {len(train_ds)}")
print(f"Testing examples: {len(test_ds)}")

# Save test dataset for later evaluation
test_data_path = "../UserFacing/db/LLMTestData.json"
print(f"Saving test dataset to {test_data_path}...")
test_ds.to_json(test_data_path, orient="records", lines=False, indent=2)

def preprocess(batch):
    prompt = f"### Instruction:\n{batch['prompt']}\n\n### Response:\n"
    # Handle both 'answer' and 'expected_output' keys for compatibility
    answer = batch.get("answer", batch.get("expected_output", ""))
    full = prompt + answer

    tokenized = tokenizer(full, truncation=True, max_length=512, padding="max_length")
    labels = tokenized["input_ids"][:]
    # mask out prompt tokens
    prompt_len = len(tokenizer(prompt)["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels
    return tokenized

# Use only training data for preprocessing and training
train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)

# Training
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,               # mixed precision for CUDA
    logging_steps=10,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
)

print("Starting training...")
trainer.train()
print("Training completed!")

print(f"Saving model to {output_dir}...")
trainer.save_model(output_dir)
print("Model saved successfully!")
