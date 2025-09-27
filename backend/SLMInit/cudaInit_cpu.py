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

# Split dataset into 80% training and 20% testing
print("Splitting dataset into training (80%) and testing (20%)...")
split_ds = ds.train_test_split(test_size=0.2, seed=42)
train_ds = split_ds["train"]
test_ds = split_ds["test"]

print(f"Training examples: {len(train_ds)}")
print(f"Testing examples: {len(test_ds)}")

# Save test dataset for later evaluation
test_data_path = "../UserFacing/db/LLMTestData_CPU.json"
print(f"Saving test dataset to {test_data_path}...")
test_ds.to_json(test_data_path, orient="records", lines=False, indent=2)

def preprocess(batch):
    prompt = f"### Instruction:\n{batch['prompt']}\n\n### Response:\n"
    # Handle both 'answer' and 'expected_output' keys for compatibility
    answer = batch.get("answer", batch.get("expected_output", ""))
    full = prompt + answer

    tokenized = tokenizer(full, truncation=True, max_length=256, padding="max_length")  # Reduced from 512
    labels = tokenized["input_ids"][:]
    # mask out prompt tokens
    prompt_len = len(tokenizer(prompt)["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels
    return tokenized

# Use only training data for preprocessing and training
train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)

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
    train_dataset=train_ds,
)

print("Starting training...")
trainer.train()
print("Training completed!")

print(f"Saving model to {output_dir}...")
trainer.save_model(output_dir)
print("Model saved successfully!")
