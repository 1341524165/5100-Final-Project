import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

# QUICK TEST for a simple start of DPO training

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# 1. Cache a smallll gpt2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. Cache a small RLHF dataset (Anthropic HH)
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:100]")

# 3. DPO training configuration
config = DPOConfig(
    output_dir="./simple_results",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    beta=0.1,  # DPO core parameter
    max_steps=10,
    logging_steps=1,
    remove_unused_columns=False,  # Keep all columns for DPO
    report_to="none",  # Disable wandb logging..
)

# 4. Create the trainer
# Note: In newer TRL versions, tokenizer is passed via processing_class or in config
trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,  # Updated API: use processing_class instead of tokenizer
)

# 5. Start training
print("Starting training...")
trainer.train()
print("Done! Model saved in ./simple_results/")
