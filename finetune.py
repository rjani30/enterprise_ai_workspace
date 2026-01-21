from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Load the Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/deepseek-coder-v2-lite-bnb-4bit", # Lite version for local use
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 2. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank for LoRA
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    use_gradient_checkpointing = "unsloth", # Saves VRAM
)

# 3. Load Dataset (from the UI upload)
dataset = load_dataset("json", data_files={"train": "data/train_data.jsonl"}, split="train")

# 4. Initialize Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps = 60, # Keep it short for local testing
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
    ),
)

# 5. Run Training and Save the Adapter
trainer.train()
model.save_pretrained_merged("trained_model_adapter", tokenizer, save_method = "lora")
print("Training Complete! Adapter saved to 'trained_model_adapter'.")