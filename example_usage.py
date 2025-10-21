#%%
"""
Example usage of the MCL wrapper with different models.
"""

import sys
import os
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from peft import LoraConfig  # Use standard LoraConfig
from mcl_wrapper import get_peft_mcl, MCLTrainer, patch_peft_for_mcl
import logging
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

#%%

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_usage():
    """Simple usage example"""
    print("Creating MCL model with simple interface...")

    patch_peft_for_mcl(enable=True)

    # Create standard LoRA configuration
    lora_r = 16
    lora_alpha = 16
    target_modules = ["c_attn", "c_proj"]
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        task_type="CAUSAL_LM"
    ) if target_modules else None  # Only create LoRA config if we want LoRA

    model = get_peft_mcl(
        model_name_or_path="gpt2",  # Any HuggingFace model
        num_hyps=3,
        wta_training_mode="wta",
        lora_config=lora_config,
        use_group_lora=True
    )
    
    print(f"Created model: {model.__class__.__name__}")
    print(f"Number of hypotheses: {model.num_hyps}")
    return model

#%%

model1 = example_usage()

#%%
# Create a trainer for the model
from transformers import Trainer
# Load tokenizer that matches the model
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# GPT-2 doesn't have a pad token by default, so we need to add one
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load WikiText-2 dataset for language modeling
train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:100]")

# Tokenize function for causal language modeling
def tokenize_function(examples):
    # Tokenize the text
    result = tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=512,
        padding=False,
    )
    
    # For causal language modeling, labels are the same as input_ids
    # The model will learn to predict the next token
    result["labels"] = [ids.copy() for ids in result["input_ids"]]
    
    return result

# Apply tokenization WITHOUT removing columns first
train_dataset = train_dataset.map(
    tokenize_function, 
    batched=True,
)

eval_dataset = eval_dataset.map(
    tokenize_function, 
    batched=True,
)

# Remove empty sequences (WikiText has some)
train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 0)
eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids"]) > 0)

# NOW remove the text column if input_ids exists
if "input_ids" in train_dataset.column_names:
    train_dataset = train_dataset.remove_columns(["text"])
    eval_dataset = eval_dataset.remove_columns(["text"])
    
    print("\nFinal columns:", train_dataset.column_names)
    print("Sample item keys:", train_dataset[0].keys())
    print("Sample input_ids length:", len(train_dataset[0]["input_ids"]))
else:
    print("ERROR: Tokenization failed - no input_ids found!")

#%%
# Only proceed if tokenization worked
from transformers import TrainingArguments
import torch

train_config = TrainingArguments(
    output_dir="output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    max_steps=30,
    logging_steps=10,
    remove_unused_columns=False,  # Important!
)

# Custom data collator
def custom_data_collator(features):
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    
    for f in features:
        # Convert to tensors if they're lists
        input_ids = torch.tensor(f["input_ids"]) if isinstance(f["input_ids"], list) else f["input_ids"]
        attention_mask = torch.tensor(f["attention_mask"]) if isinstance(f["attention_mask"], list) else f["attention_mask"]
        labels = torch.tensor(f["labels"]) if isinstance(f["labels"], list) else f["labels"]
        
        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)
    
    # Pad sequences to max length in batch
    max_length = max(len(ids) for ids in batch_input_ids)
    
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    
    for input_ids, attention_mask, labels in zip(batch_input_ids, batch_attention_mask, batch_labels):
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_ids = torch.cat([input_ids, torch.tensor([tokenizer.pad_token_id] * padding_length)])
            attention_mask = torch.cat([attention_mask, torch.tensor([0] * padding_length)])
            # For labels, use -100 for padding tokens (ignored in loss calculation)
            labels = torch.cat([labels, torch.tensor([-100] * padding_length)])
        
        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)
        padded_labels.append(labels)
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels),
    }

trainer = MCLTrainer(
    model=model1,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    args=train_config,
    data_collator=custom_data_collator,
)

print("Starting training...")
trainer.train()

# %%

# Generation
input_ids = eval_dataset[0]["input_ids"]
input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
print(input_text)

#%%

# Generate
input = torch.tensor(input_ids, device=model1.device).unsqueeze(0)
output_dict = {}
for hyp_idx in range(model1.num_hyps):
    output = model1.generate(inputs=input, max_new_tokens=256, hypothesis_idx=hyp_idx)
    output_dict[hyp_idx] = output

#%%

for hyp_idx in range(model1.num_hyps):
    output_text = tokenizer.decode(output_dict[hyp_idx][0], skip_special_tokens=True)
    print(f"Hypothesis {hyp_idx}: {output_text.split(input_text)[1].strip()}")

# %%
