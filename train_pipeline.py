"""
train_pipeline.py
End-to-End Pipeline Training for Propaganda Detection
1. Downloads SemEval-2020 Task 11 Data from Hugging Face
2. Trains DeBERTa-v3-base for Span Identification (SI)
3. Trains DeBERTa-v3-base for Technique Classification (TC)
"""

import os
import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)

# --- CONFIGURATION ---
# We use DeBERTa-v3-base because it outperforms RoBERTa on semantic tasks
MODEL_CHECKPOINT = "microsoft/deberta-v3-base" 
OUTPUT_DIR = "./propaganda_models"
SI_MODEL_DIR = f"{OUTPUT_DIR}/si_model"
TC_MODEL_DIR = f"{OUTPUT_DIR}/tc_model"

# The official 14 SemEval Labels (mapped to IDs)
# Note: The HF dataset might return integers, we map them to these names
TECHNIQUE_LABELS = [
    "Appeal_to_Authority",
    "Appeal_to_fear-prejudice",
    "Bandwagon",
    "Black-and-White_Fallacy",
    "Causal_Oversimplification",
    "Doubt",
    "Exaggeration,Minimisation",
    "Flag-Waving",
    "Loaded_Language",
    "Name_Calling,Labeling",
    "Obfuscation,Intentional_Vagueness,Confusion",
    "Red_Herring",
    "Reductio_ad_hitlerum",
    "Repetition",
    "Slogans",
    "Straw_Man",
    "Thought-terminating_Cliche",
    "Whataboutism" 
]
# Create mappings
label2id = {label: i for i, label in enumerate(TECHNIQUE_LABELS)}
id2label = {i: label for i, label in enumerate(TECHNIQUE_LABELS)}

# ==============================================================================
# 1. DATA PREPARATION HELPERS
# ==============================================================================

def prepare_si_dataset(example, tokenizer):
    """
    Converts Character Offsets -> Token Labels (BIO Scheme)
    0 = Outside (O)
    1 = Begin Propaganda (B-PROP)
    2 = Inside Propaganda (I-PROP)
    """
    text = example["text"]
    spans = example["span_identification"]
    
    # Tokenize (keep offsets mapping)
    tokenized = tokenizer(
        text, 
        truncation=True, 
        max_length=512, 
        return_offsets_mapping=True,
        padding="max_length"
    )
    
    offset_mapping = tokenized["offset_mapping"]
    sequence_labels = []
    
    # Create a mask of "Propaganda Characters" (1 for prop, 0 for clean)
    char_mask = np.zeros(len(text), dtype=int)
    for start, end in zip(spans["start_char_offset"], spans["end_char_offset"]):
        char_mask[start:end] = 1

    # Map Characters -> Tokens
    for idx, (start, end) in enumerate(offset_mapping):
        # Special tokens (CLS, SEP, PAD) have (0,0) or equivalent
        if start == end:
            sequence_labels.append(-100) # Ignore in loss
            continue
            
        # Check if this token falls inside a propaganda span
        # We consider a token "Propaganda" if >50% of its chars are marked
        token_chars = char_mask[start:end]
        if np.mean(token_chars) > 0.5:
            # Simple Binary Classification for SI (Propaganda vs Not)
            # You could do B-tag logic, but for detection I-tag (1) is sufficient usually
            sequence_labels.append(1) 
        else:
            sequence_labels.append(0)
            
    tokenized["labels"] = sequence_labels
    return tokenized

def prepare_tc_dataset(example):
    """
    Extracts Spans -> Classification Examples
    Returns a list of dicts {text, label}
    """
    text = example["text"]
    tc_data = example["technique_classification"]
    
    new_examples = []
    
    for start, end, label_id in zip(tc_data["start_char_offset"], tc_data["end_char_offset"], tc_data["technique"]):
        # Extract the snippet
        snippet = text[start:end]
        
        # Extract Context (100 chars before/after) for better accuracy
        ctx_start = max(0, start - 100)
        ctx_end = min(len(text), end + 100)
        context = text[ctx_start:ctx_end]
        
        # Input Format: "[CLS] snippet [SEP] context"
        # This tells the model "Focus on this snippet, but here is the background info"
        combined_text = f"{snippet} [SEP] {context}"
        
        new_examples.append({
            "text": combined_text,
            "label": label_id, # The dataset already provides IDs matching our list usually
            "snippet": snippet
        })
        
    return new_examples

# ==============================================================================
# 2. TRAINING FUNCTIONS
# ==============================================================================

def train_span_identification(dataset, tokenizer):
    print("\n>>> 🛠️  Training Span Identification (SI) Model...")
    
    # 1. Map Data
    tokenized_ds = dataset.map(
        lambda x: prepare_si_dataset(x, tokenizer), 
        batched=False,
        remove_columns=dataset["train"].column_names
    )
    
    # 2. Setup Model (2 labels: O, PROP)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=2
    )
    
    # 3. Trainer
    args = TrainingArguments(
        output_dir=f"{SI_MODEL_DIR}/checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"], # Use validation split
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer)
    )
    
    trainer.train()
    trainer.save_model(SI_MODEL_DIR)
    tokenizer.save_pretrained(SI_MODEL_DIR)
    print(f"✅ SI Model saved to {SI_MODEL_DIR}")

def train_technique_classification(dataset, tokenizer):
    print("\n>>> 🛠️  Training Technique Classification (TC) Model...")
    
    # 1. Flatten Dataset (Article -> Multiple Examples)
    train_list = []
    for ex in dataset["train"]:
        train_list.extend(prepare_tc_dataset(ex))
        
    val_list = []
    for ex in dataset["validation"]:
        val_list.extend(prepare_tc_dataset(ex))
        
    train_ds = Dataset.from_list(train_list)
    val_ds = Dataset.from_list(val_list)
    
    print(f"    Converted to {len(train_ds)} classification examples.")

    # 2. Tokenize
    def tokenize_func(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    
    train_ds = train_ds.map(tokenize_func, batched=True)
    val_ds = val_ds.map(tokenize_func, batched=True)
    
    # 3. Setup Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(TECHNIQUE_LABELS),
        id2label=id2label,
        label2id=label2id
    )
    
    # 4. Trainer
    args = TrainingArguments(
        output_dir=f"{TC_MODEL_DIR}/checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5, # Slightly higher for classification
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer
    )
    
    trainer.train()
    trainer.save_model(TC_MODEL_DIR)
    tokenizer.save_pretrained(TC_MODEL_DIR)
    print(f"✅ TC Model saved to {TC_MODEL_DIR}")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print(">>> 📥 Loading SemEval-2020 Task 11 Dataset from Hugging Face...")
    # This automatically downloads the dataset structure you described
    raw_dataset = load_dataset("SemEvalWorkshop/sem_eval_2020_task_11")
    
    # Initialize Tokenizer (Shared for both)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    # Step 1: Train Span Identifier
    train_span_identification(raw_dataset, tokenizer)
    
    # Step 2: Train Technique Classifier
    train_technique_classification(raw_dataset, tokenizer)
    
    print("\n🎉 ALL TRAINING COMPLETE!")
    print(f"Models are ready in {OUTPUT_DIR}/")
    print("Next: Run 'python profiler.py [url] [country] --model local'")