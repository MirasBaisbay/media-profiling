"""
train_pipeline.py
End-to-End Pipeline Training for Propaganda Detection (SemEval 2020 Task 11)

This script loads the local PTC (Propaganda Techniques Corpus) dataset and trains:
1. Span Identification (SI): Token classification with BIO tagging to find WHERE propaganda is
2. Technique Classification (TC): Sequence classification to identify WHICH technique is used

Dataset Structure Expected:
    datasets/
    ├── train/
    │   ├── articles/
    │   │   ├── article111111111.txt
    │   │   └── ...
    │   └── labels/
    │       ├── article111111111.task-flc-tc.labels
    │       └── ...
    ├── dev/
    │   ├── articles/
    │   └── labels/
    └── test/
        ├── articles/
        └── labels/

Label File Format (tab-separated):
    Article_ID    Technique_Label    Start_Offset    End_Offset
"""

import os
import re
import glob
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging

from config import (
    PROPAGANDA_TECHNIQUES,
    LABEL2ID,
    ID2LABEL,
    SI_LABELS,
    SI_LABEL2ID,
    SI_ID2LABEL,
    LEGACY_LABEL_MAPPING,
    EXCLUDED_LABELS,
    DATASET_DIR,
    TRAIN_DIR,
    DEV_DIR,
    TEST_DIR,
    ModelConfig,
    TrainingConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize configs
model_config = ModelConfig()
training_config = TrainingConfig()


# =============================================================================
# 1. LOCAL DATA LOADING
# =============================================================================

def find_article_files(data_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Finds all article text files and their corresponding label files.
    """
    article_files = {}

    # Find all .txt files (article content)
    txt_patterns = [
        os.path.join(data_dir, "*.txt"),
        os.path.join(data_dir, "articles", "*.txt"),
        os.path.join(data_dir, "**", "*.txt"),
    ]

    txt_files = []
    for pattern in txt_patterns:
        txt_files.extend(glob.glob(pattern, recursive=True))
    txt_files = list(set(txt_files))  # Remove duplicates

    for txt_path in txt_files:
        # Extract article ID from filename
        basename = os.path.basename(txt_path)
        match = re.match(r"article(\d+)\.txt", basename)
        if not match:
            continue
        article_id = match.group(1)

        # UPDATED: Added your specific filename pattern
        possible_filenames = [
            f"article{article_id}.task2-TC.labels",     # <--- YOUR SPECIFIC FILE PATTERN
            f"article{article_id}.task-flc-tc.labels", # Standard SemEval
            f"article{article_id}.labels",              # Simplified
            f"article{article_id}.task-si.labels"       # Task 1 specific
        ]

        # Look in multiple directories
        search_dirs = [
            os.path.dirname(txt_path), # Same dir
            os.path.join(os.path.dirname(txt_path), "labels"), # Subdir relative to txt
            os.path.join(os.path.dirname(os.path.dirname(txt_path)), "labels"), # Parallel dir
            os.path.join(data_dir, "labels"), # Root labels dir
        ]

        label_path = None
        for s_dir in search_dirs:
            for fname in possible_filenames:
                p = os.path.join(s_dir, fname)
                if os.path.exists(p):
                    label_path = p
                    break
            if label_path: break

        article_files[article_id] = {
            "text_path": txt_path,
            "label_path": label_path,
        }

    return article_files


def parse_label_file(label_path: str) -> List[Dict]:
    """
    Parses a .labels file with tab-separated columns:
    Article_ID, Technique_Label, Start_Offset, End_Offset

    Args:
        label_path: Path to the .labels file

    Returns:
        List of dicts with keys: technique, start, end
    """
    labels = []

    if not label_path or not os.path.exists(label_path):
        return labels

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                continue

            article_id, technique, start, end = parts[:4]

            # Skip excluded labels (V2)
            if technique in EXCLUDED_LABELS:
                continue

            # Map legacy labels to V2 merged labels
            technique = LEGACY_LABEL_MAPPING.get(technique, technique)

            # Skip if technique is not in our final label set
            if technique not in LABEL2ID:
                logger.warning(f"Unknown technique '{technique}' in {label_path}, skipping")
                continue

            labels.append({
                "technique": technique,
                "start": int(start),
                "end": int(end),
            })

    return labels


def load_local_ptc_data(data_dir: str) -> List[Dict]:
    """
    Loads all articles and labels from a local PTC data directory.

    Args:
        data_dir: Path to the data split directory (e.g., datasets/train)

    Returns:
        List of dicts, each containing:
        - article_id: str
        - text: str (full article text)
        - labels: List[Dict] with keys: technique, start, end
    """
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory not found: {data_dir}")
        return []

    article_files = find_article_files(data_dir)
    logger.info(f"Found {len(article_files)} articles in {data_dir}")

    data = []
    for article_id, paths in article_files.items():
        text_path = paths["text_path"]
        label_path = paths["label_path"]

        # Read article text
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Parse labels
        labels = parse_label_file(label_path)

        data.append({
            "article_id": article_id,
            "text": text,
            "labels": labels,
        })

    logger.info(f"Loaded {len(data)} articles with {sum(len(d['labels']) for d in data)} total spans")
    return data


def load_all_splits() -> Dict[str, List[Dict]]:
    """
    Loads train, dev, and test splits from the datasets directory.

    Returns:
        Dict with keys 'train', 'validation', 'test' mapping to lists of article dicts
    """
    splits = {}

    # Try multiple possible directory names
    train_dirs = [TRAIN_DIR, os.path.join(DATASET_DIR, "training")]
    dev_dirs = [DEV_DIR, os.path.join(DATASET_DIR, "validation"), os.path.join(DATASET_DIR, "dev-labels")]
    test_dirs = [TEST_DIR, os.path.join(DATASET_DIR, "test-articles")]

    for dirs, split_name in [(train_dirs, "train"), (dev_dirs, "validation"), (test_dirs, "test")]:
        for d in dirs:
            if os.path.exists(d):
                splits[split_name] = load_local_ptc_data(d)
                break
        if split_name not in splits:
            splits[split_name] = []
            logger.warning(f"No data found for {split_name} split")

    return splits


# =============================================================================
# 2. SPAN IDENTIFICATION (SI) - BIO TAGGING
# =============================================================================

def create_char_labels(text: str, labels: List[Dict]) -> np.ndarray:
    """
    Creates a character-level label array marking propaganda spans.

    Args:
        text: Article text
        labels: List of span annotations

    Returns:
        numpy array of shape (len(text),) with values:
        - 0: Outside (O)
        - 1: Beginning of span (B-PROP)
        - 2: Inside span (I-PROP)
    """
    char_labels = np.zeros(len(text), dtype=np.int32)

    for label in labels:
        start, end = label["start"], label["end"]
        # Ensure bounds are valid
        start = max(0, min(start, len(text)))
        end = max(0, min(end, len(text)))

        if start < end:
            char_labels[start] = 1  # B-PROP
            if end > start + 1:
                char_labels[start + 1:end] = 2  # I-PROP

    return char_labels


def prepare_si_example(
    text: str,
    labels: List[Dict],
    tokenizer,
    max_length: int = 512
) -> Dict:
    """
    Prepares a single example for Span Identification training.

    Converts character-level annotations to token-level BIO labels using
    the tokenizer's offset_mapping feature.

    Args:
        text: Article text
        labels: List of span annotations
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Dict with input_ids, attention_mask, and labels for token classification
    """
    # Tokenize with offset mapping
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors=None,
    )

    # Create character-level labels
    char_labels = create_char_labels(text, labels)

    # Map character labels to token labels
    offset_mapping = tokenized["offset_mapping"]
    token_labels = []

    prev_was_prop = False  # Track if previous token was propaganda

    for idx, (start, end) in enumerate(offset_mapping):
        # Special tokens (CLS, SEP, PAD) have (0, 0)
        if start == end:
            token_labels.append(-100)  # Ignore in loss computation
            prev_was_prop = False
            continue

        # Get character labels for this token's span
        token_char_labels = char_labels[start:end]

        if len(token_char_labels) == 0:
            token_labels.append(0)  # O
            prev_was_prop = False
            continue

        # Determine token label based on character labels
        # If any char is labeled as propaganda, token is propaganda
        if np.any(token_char_labels > 0):
            # Check if this is start of a new span
            if token_char_labels[0] == 1 or not prev_was_prop:
                token_labels.append(1)  # B-PROP
            else:
                token_labels.append(2)  # I-PROP
            prev_was_prop = True
        else:
            token_labels.append(0)  # O
            prev_was_prop = False

    tokenized["labels"] = token_labels

    # Remove offset_mapping as it's not needed for training
    del tokenized["offset_mapping"]

    return tokenized


def create_si_dataset(data: List[Dict], tokenizer, max_length: int = 512) -> Dataset:
    """
    Creates a HuggingFace Dataset for Span Identification training.

    Args:
        data: List of article dicts from load_local_ptc_data
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        HuggingFace Dataset ready for training
    """
    processed = []

    for article in data:
        example = prepare_si_example(
            article["text"],
            article["labels"],
            tokenizer,
            max_length
        )
        example["article_id"] = article["article_id"]
        processed.append(example)

    return Dataset.from_list(processed)


# =============================================================================
# 3. TECHNIQUE CLASSIFICATION (TC)
# =============================================================================

def prepare_tc_examples(
    text: str,
    labels: List[Dict],
    article_id: str,
    context_window: int = 100
) -> List[Dict]:
    """
    Prepares training examples for Technique Classification.

    Each propaganda span becomes a separate classification example with format:
    [CLS] Context [SEP] Propaganda_Snippet [SEP]

    Args:
        text: Full article text
        labels: List of span annotations with technique labels
        article_id: Article identifier
        context_window: Characters of context to include before/after snippet

    Returns:
        List of dicts with: text, label, snippet, article_id
    """
    examples = []

    for label_info in labels:
        start, end = label_info["start"], label_info["end"]
        technique = label_info["technique"]

        # Extract propaganda snippet
        snippet = text[start:end]

        # Extract surrounding context
        ctx_start = max(0, start - context_window)
        ctx_end = min(len(text), end + context_window)
        context = text[ctx_start:ctx_end]

        # Format: "[CLS] Context [SEP] Propaganda_Snippet [SEP]"
        # The model will learn to classify the snippet given its context
        combined_text = f"{context} [SEP] {snippet}"

        # Get label ID
        label_id = LABEL2ID[technique]

        examples.append({
            "text": combined_text,
            "label": label_id,
            "snippet": snippet,
            "article_id": article_id,
            "technique_name": technique,
        })

    return examples


def create_tc_dataset(
    data: List[Dict],
    tokenizer,
    max_length: int = 256,
    context_window: int = 100
) -> Dataset:
    """
    Creates a HuggingFace Dataset for Technique Classification training.

    "Explodes" the dataset so each propaganda span becomes a separate example.

    Args:
        data: List of article dicts from load_local_ptc_data
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        context_window: Characters of context around snippet

    Returns:
        HuggingFace Dataset ready for training
    """
    all_examples = []

    for article in data:
        if not article["labels"]:
            continue

        examples = prepare_tc_examples(
            article["text"],
            article["labels"],
            article["article_id"],
            context_window
        )
        all_examples.extend(examples)

    if not all_examples:
        logger.warning("No TC examples created - check if labels are present")
        return Dataset.from_list([])

    # Tokenize all examples
    texts = [ex["text"] for ex in all_examples]
    labels = [ex["label"] for ex in all_examples]

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    # Create dataset
    dataset_dict = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
        "article_id": [ex["article_id"] for ex in all_examples],
        "snippet": [ex["snippet"] for ex in all_examples],
        "technique_name": [ex["technique_name"] for ex in all_examples],
    }

    return Dataset.from_dict(dataset_dict)


# =============================================================================
# 4. METRICS
# =============================================================================

def compute_si_metrics(eval_pred):
    """
    Computes metrics for Span Identification (token classification).
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Flatten and filter out ignored indices (-100)
    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                true_labels.append(label)
                pred_labels.append(pred)

    # Binary: propaganda (1 or 2) vs non-propaganda (0)
    true_binary = [1 if l > 0 else 0 for l in true_labels]
    pred_binary = [1 if p > 0 else 0 for p in pred_labels]

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_binary, pred_binary, average="binary", zero_division=0
    )
    accuracy = accuracy_score(true_binary, pred_binary)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_tc_metrics(eval_pred):
    """
    Computes metrics for Technique Classification.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="micro", zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)

    # Also compute per-class metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision_micro": precision,
        "recall_micro": recall,
        "f1_micro": f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }


# =============================================================================
# 5. CLASS WEIGHTS AND CUSTOM TRAINERS
# =============================================================================

def compute_si_class_weights(train_dataset: Dataset, max_weight_ratio: float = 10.0) -> torch.Tensor:
    """
    Computes class weights for Span Identification based on label distribution.

    Uses sqrt-dampened inverse frequency weighting with a maximum cap to handle
    severe class imbalance without causing the model to predict all propaganda.

    Args:
        train_dataset: Training dataset with 'labels' field
        max_weight_ratio: Maximum ratio between highest and lowest weight (default: 10.0)

    Returns:
        torch.Tensor of shape (3,) with weights for [O, B-PROP, I-PROP]
    """
    label_counts = Counter()

    for example in train_dataset:
        labels = example["labels"]
        for label in labels:
            if label != -100:  # Skip ignored tokens
                label_counts[label] += 1

    total = sum(label_counts.values())
    num_classes = 3  # O, B-PROP, I-PROP

    # Compute sqrt-dampened inverse frequency weights
    # sqrt dampening prevents extreme weights while still addressing imbalance
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)  # Avoid division by zero
        # Use sqrt of inverse frequency for gentler weighting
        weight = np.sqrt(total / (num_classes * count))
        weights.append(weight)

    weights = np.array(weights)

    # Cap the maximum weight ratio to prevent extreme imbalance
    min_weight = weights.min()
    max_allowed = min_weight * max_weight_ratio
    weights = np.clip(weights, None, max_allowed)

    # Normalize so minimum weight is 1.0
    weights = weights / weights.min()

    weights = torch.tensor(weights, dtype=torch.float32)

    logger.info(f"SI Class distribution: O={label_counts.get(0, 0)}, "
                f"B-PROP={label_counts.get(1, 0)}, I-PROP={label_counts.get(2, 0)}")
    logger.info(f"SI Class weights: O={weights[0]:.3f}, B-PROP={weights[1]:.3f}, I-PROP={weights[2]:.3f}")

    return weights


def compute_tc_class_weights(train_dataset: Dataset, max_weight_ratio: float = 10.0) -> torch.Tensor:
    """
    Computes class weights for Technique Classification based on label distribution.

    Uses sqrt-dampened inverse frequency weighting with a maximum cap.

    Args:
        train_dataset: Training dataset with 'labels' field
        max_weight_ratio: Maximum ratio between highest and lowest weight (default: 10.0)

    Returns:
        torch.Tensor of shape (num_classes,) with weights for each technique
    """
    label_counts = Counter()

    for example in train_dataset:
        label = example["labels"]
        label_counts[label] += 1

    total = sum(label_counts.values())
    num_classes = len(PROPAGANDA_TECHNIQUES)

    # Compute sqrt-dampened inverse frequency weights
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)  # Avoid division by zero
        weight = np.sqrt(total / (num_classes * count))
        weights.append(weight)

    weights = np.array(weights)

    # Cap the maximum weight ratio
    min_weight = weights.min()
    max_allowed = min_weight * max_weight_ratio
    weights = np.clip(weights, None, max_allowed)

    # Normalize so minimum weight is 1.0
    weights = weights / weights.min()

    weights = torch.tensor(weights, dtype=torch.float32)

    # Log class distribution
    logger.info("TC Class distribution:")
    for i, technique in enumerate(PROPAGANDA_TECHNIQUES):
        logger.info(f"  {technique}: {label_counts.get(i, 0)} samples, weight={weights[i]:.3f}")

    return weights


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focal loss down-weights easy examples and focuses on hard misclassified examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        weight: Class weights tensor
        gamma: Focusing parameter (default=2.0)
        reduction: 'mean' or 'sum'
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean', ignore_index=-100):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.weight,
            reduction='none',
            ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class WeightedSITrainer(Trainer):
    """
    Custom Trainer for Span Identification with class-weighted loss.

    Handles the severe class imbalance between O/B-PROP/I-PROP tokens.
    """
    def __init__(self, class_weights=None, use_focal_loss=False, focal_gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Reshape for loss computation
        # logits: (batch, seq_len, num_classes) -> (batch * seq_len, num_classes)
        # labels: (batch, seq_len) -> (batch * seq_len)
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        # Move class weights to correct device
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
        else:
            weights = None

        if self.use_focal_loss:
            loss_fct = FocalLoss(
                weight=weights,
                gamma=self.focal_gamma,
                ignore_index=-100
            )
            loss = loss_fct(logits_flat, labels_flat)
        else:
            loss = F.cross_entropy(
                logits_flat,
                labels_flat,
                weight=weights,
                ignore_index=-100
            )

        return (loss, outputs) if return_outputs else loss


class WeightedTCTrainer(Trainer):
    """
    Custom Trainer for Technique Classification with class-weighted loss.

    Handles class imbalance among the 14 propaganda techniques.
    """
    def __init__(self, class_weights=None, use_focal_loss=False, focal_gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Move class weights to correct device
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
        else:
            weights = None

        if self.use_focal_loss:
            loss_fct = FocalLoss(
                weight=weights,
                gamma=self.focal_gamma
            )
            loss = loss_fct(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels, weight=weights)

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# 6. TRAINING FUNCTIONS
# =============================================================================

def train_span_identification(
    train_data: List[Dict],
    val_data: List[Dict],
    tokenizer,
) -> None:
    """
    Trains the Span Identification model using BIO tagging.

    Uses class-weighted loss and optional focal loss to handle the severe
    class imbalance between O/B-PROP/I-PROP tokens.

    Args:
        train_data: Training articles with labels
        val_data: Validation articles with labels
        tokenizer: HuggingFace tokenizer
    """
    logger.info(">>> Training Span Identification (SI) Model...")

    # Create datasets
    train_ds = create_si_dataset(train_data, tokenizer, model_config.max_length)
    val_ds = create_si_dataset(val_data, tokenizer, model_config.max_length)

    logger.info(f"    SI Train samples: {len(train_ds)}")
    logger.info(f"    SI Val samples: {len(val_ds)}")

    if len(train_ds) == 0:
        logger.error("No training data for SI. Skipping.")
        return

    # Compute class weights for handling imbalance
    class_weights = None
    if training_config.use_class_weights:
        logger.info("    Computing SI class weights...")
        class_weights = compute_si_class_weights(train_ds, training_config.max_class_weight_ratio)

    # Initialize model (3 labels: O, B-PROP, I-PROP)
    model = AutoModelForTokenClassification.from_pretrained(
        model_config.model_checkpoint,
        num_labels=len(SI_LABELS),
        id2label=SI_ID2LABEL,
        label2id=SI_LABEL2ID,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{training_config.si_model_dir}/checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=training_config.learning_rate_si,
        per_device_train_batch_size=training_config.batch_size_si,
        per_device_eval_batch_size=training_config.batch_size_si,
        num_train_epochs=training_config.num_epochs_si,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=f"{training_config.si_model_dir}/logs",
        logging_steps=50,  # More frequent logging
        report_to="none",
    )

    # Data collator for token classification
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Remove non-tensor columns for training
    train_ds_clean = train_ds.remove_columns(["article_id"])
    val_ds_clean = val_ds.remove_columns(["article_id"])

    # Initialize custom weighted trainer
    if training_config.use_class_weights or training_config.use_focal_loss:
        logger.info(f"    Using weighted trainer (focal_loss={training_config.use_focal_loss}, "
                    f"gamma={training_config.focal_loss_gamma})")
        trainer = WeightedSITrainer(
            class_weights=class_weights,
            use_focal_loss=training_config.use_focal_loss,
            focal_gamma=training_config.focal_loss_gamma,
            model=model,
            args=training_args,
            train_dataset=train_ds_clean,
            eval_dataset=val_ds_clean,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_si_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=training_config.si_early_stopping_patience
            )],
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds_clean,
            eval_dataset=val_ds_clean,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_si_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=training_config.si_early_stopping_patience
            )],
        )

    # Train
    trainer.train()

    # Save model
    trainer.save_model(training_config.si_model_dir)
    tokenizer.save_pretrained(training_config.si_model_dir)
    logger.info(f"SI Model saved to {training_config.si_model_dir}")


def train_technique_classification(
    train_data: List[Dict],
    val_data: List[Dict],
    tokenizer,
) -> None:
    """
    Trains the Technique Classification model.

    Uses class-weighted loss and optional focal loss to handle class
    imbalance among the 14 propaganda techniques.

    Args:
        train_data: Training articles with labels
        val_data: Validation articles with labels
        tokenizer: HuggingFace tokenizer
    """
    logger.info(">>> Training Technique Classification (TC) Model...")

    # Create datasets (exploded - one example per span)
    train_ds = create_tc_dataset(
        train_data,
        tokenizer,
        model_config.tc_max_length,
        model_config.context_window
    )
    val_ds = create_tc_dataset(
        val_data,
        tokenizer,
        model_config.tc_max_length,
        model_config.context_window
    )

    logger.info(f"    TC Train samples: {len(train_ds)} (exploded from articles)")
    logger.info(f"    TC Val samples: {len(val_ds)}")

    if len(train_ds) == 0:
        logger.error("No training data for TC. Skipping.")
        return

    # Compute class weights for handling imbalance
    class_weights = None
    if training_config.use_class_weights:
        logger.info("    Computing TC class weights...")
        class_weights = compute_tc_class_weights(train_ds, training_config.max_class_weight_ratio)

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_checkpoint,
        num_labels=len(PROPAGANDA_TECHNIQUES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{training_config.tc_model_dir}/checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=training_config.learning_rate_tc,
        per_device_train_batch_size=training_config.batch_size_tc,
        per_device_eval_batch_size=training_config.batch_size_tc,
        num_train_epochs=training_config.num_epochs_tc,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_dir=f"{training_config.tc_model_dir}/logs",
        logging_steps=50,  # More frequent logging
        report_to="none",
    )

    # Remove non-tensor columns for training
    columns_to_remove = ["article_id", "snippet", "technique_name"]
    train_ds_clean = train_ds.remove_columns(columns_to_remove)
    val_ds_clean = val_ds.remove_columns(columns_to_remove)

    # Initialize custom weighted trainer
    if training_config.use_class_weights or training_config.use_focal_loss:
        logger.info(f"    Using weighted trainer (focal_loss={training_config.use_focal_loss}, "
                    f"gamma={training_config.focal_loss_gamma})")
        trainer = WeightedTCTrainer(
            class_weights=class_weights,
            use_focal_loss=training_config.use_focal_loss,
            focal_gamma=training_config.focal_loss_gamma,
            model=model,
            args=training_args,
            train_dataset=train_ds_clean,
            eval_dataset=val_ds_clean,
            processing_class=tokenizer,
            compute_metrics=compute_tc_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=training_config.tc_early_stopping_patience
            )],
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds_clean,
            eval_dataset=val_ds_clean,
            processing_class=tokenizer,
            compute_metrics=compute_tc_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=training_config.tc_early_stopping_patience
            )],
        )

    # Train
    trainer.train()

    # Save model
    trainer.save_model(training_config.tc_model_dir)
    tokenizer.save_pretrained(training_config.tc_model_dir)
    logger.info(f"TC Model saved to {training_config.tc_model_dir}")


# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================

def main():
    """
    Main training pipeline entry point.
    """
    logger.info("=" * 60)
    logger.info("SemEval 2020 Task 11 - Propaganda Detection Training Pipeline")
    logger.info("=" * 60)

    # Verify dataset directory exists
    if not os.path.exists(DATASET_DIR):
        logger.error(f"Dataset directory not found: {DATASET_DIR}")
        logger.error("Please ensure your SemEval data is in the 'datasets/' folder.")
        logger.error("Expected structure:")
        logger.error("  datasets/")
        logger.error("  ├── train/")
        logger.error("  │   ├── articles/")
        logger.error("  │   │   └── article*.txt")
        logger.error("  │   └── labels/")
        logger.error("  │       └── article*.task-flc-tc.labels")
        logger.error("  ├── dev/")
        logger.error("  └── test/")
        return

    # Load all data splits
    logger.info(">>> Loading local PTC dataset...")
    splits = load_all_splits()

    train_data = splits.get("train", [])
    val_data = splits.get("validation", [])
    test_data = splits.get("test", [])

    logger.info(f"    Train articles: {len(train_data)}")
    logger.info(f"    Validation articles: {len(val_data)}")
    logger.info(f"    Test articles: {len(test_data)}")

    if not train_data:
        logger.error("No training data found. Cannot proceed.")
        return

    val_has_labels = False
    if val_data:
        val_has_labels = any(len(article['labels']) > 0 for article in val_data)

    # If no validation data OR validation data is empty of labels, split the training set
    if not val_data or not val_has_labels:
        logger.warning("Validation data missing or has no labels. Using 10% of training data for validation.")
        split_idx = int(len(train_data) * 0.9)
        # Slicing the list
        val_data = train_data[split_idx:]
        train_data = train_data[:split_idx]

    # Initialize tokenizer
    logger.info(f">>> Loading tokenizer: {model_config.model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_checkpoint)

    # Create output directories
    os.makedirs(training_config.si_model_dir, exist_ok=True)
    os.makedirs(training_config.tc_model_dir, exist_ok=True)

    # Train models
    logger.info("")
    train_span_identification(train_data, val_data, tokenizer)

    logger.info("")
    train_technique_classification(train_data, val_data, tokenizer)

    logger.info("")
    logger.info("=" * 60)
    logger.info("ALL TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"SI Model: {training_config.si_model_dir}")
    logger.info(f"TC Model: {training_config.tc_model_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Evaluate models on test set")
    logger.info("  2. Run inference: python profiler.py [url] [country] --model local")


if __name__ == "__main__":
    main()
