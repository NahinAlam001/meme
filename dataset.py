"""
Dataset loader for Bengali Hateful Meme (BHM) dataset.
Supports both Task 1 (binary) and Task 2 (multi-class) classification.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Optional
import pandas as pd
from transformers import CLIPProcessor, AutoTokenizer, BartTokenizer
from collections import Counter


# Task 2 label mappings
TASK2_LABEL_MAP = {
    'TI': 0,  # Target Individual
    'TC': 1,  # Target Community
    'TO': 2,  # Target Organization
    'TS': 3,  # Target Society
}

TASK2_LABEL_NAMES = ['Target Individual', 'Target Community', 'Target Organization', 'Target Society']


def load_excel_to_lists(
    excel_path: str,
    meme_dir: str,
    task: str = "task1"
) -> Tuple[List[str], List[str], List[int], List[str]]:
    """
    Load data from Excel file for either task1 or task2.
    
    Args:
        excel_path: Path to Excel file
        meme_dir: Directory containing meme images
        task: "task1" (binary) or "task2" (multi-class)
    
    Returns:
        Tuple of (image_paths, texts, labels, rationales)
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    print(f"📂 Loading data from {excel_path}")
    df = pd.read_excel(excel_path)
    
    print(f"   Dataset: {task.upper()}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Expected columns: image_name, Captions, Labels
    if 'image_name' not in df.columns:
        raise ValueError("Missing 'image_name' column in Excel file")
    if 'Captions' not in df.columns:
        raise ValueError("Missing 'Captions' column in Excel file")
    if 'Labels' not in df.columns:
        raise ValueError("Missing 'Labels' column in Excel file")
    
    # Check for rationale column
    rationale_col = None
    for col in ['rationale', 'Rationale', 'explanation', 'Explanation']:
        if col in df.columns:
            rationale_col = col
            break
    
    # Define label mapping based on task
    if task == "task1":
        label_map = {
            "hate": 1,
            "Hate": 1,
            "HATE": 1,
            "hateful": 1,
            "non-hate": 0,
            "non hate": 0,
            "Non-Hate": 0,
            "NOT HATE": 0,
            "not hate": 0,
        }
        num_classes = 2
    elif task == "task2":
        label_map = TASK2_LABEL_MAP
        num_classes = 4
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'task1' or 'task2'")
    
    image_paths = []
    texts = []
    labels = []
    rationales = []
    
    skipped = 0
    label_errors = []
    
    for idx, row in df.iterrows():
        # Get image path
        img_name = str(row['image_name']).strip()
        img_path = os.path.join(meme_dir, img_name)
        
        # Check if image exists
        if not os.path.exists(img_path):
            # Try different extensions
            base_name = os.path.splitext(img_name)[0]
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                test_path = os.path.join(meme_dir, base_name + ext)
                if os.path.exists(test_path):
                    img_path = test_path
                    found = True
                    break
            
            if not found:
                skipped += 1
                continue
        
        # Get text (handle Bengali text)
        text = str(row['Captions']) if pd.notna(row['Captions']) else ""
        
        # Get label
        label_value = str(row['Labels']).strip()
        
        if label_value not in label_map:
            label_errors.append((idx, label_value))
            skipped += 1
            continue
        
        label = label_map[label_value]
        
        # Get rationale (if available)
        rationale = ""
        if rationale_col and rationale_col in df.columns:
            rationale = str(row[rationale_col]) if pd.notna(row[rationale_col]) else ""
        
        # Append to lists
        image_paths.append(img_path)
        texts.append(text)
        labels.append(label)
        rationales.append(rationale)
    
    print(f"   ✅ Loaded {len(labels)} samples (skipped {skipped})")
    
    if label_errors:
        print(f"   ⚠️  Label errors found: {len(label_errors)}")
        for idx, label_val in label_errors[:5]:
            print(f"      Row {idx}: '{label_val}'")
    
    # Print class distribution
    label_counts = Counter(labels)
    print(f"   📊 Class distribution:")
    
    if task == "task1":
        print(f"      Non-hate (0): {label_counts.get(0, 0)}")
        print(f"      Hate (1): {label_counts.get(1, 0)}")
    else:
        for label_idx, label_name in enumerate(TASK2_LABEL_NAMES):
            print(f"      {label_name} ({label_idx}): {label_counts.get(label_idx, 0)}")
    
    # Calculate imbalance ratio
    counts = list(label_counts.values())
    if counts:
        imbalance_ratio = max(counts) / min(counts)
        print(f"   ⚖️  Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    return image_paths, texts, labels, rationales


class BanglaMemeDataset(Dataset):
    """
    Dataset for Bengali Hateful Meme detection.
    Supports both binary (task1) and multi-class (task2) classification.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        texts: List[str],
        labels: List[int],
        rationales: List[str],
        clip_processor: CLIPProcessor,
        text_tokenizer: AutoTokenizer,
        rationale_tokenizer: BartTokenizer,
        max_length: int = 128,
        task: str = "task1"
    ):
        assert len(image_paths) == len(texts) == len(labels) == len(rationales), \
            "All input lists must have the same length"
        
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.rationales = rationales
        self.clip_processor = clip_processor
        self.text_tokenizer = text_tokenizer
        self.rationale_tokenizer = rationale_tokenizer
        self.max_length = max_length
        self.task = task
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int):
        # Load and preprocess image
        try:
            image_pil = Image.open(self.image_paths[idx]).convert("RGB")
            image = self.clip_processor(images=image_pil, return_tensors="pt")['pixel_values'].squeeze(0)
        except Exception as e:
            print(f"⚠️ Error loading image {self.image_paths[idx]}: {e}")
            # Return a blank image as fallback
            image_pil = Image.new('RGB', (224, 224), color='white')
            image = self.clip_processor(images=image_pil, return_tensors="pt")['pixel_values'].squeeze(0)
        
        # Tokenize text (handles Bengali Unicode properly)
        text_tokenized = self.text_tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = text_tokenized["input_ids"].squeeze(0)
        attention_mask = text_tokenized["attention_mask"].squeeze(0)
        
        # Label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Tokenize rationale
        rationale_tokenized = self.rationale_tokenizer(
            self.rationales[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        rationale_ids = rationale_tokenized["input_ids"].squeeze(0)
        rationale_mask = rationale_tokenized["attention_mask"].squeeze(0)
        
        return (
            image,
            input_ids,
            attention_mask,
            label,
            rationale_ids,
            rationale_mask,
            self.rationales[idx]  # Original text for evaluation
        )
