"""
Complete imbalance handling utilities.
Includes loss functions, samplers, and metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Sampler
from typing import List, Dict
from collections import Counter


# ==================== LOSS FUNCTIONS ====================


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss using effective number of samples"""

    def __init__(
        self,
        samples_per_class: List[int],
        beta: float = 0.9999,
        loss_type: str = "focal",
    ):
        super().__init__()
        self.samples_per_class = samples_per_class
        self.beta = beta
        self.loss_type = loss_type

        # Compute effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * len(samples_per_class)

        self.class_weights = torch.tensor(weights, dtype=torch.float32)

        # Initialize base loss
        if loss_type == "focal":
            self.base_loss = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.base_loss = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.class_weights = self.class_weights.to(inputs.device)

        if self.loss_type == "focal":
            return self.base_loss(inputs, targets)
        elif self.loss_type == "ce":
            return F.cross_entropy(inputs, targets, weight=self.class_weights)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


# ==================== SAMPLING STRATEGIES ====================


class BalancedBatchSampler(Sampler):
    """Sample batches with balanced class distribution"""

    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        n_batches: int = None,
        alpha: float = 1.0,
    ):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.alpha = alpha

        # Get class indices
        self.class_indices = {}
        for class_id in np.unique(labels):
            self.class_indices[class_id] = np.where(self.labels == class_id)[0]

        self.num_classes = len(self.class_indices)

        # Ensure batch_size is compatible
        if batch_size % self.num_classes != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by num_classes ({self.num_classes})"
            )

        self.samples_per_class = batch_size // self.num_classes

        # Calculate number of batches
        if n_batches is None:
            min_class_size = min(
                len(indices) for indices in self.class_indices.values()
            )
            self.n_batches = min_class_size // self.samples_per_class
        else:
            self.n_batches = n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            batch_indices = []

            for class_id in self.class_indices.keys():
                # Sample with replacement from this class
                class_batch = np.random.choice(
                    self.class_indices[class_id],
                    size=self.samples_per_class,
                    replace=True,
                )
                batch_indices.extend(class_batch.tolist())

            # Shuffle within batch
            np.random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.n_batches


class ImbalancedDatasetSampler(Sampler):
    """Sample elements to balance class distribution using weighted sampling"""

    def __init__(self, labels: List[int], replacement: bool = True):
        self.labels = labels
        self.replacement = replacement

        # Calculate weights
        label_counts = Counter(labels)
        weights = [1.0 / label_counts[label] for label in labels]
        self.weights = torch.DoubleTensor(weights)
        self.num_samples = len(labels)

    def __iter__(self):
        return iter(
            torch.multinomial(
                self.weights, self.num_samples, replacement=self.replacement
            ).tolist()
        )

    def __len__(self):
        return self.num_samples


# ==================== ANALYSIS UTILITIES ====================


def analyze_class_distribution(labels: List[int], dataset_name: str = "Dataset"):
    """Analyze and report class distribution statistics"""
    from collections import Counter

    label_counts = Counter(labels)
    total = len(labels)

    print(f"\n{'=' * 60}")
    print(f"Class Distribution Analysis: {dataset_name}")
    print(f"{'=' * 60}")
    print(f"Total samples: {total}")
    print(f"\nPer-class breakdown:")

    for class_id, count in sorted(label_counts.items()):
        percentage = (count / total) * 100
        print(f"  Class {class_id}: {count:5d} samples ({percentage:5.2f}%)")

    # Compute imbalance ratio
    counts = list(label_counts.values())
    imbalance_ratio = max(counts) / min(counts)

    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")

    if imbalance_ratio > 10:
        print("⚠️  SEVERE imbalance detected! Strongly recommend using:")
        print("   1. Focal Loss or Class-Balanced Loss")
        print("   2. Balanced Batch Sampler")
    elif imbalance_ratio > 3:
        print("⚠️  Moderate imbalance detected. Recommend using:")
        print("   1. Weighted Cross-Entropy or Focal Loss")
        print("   2. Monitor minority class F1-score")
    else:
        print("✅ Relatively balanced dataset")

    print(f"{'=' * 60}\n")

    return {
        "total": total,
        "class_counts": dict(label_counts),
        "imbalance_ratio": imbalance_ratio,
    }


def compute_class_weights(labels: List[int], method: str = "balanced") -> torch.Tensor:
    """Compute class weights for loss functions"""
    from collections import Counter

    label_counts = Counter(labels)
    num_classes = len(label_counts)
    total = len(labels)

    weights = np.zeros(num_classes)

    if method == "balanced":
        for class_id, count in label_counts.items():
            weights[class_id] = total / (num_classes * count)
    elif method == "sqrt":
        for class_id, count in label_counts.items():
            weights[class_id] = np.sqrt(total / count)
    elif method == "effective":
        beta = 0.9999
        for class_id, count in label_counts.items():
            effective_num = 1.0 - np.power(beta, count)
            weights[class_id] = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)
