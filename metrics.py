"""
Comprehensive metrics tracking for imbalanced classification.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
)
from typing import Dict, List
from evaluate import load


class MetricsTracker:
    """Track and compute classification and generation metrics"""

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.bleu_metric = load("bleu")
        self.rouge_metric = load("rouge")

        # Initialize accumulators
        self.reset()

    def reset(self):
        """Reset all accumulators"""
        self.predictions = []
        self.labels = []
        self.probabilities = []
        self.generated_rationales = []
        self.reference_rationales = []
        self.losses = {"total": [], "ce": [], "rgcl": [], "rationale": []}

    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        loss_dict: Dict[str, float],
        probabilities: torch.Tensor = None,
        generated_rationales: List[str] = None,
        reference_rationales: List[str] = None,
    ):
        """Update metrics with batch results"""
        self.predictions.extend(predictions.cpu().numpy().tolist())
        self.labels.extend(labels.cpu().numpy().tolist())

        if probabilities is not None:
            # FIX: Detach gradients before converting to numpy
            self.probabilities.extend(probabilities.detach().cpu().numpy().tolist())

        for key, value in loss_dict.items():
            if key in self.losses:
                self.losses[key].append(value)

        if generated_rationales and reference_rationales:
            for gen, ref in zip(generated_rationales, reference_rationales):
                if ref and ref.strip():
                    self.generated_rationales.append(gen)
                    self.reference_rationales.append([ref])

    def compute_classification_metrics(self) -> Dict[str, float]:
        """Compute classification metrics"""
        if len(self.predictions) == 0:
            return {}

        preds = np.array(self.predictions)
        labs = np.array(self.labels)

        # Basic metrics
        accuracy = accuracy_score(labs, preds)
        balanced_acc = balanced_accuracy_score(labs, preds)
        mcc = matthews_corrcoef(labs, preds)
        kappa = cohen_kappa_score(labs, preds)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labs, preds, average=None, zero_division=0
        )

        # Macro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labs, preds, average="macro", zero_division=0
        )

        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = (
            precision_recall_fscore_support(
                labs, preds, average="weighted", zero_division=0
            )
        )

        metrics = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "mcc": mcc,
            "kappa": kappa,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
        }

        # Per-class metrics
        for i in range(len(precision)):
            metrics[f"class_{i}_precision"] = precision[i]
            metrics[f"class_{i}_recall"] = recall[i]
            metrics[f"class_{i}_f1"] = f1[i]
            metrics[f"class_{i}_support"] = int(support[i])

        # Minority class metrics (last class typically minority in imbalanced data)
        if len(precision) > 1:
            metrics["minority_precision"] = precision[-1]
            metrics["minority_recall"] = recall[-1]
            metrics["minority_f1"] = f1[-1]

        # Probability-based metrics
        if len(self.probabilities) > 0:
            probs = np.array(self.probabilities)
            try:
                if probs.shape[1] == 2:
                    metrics["roc_auc"] = roc_auc_score(labs, probs[:, 1])
                    metrics["pr_auc"] = average_precision_score(labs, probs[:, 1])
                else:
                    metrics["roc_auc"] = roc_auc_score(
                        labs, probs, multi_class="ovr", average="macro"
                    )
            except Exception:
                pass

        return metrics

    def compute_generation_metrics(self) -> Dict[str, float]:
        """Compute rationale generation metrics"""
        if len(self.generated_rationales) == 0:
            return {}

        try:
            # BLEU score
            bleu_result = self.bleu_metric.compute(
                predictions=self.generated_rationales,
                references=self.reference_rationales,
            )
            bleu_score = bleu_result.get("bleu", bleu_result.get("score", 0.0))

            # ROUGE scores
            rouge_result = self.rouge_metric.compute(
                predictions=self.generated_rationales,
                references=[ref[0] for ref in self.reference_rationales],
            )

            return {
                "bleu": bleu_score,
                "rouge1": rouge_result.get("rouge1", 0.0),
                "rouge2": rouge_result.get("rouge2", 0.0),
                "rougeL": rouge_result.get(
                    "rougeL", rouge_result.get("rougeLsum", 0.0)
                ),
            }
        except Exception as e:
            print(f"Warning: Error computing generation metrics: {e}")
            return {}

    def compute_loss_metrics(self) -> Dict[str, float]:
        """Compute average losses"""
        loss_metrics = {}
        for key, values in self.losses.items():
            if values:
                loss_metrics[f"{key}_loss"] = np.mean(values)
        return loss_metrics

    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all metrics"""
        metrics = {}
        metrics.update(self.compute_loss_metrics())
        metrics.update(self.compute_classification_metrics())
        metrics.update(self.compute_generation_metrics())
        return metrics

    def get_classification_report(self) -> str:
        """Get detailed classification report"""
        if len(self.predictions) == 0:
            return "No predictions available"

        return classification_report(self.labels, self.predictions, zero_division=0)
