"""
Complete RGDORA trainer with full resumability, early stopping, and evaluation.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional
import random

from transformers import (
    CLIPProcessor,
    AutoTokenizer,
    BartTokenizer,
    get_linear_schedule_with_warmup,
)

# Import custom modules
from config import RGDORAConfig
from dataset import BanglaMemeDataset, load_excel_to_lists
from models import RGDORA, NegativeGenerator
from metrics import MetricsTracker
from early_stopping import EarlyStopping
from checkpoint_manager import CheckpointManager
from faiss_manager import FAISSIndexManager
from imbalance_handler import (
    FocalLoss,
    ClassBalancedLoss,
    BalancedBatchSampler,
    ImbalancedDatasetSampler,
    analyze_class_distribution,
    compute_class_weights,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RGDORATrainer:
    """Complete training system for RGDORA with resumability and early stopping"""

    def __init__(self, config: RGDORAConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # Set seed
        set_seed(config.seed)

        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_training()
        self._setup_logging()

        print(f"🚀 Trainer initialized on {self.device}")
        print(f"   Training samples: {len(self.train_dataset)}")
        print(f"   Validation samples: {len(self.val_dataset)}")
        print(f"   Test samples: {len(self.test_dataset)}")

    def _setup_data(self):
        """Setup datasets and dataloaders"""
        print("📊 Setting up datasets...")

        # Load processors and tokenizers
        self.clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model_name)
        self.xglm_tokenizer = AutoTokenizer.from_pretrained(
            self.config.xglm_model_name, use_fast=True
        )
        self.rationale_tokenizer = BartTokenizer.from_pretrained(
            self.config.bart_model_name
        )

        if self.xglm_tokenizer.pad_token is None:
            self.xglm_tokenizer.pad_token = self.xglm_tokenizer.eos_token

        # Load data files
        def get_file_path(filename):
            path = os.path.join(self.config.files_dir, filename)
            if not os.path.exists(path):
                path = os.path.join(self.config.data_dir, filename)
            return path

        train_path = get_file_path(self.config.train_file)
        val_path = get_file_path(self.config.val_file)
        test_path = get_file_path(self.config.test_file)

        # Load datasets
        train_imgs, train_texts, train_labels, train_rationales = load_excel_to_lists(
            train_path, self.config.memes_dir, task=self.config.task
        )
        val_imgs, val_texts, val_labels, val_rationales = load_excel_to_lists(
            val_path, self.config.memes_dir, task=self.config.task
        )
        test_imgs, test_texts, test_labels, test_rationales = load_excel_to_lists(
            test_path, self.config.memes_dir, task=self.config.task
        )

        # Create datasets
        self.train_dataset = BanglaMemeDataset(
            train_imgs,
            train_texts,
            train_labels,
            train_rationales,
            self.clip_processor,
            self.xglm_tokenizer,
            self.rationale_tokenizer,
            max_length=self.config.max_seq_length,
            task=self.config.task,
        )
        self.val_dataset = BanglaMemeDataset(
            val_imgs,
            val_texts,
            val_labels,
            val_rationales,
            self.clip_processor,
            self.xglm_tokenizer,
            self.rationale_tokenizer,
            max_length=self.config.max_seq_length,
            task=self.config.task,
        )
        self.test_dataset = BanglaMemeDataset(
            test_imgs,
            test_texts,
            test_labels,
            test_rationales,
            self.clip_processor,
            self.xglm_tokenizer,
            self.rationale_tokenizer,
            max_length=self.config.max_seq_length,
            task=self.config.task,
        )

        # Create dataloaders (will be recreated with balanced sampler if needed)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    def _setup_model(self):
        """Initialize models"""
        print("🏗️  Building models...")

        # Main model
        self.model = RGDORA(
            clip_model_name=self.config.clip_model_name,
            xglm_model_name=self.config.xglm_model_name,
            bart_model_name=self.config.bart_model_name,
            d_model=self.config.d_model,
            num_classes=self.config.num_classes,
            use_lora=self.config.use_lora,
            lora_config={
                "r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
            },
        ).to(self.device)

        # Set rationale tokenizer
        self.model.rationale_tokenizer = self.rationale_tokenizer

        # Negative generator
        self.generator = NegativeGenerator(
            embedding_dim=256,
            hidden_dim=self.config.neg_gen_hidden_dim,
            k=self.config.neg_gen_k,
        ).to(self.device)

        print(
            f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )
        print(
            f"   Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )

    def _setup_training(self):
        """Setup optimizers, schedulers, and training components"""
        print("⚙️  Setting up training components...")

        # Analyze class distribution
        train_labels = [
            self.train_dataset.labels[i] for i in range(len(self.train_dataset))
        ]
        dist_stats = analyze_class_distribution(
            train_labels, f"Training Set ({self.config.task})"
        )

        # Compute class weights
        self.class_weights = compute_class_weights(
            train_labels, method=self.config.class_weight_method
        ).to(self.device)

        print(f"📊 Class weights: {self.class_weights.cpu().numpy()}")

        # Choose loss function based on config
        if self.config.loss_type == "focal":
            self.ce_loss = FocalLoss(
                alpha=self.config.focal_alpha, gamma=self.config.focal_gamma
            )
            print(
                f"✅ Using Focal Loss (α={self.config.focal_alpha}, γ={self.config.focal_gamma})"
            )

        elif self.config.loss_type == "class_balanced":
            samples_per_class = [
                dist_stats["class_counts"].get(i, 0)
                for i in range(self.config.num_classes)
            ]
            self.ce_loss = ClassBalancedLoss(
                samples_per_class=samples_per_class,
                beta=self.config.cb_beta,
                loss_type="focal",
            )
            print(f"✅ Using Class-Balanced Focal Loss (β={self.config.cb_beta})")

        elif self.config.loss_type == "weighted_ce":
            self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
            print(f"✅ Using Weighted Cross-Entropy")

        else:
            self.ce_loss = nn.CrossEntropyLoss()
            print(f"✅ Using Standard Cross-Entropy")

        # Setup balanced sampler if enabled
        if self.config.use_balanced_sampler:
            if self.config.sampler_type == "balanced_batch":
                batch_sampler = BalancedBatchSampler(
                    labels=train_labels,
                    batch_size=self.config.batch_size,
                    alpha=self.config.sampler_alpha,
                )
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_sampler=batch_sampler,
                    num_workers=self.config.num_workers,
                    pin_memory=True if torch.cuda.is_available() else False,
                )
                print(
                    f"✅ Using Balanced Batch Sampler (α={self.config.sampler_alpha})"
                )

            elif self.config.sampler_type == "weighted_random":
                sampler = ImbalancedDatasetSampler(labels=train_labels)
                self.train_loader = DataLoader(
                    self.train_dataset,
                    sampler=sampler,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers,
                    pin_memory=True if torch.cuda.is_available() else False,
                )
                print(f"✅ Using Weighted Random Sampler")

        # Optimizers
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.optimizer_g = optim.AdamW(
            self.generator.parameters(),
            lr=self.config.generator_lr,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.config.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        # Mixed precision training
        self.scaler = (
            GradScaler()
            if self.config.mixed_precision and torch.cuda.is_available()
            else None
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mode=self.config.mode,
            verbose=True,
        )

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            save_best_only=self.config.save_best_only,
            max_checkpoints=3,
        )

        # FAISS index manager
        self.faiss_manager = FAISSIndexManager(
            embedding_dim=256, rebuild_freq=self.config.faiss_rebuild_freq
        )

        # Metrics trackers
        self.train_metrics = MetricsTracker(num_classes=self.config.num_classes)
        self.val_metrics = MetricsTracker(num_classes=self.config.num_classes)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = -np.inf if self.config.mode == "max" else np.inf
        self.optimal_threshold = 0.5

    def _setup_logging(self):
        """Setup logging directory"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.config.log_dir, "training.log")

    def resume_from_checkpoint(self, checkpoint_path: Optional[str] = None):
        """Resume training from checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_manager.find_latest_checkpoint()

        if checkpoint_path is None:
            print("ℹ️  No checkpoint found, starting from scratch")
            return

        metadata = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            generator=self.generator,
            optimizer=self.optimizer,
            optimizer_g=self.optimizer_g,
            scheduler=self.scheduler,
            early_stopping=self.early_stopping,
            device=self.device,
            load_optimizer_state=True,
        )

        self.current_epoch = metadata["epoch"] + 1
        self.global_step = metadata["step"]
        self.best_metric = self.early_stopping.best_score

        print(f"🔄 Resumed from epoch {self.current_epoch}, step {self.global_step}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.generator.train()
        self.train_metrics.reset()

        # Build/rebuild FAISS index if needed
        if epoch == 0 or self.global_step % self.config.faiss_rebuild_freq == 0:
            self.faiss_manager.build_index(self.model, self.train_loader, self.device)

        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            (
                images,
                input_ids,
                attention_mask,
                labels,
                rationale_ids,
                rationale_mask,
                _,
            ) = batch
            images = images.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            rationale_ids = rationale_ids.to(self.device)
            rationale_mask = rationale_mask.to(self.device)

            # Retrieve positives and negatives
            pos_embs, neg_embs = self.faiss_manager.retrieve_pairs(
                self.model,
                images,
                input_ids,
                attention_mask,
                labels,
                k=self.config.faiss_top_k,
                device=self.device,
            )

            # Generate adversarial negatives
            with torch.no_grad():
                adv_neg_embs = self.generator(pos_embs.detach())

            # Combine negatives
            negatives = torch.cat((neg_embs, adv_neg_embs), dim=1)

            # --- Train main model ---
            self.optimizer.zero_grad()

            if self.scaler:
                with torch.amp.autocast("cuda"):
                    logits, fused_repr, rgcl_loss, rationale_loss = self.model(
                        images,
                        input_ids,
                        attention_mask,
                        positives=pos_embs,
                        negatives=negatives,
                        rationale_ids=rationale_ids,
                        rationale_mask=rationale_mask,
                        temperature=self.config.temperature,
                    )
                    ce_loss = self.ce_loss(logits, labels)

                    # Compute total loss
                    total_loss = (
                        self.config.w_ce * ce_loss
                        + self.config.w_rgcl
                        * (rgcl_loss if rgcl_loss is not None else 0.0)
                        + self.config.w_rationale
                        * (rationale_loss if rationale_loss is not None else 0.0)
                    )

                # Check for NaN
                if torch.isnan(total_loss):
                    print(
                        f"⚠️  NaN detected in total loss at step {self.global_step}, skipping batch"
                    )
                    continue

                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, fused_repr, rgcl_loss, rationale_loss = self.model(
                    images,
                    input_ids,
                    attention_mask,
                    positives=pos_embs,
                    negatives=negatives,
                    rationale_ids=rationale_ids,
                    rationale_mask=rationale_mask,
                    temperature=self.config.temperature,
                )
                ce_loss = self.ce_loss(logits, labels)

                total_loss = (
                    self.config.w_ce * ce_loss
                    + self.config.w_rgcl * (rgcl_loss if rgcl_loss is not None else 0.0)
                    + self.config.w_rationale
                    * (rationale_loss if rationale_loss is not None else 0.0)
                )

                if torch.isnan(total_loss):
                    print(f"⚠️  NaN detected, skipping batch")
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

            self.scheduler.step()

            # --- Train generator ---
            self.optimizer_g.zero_grad()

            adv_neg_embs_for_g = self.generator(pos_embs.detach())

            # Compute generator loss
            with torch.no_grad():
                fused_repr_detached = fused_repr.detach()

            fused_norm = nn.functional.normalize(fused_repr_detached, dim=1)
            pos_norm = nn.functional.normalize(pos_embs.detach(), dim=1)
            pos_sim = (fused_norm * pos_norm).sum(dim=1) / self.config.temperature

            neg_norm = nn.functional.normalize(adv_neg_embs_for_g, dim=2)
            neg_sims = (
                torch.bmm(neg_norm, fused_norm.unsqueeze(2)).squeeze(2)
                / self.config.temperature
            )

            numerator = torch.exp(pos_sim)
            denominator = numerator + torch.exp(neg_sims).sum(dim=1) + 1e-8
            rgcl_loss_g = -torch.log(numerator / denominator).mean()

            generator_loss = -rgcl_loss_g

            if not torch.isnan(generator_loss):
                generator_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(), self.config.max_grad_norm
                )
                self.optimizer_g.step()

            # Update metrics
            predictions = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)
            loss_dict = {
                "total": total_loss.item(),
                "ce": ce_loss.item(),
                "rgcl": rgcl_loss.item() if rgcl_loss is not None else 0.0,
                "rationale": rationale_loss.item()
                if rationale_loss is not None
                else 0.0,
            }
            self.train_metrics.update(predictions, labels, loss_dict, probabilities)

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "ce": f"{ce_loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                }
            )

            self.global_step += 1

            # Rebuild FAISS index periodically
            if self.global_step % self.config.faiss_rebuild_freq == 0:
                print(f"\n🔄 Rebuilding FAISS index at step {self.global_step}")
                self.faiss_manager.build_index(
                    self.model, self.train_loader, self.device
                )

        # Compute epoch metrics
        epoch_metrics = self.train_metrics.compute_all_metrics()
        epoch_metrics = {f"train_{k}": v for k, v in epoch_metrics.items()}

        return epoch_metrics

    @torch.no_grad()
    def validate(self, loader: DataLoader, split_name: str = "val") -> Dict[str, float]:
        """Validate on given dataloader"""
        self.model.eval()
        metrics_tracker = MetricsTracker(num_classes=self.config.num_classes)

        progress_bar = tqdm(loader, desc=f"Validating ({split_name})")

        for batch in progress_bar:
            (
                images,
                input_ids,
                attention_mask,
                labels,
                rationale_ids,
                rationale_mask,
                original_rationales,
            ) = batch
            images = images.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            # Forward pass (no retrieval needed for validation)
            logits, fused_repr, _, _ = self.model(images, input_ids, attention_mask)

            # Compute loss
            ce_loss = self.ce_loss(logits, labels)

            # Get probabilities
            probabilities = torch.softmax(logits, dim=1)

            # Generate rationales
            generated_rationales = []
            if self.config.w_rationale > 0:
                inputs_embeds = self.model.encoder_proj(fused_repr).unsqueeze(1)
                encoder_attention_mask = torch.ones(
                    inputs_embeds.size(0), 1, device=self.device
                )

                generated_ids = self.model.rationale_decoder.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_attention_mask,
                    max_length=self.config.rationale_max_length,
                    num_beams=self.config.rationale_num_beams,
                    early_stopping=True,
                )

                generated_rationales = [
                    self.model.rationale_tokenizer.decode(
                        g,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                    for g in generated_ids
                ]

            # Update metrics
            predictions = torch.argmax(logits, dim=1)
            loss_dict = {"total": ce_loss.item(), "ce": ce_loss.item()}
            metrics_tracker.update(
                predictions,
                labels,
                loss_dict,
                probabilities,
                generated_rationales,
                original_rationales,
            )

        # Compute metrics
        metrics = metrics_tracker.compute_all_metrics()

        # Find optimal threshold on validation set
        if (
            split_name == "val"
            and self.config.optimize_threshold
            and len(metrics_tracker.probabilities) > 0
        ):
            from sklearn.metrics import f1_score

            probs = np.array(metrics_tracker.probabilities)
            labels_arr = np.array(metrics_tracker.labels)

            thresholds = np.arange(0.1, 0.9, 0.01)
            best_f1 = 0
            best_threshold = 0.5

            for threshold in thresholds:
                if self.config.num_classes == 2:
                    y_pred = (probs[:, 1] >= threshold).astype(int)
                    f1 = f1_score(labels_arr, y_pred)

                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold

            if self.config.num_classes == 2:
                self.optimal_threshold = best_threshold
                print(
                    f"📊 Optimal threshold: {self.optimal_threshold:.3f} (F1={best_f1:.4f})"
                )

        metrics = {f"{split_name}_{k}": v for k, v in metrics.items()}

        return metrics

    def train(self, resume: bool = True):
        """Main training loop with early stopping"""
        print("\n" + "=" * 80)
        print(f"🚀 Starting RGDORA Training - {self.config.task.upper()}")
        print("=" * 80)

        # Try to resume
        if resume:
            self.resume_from_checkpoint()

        # Training loop
        for epoch in range(self.current_epoch, self.config.epochs):
            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch + 1}/{self.config.epochs}")
            print(f"{'=' * 80}")

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(self.val_loader, split_name="val")

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}

            # Print metrics
            print(f"\n📊 Epoch {epoch + 1} Results:")
            print(f"   Train Loss: {train_metrics.get('train_total_loss', 0):.4f}")
            print(f"   Train Acc:  {train_metrics.get('train_accuracy', 0):.4f}")
            print(f"   Train F1:   {train_metrics.get('train_macro_f1', 0):.4f}")
            print(f"   Val Loss:   {val_metrics.get('val_total_loss', 0):.4f}")
            print(f"   Val Acc:    {val_metrics.get('val_accuracy', 0):.4f}")
            print(f"   Val F1:     {val_metrics.get('val_macro_f1', 0):.4f}")

            # Check if best model
            monitor_value = all_metrics.get(self.config.monitor_metric, 0.0)
            is_best = False
            if self.config.mode == "max":
                is_best = monitor_value > self.best_metric
            else:
                is_best = monitor_value < self.best_metric

            if is_best:
                self.best_metric = monitor_value
                print(f"✨ New best {self.config.monitor_metric}: {monitor_value:.4f}")

            # Save checkpoint
            if epoch % self.config.save_every_n_epochs == 0 or is_best:
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    step=self.global_step,
                    model=self.model,
                    generator=self.generator,
                    optimizer=self.optimizer,
                    optimizer_g=self.optimizer_g,
                    scheduler=self.scheduler,
                    early_stopping=self.early_stopping,
                    metrics=all_metrics,
                    is_best=is_best,
                )

            # Early stopping check
            if self.early_stopping(monitor_value, epoch):
                print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
                break

            self.current_epoch = epoch + 1

        print("\n" + "=" * 80)
        print("✅ Training completed!")
        print("=" * 80)

        # Load best model for evaluation
        best_ckpt = self.checkpoint_manager.find_best_checkpoint()
        if best_ckpt:
            print(f"\n📥 Loading best model for final evaluation...")
            self.checkpoint_manager.load_checkpoint(
                checkpoint_path=best_ckpt,
                model=self.model,
                generator=self.generator,
                optimizer=self.optimizer,
                optimizer_g=self.optimizer_g,
                scheduler=self.scheduler,
                early_stopping=self.early_stopping,
                device=self.device,
                load_optimizer_state=False,
            )

    def evaluate(self, split: str = "test") -> Dict[str, float]:
        """Comprehensive evaluation"""
        print(f"\n{'=' * 80}")
        print(f"📊 Evaluating on {split} set")
        print(f"{'=' * 80}")

        loader = self.test_loader if split == "test" else self.val_loader
        metrics = self.validate(loader, split_name=split)

        # Print results
        print(f"\n{split.upper()} RESULTS:")
        print(f"  Accuracy: {metrics.get(f'{split}_accuracy', 0):.4f}")
        print(
            f"  Balanced Accuracy: {metrics.get(f'{split}_balanced_accuracy', 0):.4f}"
        )
        print(f"  Macro F1: {metrics.get(f'{split}_macro_f1', 0):.4f}")
        print(f"  Weighted F1: {metrics.get(f'{split}_weighted_f1', 0):.4f}")
        print(f"  MCC: {metrics.get(f'{split}_mcc', 0):.4f}")

        # Per-class metrics
        print(f"\nPer-class metrics:")
        for i in range(self.config.num_classes):
            f1 = metrics.get(f"{split}_class_{i}_f1", 0)
            prec = metrics.get(f"{split}_class_{i}_precision", 0)
            rec = metrics.get(f"{split}_class_{i}_recall", 0)
            print(f"  Class {i}: F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")

        if f"{split}_bleu" in metrics:
            print(f"\nRATIONALE GENERATION:")
            print(f"  BLEU: {metrics.get(f'{split}_bleu', 0):.4f}")
            print(f"  ROUGE-L: {metrics.get(f'{split}_rougeL', 0):.4f}")

        # Detailed classification report
        self.model.eval()
        metrics_tracker = MetricsTracker(num_classes=self.config.num_classes)

        with torch.no_grad():
            for batch in loader:
                images, input_ids, attention_mask, labels, _, _, _ = batch
                images = images.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                logits, _, _, _ = self.model(images, input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=1)
                probabilities = torch.softmax(logits, dim=1)

                loss_dict = {"total": 0.0}
                metrics_tracker.update(predictions, labels, loss_dict, probabilities)

        print(f"\nDETAILED CLASSIFICATION REPORT:")
        print(metrics_tracker.get_classification_report())

        # Save metrics to file
        metrics_file = os.path.join(self.config.log_dir, f"{split}_metrics.json")
        import json

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n💾 Metrics saved to {metrics_file}")

        return metrics
