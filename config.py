"""
Complete configuration for RGDORA training system.
Supports both Task 1 (binary) and Task 2 (multi-class) classification.
"""

import os
from dataclasses import dataclass

@dataclass
class RGDORAConfig:
    """Complete configuration for RGDORA training"""
    
    # ==================== TASK SELECTION ====================
    task: str = "task1"  # "task1" (binary) or "task2" (multi-class)
    
    # Data paths
    data_dir: str = "BHM"
    memes_dir: str = "BHM/Memes"
    files_dir: str = "BHM/Files"
    
    # Task-specific file names (will be set automatically)
    train_file: str = "train_task1.xlsx"
    val_file: str = "valid_task1.xlsx"
    test_file: str = "test_task1.xlsx"
    
    # Model architecture
    clip_model_name: str = "openai/clip-vit-base-patch32"
    xglm_model_name: str = "facebook/xglm-564M"
    bart_model_name: str = "facebook/bart-base"
    d_model: int = 512
    num_classes: int = 2  # Will be set based on task
    attn_heads: int = 8
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Training hyperparameters
    batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    generator_lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 20
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Loss weights
    w_ce: float = 1.0
    w_rgcl: float = 0.3
    w_rationale: float = 0.1
    temperature: float = 0.07
    
    # Generator settings
    neg_gen_hidden_dim: int = 512
    neg_gen_k: int = 5
    
    # FAISS settings
    faiss_rebuild_freq: int = 100
    faiss_top_k: int = 10
    
    # ==================== IMBALANCE HANDLING ====================
    # Loss function choice
    loss_type: str = "focal"  # "focal", "weighted_ce", "class_balanced"
    
    # Focal loss parameters
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Class-balanced loss
    cb_beta: float = 0.9999
    
    # Sampling strategy
    use_balanced_sampler: bool = True
    sampler_type: str = "balanced_batch"  # "balanced_batch" or "weighted_random"
    sampler_alpha: float = 0.8
    
    # Asymmetric contrastive loss
    acl_tau_plus: float = 0.1
    acl_beta: float = 1.0
    acl_gamma_pos: float = 0.5
    acl_gamma_neg: float = 1.5
    
    # Threshold adjustment
    optimize_threshold: bool = True
    
    # Class weighting method
    class_weight_method: str = "effective"  # "balanced", "sqrt", "effective"
    
    # ==================== EARLY STOPPING ====================
    patience: int = 5
    min_delta: float = 0.001
    monitor_metric: str = "val_macro_f1"
    mode: str = "max"
    
    # ==================== CHECKPOINTING ====================
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = False
    save_every_n_epochs: int = 1
    
    # ==================== LOGGING ====================
    log_dir: str = "./logs"
    log_every_n_steps: int = 10
    
    # ==================== DATA PROCESSING ====================
    max_seq_length: int = 128
    num_workers: int = 2
    
    # ==================== REPRODUCIBILITY ====================
    seed: int = 42
    
    # ==================== DEVICE ====================
    device: str = "cuda"
    mixed_precision: bool = True
    
    # ==================== RATIONALE GENERATION ====================
    rationale_max_length: int = 128
    rationale_num_beams: int = 4
    
    def __post_init__(self):
        """Set task-specific parameters"""
        # Set file names based on task
        if self.task == "task1":
            self.train_file = "train_task1.xlsx"
            self.val_file = "valid_task1.xlsx"
            self.test_file = "test_task1.xlsx"
            self.num_classes = 2
            self.focal_alpha = 0.25
            self.sampler_alpha = 0.7
            self.monitor_metric = "val_macro_f1"
        elif self.task == "task2":
            self.train_file = "train_task2.xlsx"
            self.val_file = "valid_task2.xlsx"
            self.test_file = "test_task2.xlsx"
            self.num_classes = 4
            self.focal_alpha = 0.5
            self.sampler_alpha = 0.9
            self.monitor_metric = "val_macro_f1"
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        # Update checkpoint and log directories with task name
        self.checkpoint_dir = f"./checkpoints_{self.task}"
        self.log_dir = f"./logs_{self.task}"
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"CONFIGURATION: {self.task.upper()}")
        print(f"{'='*80}")
        print(f"Num classes: {self.num_classes}")
        print(f"Training file: {self.train_file}")
        print(f"Loss type: {self.loss_type}")
        print(f"Focal alpha: {self.focal_alpha}")
        print(f"Sampler alpha: {self.sampler_alpha}")
        print(f"Monitor metric: {self.monitor_metric}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"{'='*80}\n")
