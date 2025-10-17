"""
Comprehensive checkpoint management.
"""

import os
import torch
from typing import Dict, Optional, Any
from datetime import datetime
import json


class CheckpointManager:
    """Manages model checkpoints with full training state persistence"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_best_only: bool = False,
        max_checkpoints: int = 3
    ):
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.max_checkpoints = max_checkpoints
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_history = []
        
    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        model: torch.nn.Module,
        generator: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        optimizer_g: torch.optim.Optimizer,
        scheduler: Optional[Any],
        early_stopping: Any,
        metrics: Dict[str, float],
        is_best: bool = False,
        additional_state: Optional[Dict] = None
    ) -> str:
        """Save complete training state"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'model_state_dict': model.state_dict(),
            'generator_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'early_stopping_state': early_stopping.state_dict(),
            'metrics': metrics,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'additional_state': additional_state or {}
        }
        
        # Determine checkpoint filename
        if is_best:
            checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            print(f"💾 Saving BEST checkpoint (epoch {epoch+1}, step {step})")
        else:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f"checkpoint_epoch{epoch+1}_step{step}.pt"
            )
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Also save latest for easy resumption
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        torch.save(checkpoint, latest_path)
        
        # Manage checkpoint history
        if not is_best:
            self.checkpoint_history.append(checkpoint_path)
            self._cleanup_old_checkpoints()
        
        # Save metrics log
        self._log_metrics(epoch, step, metrics)
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        generator: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        optimizer_g: torch.optim.Optimizer,
        scheduler: Optional[Any],
        early_stopping: Any,
        device: torch.device,
        load_optimizer_state: bool = True
    ) -> Dict[str, Any]:
        """Load complete training state from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"📥 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load model states
        model.load_state_dict(checkpoint['model_state_dict'])
        generator.load_state_dict(checkpoint['generator_state_dict'])
        
        # Load optimizer states (only for training resumption)
        if load_optimizer_state:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            
            # Load scheduler state
            if scheduler and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load early stopping state
            early_stopping.load_state_dict(checkpoint['early_stopping_state'])
            
            # Restore RNG states for reproducibility
            torch.set_rng_state(checkpoint['rng_state'])
            if checkpoint['cuda_rng_state'] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        
        metadata = {
            'epoch': checkpoint['epoch'],
            'step': checkpoint['step'],
            'metrics': checkpoint['metrics'],
            'timestamp': checkpoint.get('timestamp', 'unknown')
        }
        
        print(f"✅ Loaded checkpoint from epoch {metadata['epoch']+1}, step {metadata['step']}")
        
        return metadata
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only max_checkpoints recent ones"""
        if len(self.checkpoint_history) > self.max_checkpoints:
            old_checkpoints = self.checkpoint_history[:-self.max_checkpoints]
            for ckpt_path in old_checkpoints:
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
            self.checkpoint_history = self.checkpoint_history[-self.max_checkpoints:]
    
    def _log_metrics(self, epoch: int, step: int, metrics: Dict[str, float]):
        """Append metrics to JSON log file"""
        log_path = os.path.join(self.checkpoint_dir, "training_log.json")
        
        log_entry = {
            'epoch': epoch + 1,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        # Append to log file
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = []
        
        log_data.append(log_entry)
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file"""
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        if os.path.exists(latest_path):
            return latest_path
        return None
    
    def find_best_checkpoint(self) -> Optional[str]:
        """Find the best checkpoint file"""
        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            return best_path
        return None
