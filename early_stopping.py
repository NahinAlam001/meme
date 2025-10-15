"""
Early stopping implementation.
"""

import numpy as np


class EarlyStopping:
    """Early stopping handler to stop training when monitored metric stops improving"""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        # Set comparison operator based on mode
        if mode == "min":
            self.is_better = lambda new, best: new < best - min_delta
            self.best_score = np.inf
        else:
            self.is_better = lambda new, best: new > best + min_delta
            self.best_score = -np.inf
    
    def __call__(self, metric: float, epoch: int) -> bool:
        """Check if should stop training"""
        if self.is_better(metric, self.best_score):
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"✅ Metric improved to {metric:.4f} at epoch {epoch+1}")
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠️  No improvement for {self.counter}/{self.patience} epochs "
                      f"(best: {self.best_score:.4f} at epoch {self.best_epoch+1})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"🛑 Early stopping triggered! Best metric: "
                          f"{self.best_score:.4f} at epoch {self.best_epoch+1}")
                return True
            return False
    
    def state_dict(self):
        """Return state for checkpointing"""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'best_epoch': self.best_epoch
        }
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint"""
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']
        self.best_epoch = state_dict['best_epoch']
