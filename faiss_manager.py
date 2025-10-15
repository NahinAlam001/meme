"""
FAISS index management for retrieval-guided contrastive learning.
"""

import torch
import torch.nn as nn
import faiss
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader


class FAISSIndexManager:
    """Manages FAISS index for retrieval-guided contrastive learning"""
    
    def __init__(self, embedding_dim: int, rebuild_freq: int = 100):
        self.embedding_dim = embedding_dim
        self.rebuild_freq = rebuild_freq
        self.index = None
        self.db_embeddings = None
        self.db_labels = None
        self.step_counter = 0
    
    def build_index(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
    ):
        """Build FAISS index from dataset embeddings"""
        model.eval()
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                images, input_ids, attention_mask, labels, _, _, _ = batch
                images = images.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                
                fused = model.get_fused_embedding(images, input_ids, attention_mask)
                all_embeddings.append(fused.cpu().numpy())
                all_labels.extend(labels.numpy().tolist())
        
        self.db_embeddings = np.vstack(all_embeddings).astype(np.float32)
        self.db_labels = np.array(all_labels)
        
        # Normalize and build index
        faiss.normalize_L2(self.db_embeddings)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.db_embeddings)
        
        model.train()
        print(f"✅ FAISS index built with {len(self.db_labels)} samples")
    
    def retrieve_pairs(
        self,
        model: nn.Module,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        k: int = 10,
        device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve positive and negative pairs"""
        if self.index is None:
            # Return dummy tensors if index not built
            batch_size = images.size(0)
            pos = torch.zeros(batch_size, self.embedding_dim, device=device)
            neg = torch.zeros(batch_size, 5, self.embedding_dim, device=device)
            return pos, neg
        
        # Get query embeddings
        with torch.no_grad():
            query_embeddings = model.get_fused_embedding(images, input_ids, attention_mask)
            q = query_embeddings.cpu().numpy().astype(np.float32)
            faiss.normalize_L2(q)
        
        # Search
        D, I = self.index.search(q, k)
        
        # Extract positives and negatives
        pos_list = []
        neg_list = []
        
        for i in range(len(labels)):
            query_label = labels[i].item()
            retrieved_indices = I[i].tolist()
            
            # Find positive (same class)
            pos_idx = None
            for idx in retrieved_indices:
                if 0 <= idx < len(self.db_labels) and self.db_labels[idx] == query_label:
                    pos_idx = idx
                    break
            
            if pos_idx is None:
                pos_list.append(np.zeros(self.embedding_dim, dtype=np.float32))
            else:
                pos_list.append(self.db_embeddings[pos_idx])
            
            # Find negatives (different class)
            neg_indices = [
                idx for idx in retrieved_indices
                if 0 <= idx < len(self.db_labels) and self.db_labels[idx] != query_label
            ]
            
            if len(neg_indices) == 0:
                neg_indices = np.random.choice(len(self.db_labels), size=5, replace=True)
            elif len(neg_indices) < 5:
                extra = 5 - len(neg_indices)
                neg_indices += np.random.choice(len(self.db_labels), size=extra, replace=True).tolist()
            else:
                neg_indices = neg_indices[:5]
            
            neg_list.append(self.db_embeddings[neg_indices])
        
        pos_embeddings = torch.tensor(np.vstack(pos_list), dtype=torch.float32, device=device)
        neg_embeddings = torch.tensor(np.stack(neg_list), dtype=torch.float32, device=device)
        
        return pos_embeddings, neg_embeddings
