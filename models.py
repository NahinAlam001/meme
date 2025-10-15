"""
RGDORA model architecture with negative generator.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, AutoModelForCausalLM, BartForConditionalGeneration
from peft import get_peft_model, LoraConfig
from typing import Optional, Tuple


class NegativeGenerator(nn.Module):
    """Adversarial negative generator for contrastive learning"""
    
    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 512, k: int = 5):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim * k)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, positive_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positive_embeddings: [batch_size, embedding_dim]
        Returns:
            [batch_size, k, embedding_dim]
        """
        emb = self.net(positive_embeddings)
        return emb.view(positive_embeddings.size(0), self.k, positive_embeddings.size(1))


class RGDORA(nn.Module):
    """Retrieval-Guided DOmain Robust Architecture"""
    
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        xglm_model_name: str = "facebook/xglm-564M",
        bart_model_name: str = "facebook/bart-base",
        d_model: int = 512,
        num_classes: int = 2,
        use_lora: bool = True,
        lora_config: dict = None
    ):
        super().__init__()
        
        # Visual encoder
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.v_dim = getattr(
            self.clip.config, 'projection_dim',
            getattr(self.clip.config, 'hidden_size', 512)
        )
        
        # Text encoder
        self.xglm = AutoModelForCausalLM.from_pretrained(xglm_model_name)
        self.t_dim = self.xglm.config.hidden_size
        
        # Apply LoRA
        if use_lora:
            if lora_config is None:
                lora_config = {'r': 16, 'lora_alpha': 32, 'lora_dropout': 0.1}
            
            lora_cfg = LoraConfig(
                r=lora_config['r'],
                lora_alpha=lora_config['lora_alpha'],
                lora_dropout=lora_config.get('lora_dropout', 0.1),
                target_modules=["q_proj", "v_proj"],
                bias="none"
            )
            
            try:
                self.clip = get_peft_model(self.clip, lora_cfg)
                self.xglm = get_peft_model(self.xglm, lora_cfg)
                print("✅ LoRA applied successfully")
            except Exception as e:
                print(f"⚠️  LoRA application failed: {e}. Continuing without LoRA.")
        
        # Projections
        self.d_model = d_model
        self.proj_v = nn.Linear(self.v_dim, d_model)
        self.proj_t = nn.Linear(self.t_dim, d_model)
        
        nn.init.xavier_uniform_(self.proj_v.weight)
        nn.init.xavier_uniform_(self.proj_t.weight)
        
        # Fusion and classification
        self.fc_fusion = nn.Sequential(
            nn.Linear(d_model * d_model, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(256, num_classes)
        
        # Rationale decoder
        self.rationale_decoder = BartForConditionalGeneration.from_pretrained(bart_model_name)
        self.rationale_tokenizer = None  # Set externally
        self.encoder_proj = nn.Linear(256, self.rationale_decoder.config.d_model)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(256)
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        positives: Optional[torch.Tensor] = None,
        negatives: Optional[torch.Tensor] = None,
        rationale_ids: Optional[torch.Tensor] = None,
        rationale_mask: Optional[torch.Tensor] = None,
        temperature: float = 0.07
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
            logits, fused_repr, rgcl_loss, rationale_loss
        """
        # Visual features
        v_feats = self.clip.get_image_features(pixel_values=images)
        
        # Text features
        x_out = self.xglm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        if hasattr(x_out, "hidden_states") and x_out.hidden_states is not None:
            t_last = x_out.hidden_states[-1].mean(dim=1)
        else:
            t_last = x_out.logits.mean(dim=1)
        
        # Project
        v_proj = self.proj_v(v_feats)
        t_proj = self.proj_t(t_last)
        
        # Feature Interaction Matrix
        v_proj_reshaped = v_proj.unsqueeze(2)
        t_proj_reshaped = t_proj.unsqueeze(1)
        fim = torch.bmm(v_proj_reshaped, t_proj_reshaped)
        fused = fim.view(images.size(0), -1)
        
        # Fusion
        fused_repr = self.fc_fusion(fused)
        fused_repr = self.layer_norm(fused_repr)
        
        # Classification
        logits = self.classifier(fused_repr)
        
        # RGCL loss
        rgcl_loss = None
        if positives is not None and negatives is not None:
            fused_norm = nn.functional.normalize(fused_repr, dim=1)
            pos_norm = nn.functional.normalize(positives, dim=1)
            pos_sim = (fused_norm * pos_norm).sum(dim=1) / temperature
            
            if negatives.dim() == 3:
                neg_norm = nn.functional.normalize(negatives, dim=2)
                neg_sims = torch.bmm(neg_norm, fused_norm.unsqueeze(2)).squeeze(2) / temperature
            else:
                neg_norm = nn.functional.normalize(negatives, dim=1)
                neg_sims = (fused_norm @ neg_norm.T) / temperature
            
            numerator = torch.exp(pos_sim)
            denominator = numerator + torch.exp(neg_sims).sum(dim=1) + 1e-8
            rgcl_loss = -torch.log(numerator / denominator).mean()
        
        # Rationale loss
        rationale_loss = None
        if rationale_ids is not None and self.rationale_tokenizer is not None:
            labels = rationale_ids.clone()
            labels[labels == self.rationale_tokenizer.pad_token_id] = -100
            
            seq_lens = rationale_mask.sum(dim=1)
            ignore_mask = seq_lens <= 2
            labels[ignore_mask] = -100
            
            valid_rationales = (labels != -100).any()
            
            if valid_rationales:
                inputs_embeds = self.encoder_proj(fused_repr).unsqueeze(1)
                encoder_attention_mask = torch.ones(inputs_embeds.size(0), 1, device=inputs_embeds.device)
                
                try:
                    outputs = self.rationale_decoder(
                        inputs_embeds=inputs_embeds,
                        attention_mask=encoder_attention_mask,
                        labels=labels,
                        decoder_attention_mask=rationale_mask
                    )
                    
                    rationale_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=labels.device)
                    
                    if torch.isnan(rationale_loss):
                        rationale_loss = torch.tensor(0.0, device=labels.device)
                except Exception as e:
                    rationale_loss = torch.tensor(0.0, device=labels.device)
            else:
                rationale_loss = torch.tensor(0.0, device=labels.device)
        
        return logits, fused_repr, rgcl_loss, rationale_loss
    
    def get_fused_embedding(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract fused embeddings for FAISS indexing"""
        self.eval()
        with torch.no_grad():
            _, fused_repr, _, _ = self.forward(images, input_ids, attention_mask)
        return fused_repr
