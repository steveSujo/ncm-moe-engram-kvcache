"""
VQ-VAE Implementation matching HazyResearch NCM tokenizer checkpoint.
Loads pre-trained weights from tokenizer.pt checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from typing import List, Optional, Tuple


class VQVAE(nn.Module):
    """VQ-VAE model matching HazyResearch checkpoint structure."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder (BPE tokenizer -> embeddings)
        self.embed_dim = config.embed_dim
        self.in_vocab_size = config.in_vocab_size
        self.context_length = config.context_length
        
        # Encoder embeddings (BPE)
        self.encoder = nn.Module()
        self.encoder.wte = nn.Embedding(self.in_vocab_size, self.embed_dim)
        self.encoder.wpe = nn.Embedding(self.context_length, self.embed_dim)
        
        # Multi-scale residual VQ
        self.vqvae_vocab_size = config.vqvae_vocab_size
        self.num_levels = len(self.vqvae_vocab_size)
        
        # VQ embeddings (codebooks)
        self.vq = nn.Module()
        self.vq.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, self.embed_dim) 
            for vocab_size in self.vqvae_vocab_size
        ])
        
        # Quantization residual connections
        self.vq.quant_resi = nn.ModuleList([
            nn.Conv1d(self.embed_dim, self.embed_dim, 3, padding=1, bias=True)
            for _ in range(self.num_levels)
        ])
        
        # Learned downsampling for level 1
        self.vq.learned_down_l1 = nn.ModuleList([
            nn.Conv1d(self.embed_dim, self.embed_dim, 4, padding=0, bias=True)
            for _ in range(3)  # 3 levels based on checkpoint
        ])
        
        # Level 0 unigram
        self.level0_unigram = None
    
    def get_codebooks(self) -> List[torch.Tensor]:
        """Get VQ codebooks for head tying."""
        return [emb.weight for emb in self.vq.embeddings]
    
    def get_unigram(self) -> torch.Tensor:
        """Get level 0 unigram distribution."""
        if self.level0_unigram is not None:
            return self.level0_unigram
        # Uniform fallback
        return torch.ones(self.vqvae_vocab_size[0]) / self.vqvae_vocab_size[0]
    
    def encode_bpe(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode BPE tokens through encoder embeddings."""
        B, T = tokens.shape
        device = tokens.device
        
        tok_emb = self.encoder.wte(tokens)
        pos = torch.arange(T, device=device).unsqueeze(0)
        pos_emb = self.encoder.wpe(pos)
        
        return tok_emb + pos_emb
    
    def get_ncm_inputs(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get NCM multi-level inputs from token sequence.
        In real NCM, this quantizes through VQ-VAE to get 11-level codes.
        For inference, we need the actual quantization logic.
        
        Returns: (x: [B, 11, T, D], targets: [B, 128])
        """
        # This is a simplified version - real NCM uses full VQ-VAE forward
        # For now, return encoded representation as placeholder
        z = self.encode_bpe(tokens)  # [B, T, D]
        B, T, D = z.shape
        
        # Dummy multi-level codes for inference
        # Real implementation would quantize through vq.embeddings
        x = z.unsqueeze(1).expand(-1, self.num_levels, -1, -1)  # [B, 11, T, D]
        targets = torch.zeros(B, 128, dtype=torch.long, device=z.device)
        
        return x, targets
    
    def forward(self, x):
        return self.get_ncm_inputs(x)


def load_vqvae_from_checkpoint(checkpoint_path: str, config_path: str = None, device: str = 'cpu'):
    """Load VQ-VAE from tokenizer.pt checkpoint."""
    if config_path:
        config = OmegaConf.load(config_path).tokenizer.vqvae
    else:
        config = OmegaConf.create({
            'n_layers': 6,
            'embed_dim': 384,
            'in_vocab_size': 8192,
            'context_length': 128,
            'compression_factor': 1,
            'dropout': 0.3,
            'pre_quant_groupnorm': 8,
            'use_qk_norm': True,
            'pre_quant_dropout': 0.0,
            'vqvae_vocab_size': [4096] * 11,
        })
    
    model = VQVAE(config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict (strict=False to handle missing decoder keys)
    model.load_state_dict(ckpt['model'], strict=False)
    
    # Load level0_unigram if available
    if 'inference' in ckpt and 'level0_unigram' in ckpt['inference']:
        model.level0_unigram = ckpt['inference']['level0_unigram'].to(device)
    
    model.eval()
    return model


if __name__ == "__main__":
    # Quick test
    import tempfile
    import os
    
    config = OmegaConf.create({
        'n_layers': 2,
        'embed_dim': 384,
        'in_vocab_size': 8192,
        'context_length': 128,
        'compression_factor': 1,
        'dropout': 0.3,
        'pre_quant_groupnorm': 8,
        'use_qk_norm': True,
        'pre_quant_dropout': 0.0,
        'vqvae_vocab_size': [4096] * 11,
    })
    
    model = VQVAE(config)
    print(f"VQ-VAE params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print("VQ-VAE test passed!")