"""
VQ-VAE Tokenizer Interface (frozen)

Wrapper around HazyResearch NCM tokenizer for get_ncm_inputs.
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import pickle


class VQVAETokenizer(nn.Module):
    """VQ-VAE Tokenizer for NCM (frozen, pre-trained)."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # BPE tokenizer
        with open(config.pre_tokenizer.vocab_file_path, 'rb') as f:
            self.bpe_vocab = pickle.load(f)
        with open(config.pre_tokenizer.merges_file_path, 'rb') as f:
            self.bpe_merges = pickle.load(f)
        
        self.special_tokens = config.pre_tokenizer.special_tokens
        self.eos_token_id = self.bpe_vocab.get(config.pre_tokenizer.special_tokens.eos_token, 0)
        self.pad_token_id = self.bpe_vocab.get(config.pre_tokenizer.special_tokens.pad_token, 1)
        
        # VQ-VAE
        from models.vqvae import VQVAE
        self.vqvae = VQVAE(config.vqvae)
        
        # Load checkpoint
        checkpoint_path = config.vqvae.checkpoint_path
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        self.vqvae.load_state_dict(ckpt['model'])
        
        # Freeze
        self.vqvae.eval()
        for p in self.vqvae.parameters():
            p.requires_grad = False
    
    @classmethod
    def from_config(cls, config):
        return cls(config)
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to BPE tokens."""
        # Simple BPE encoding (placeholder)
        # Real implementation would use the BPE merges
        tokens = [self.bpe_vocab.get(c, self.pad_token_id) for c in text]
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Decode BPE tokens to text."""
        inv_vocab = {v: k for k, v in self.bpe_vocab.items()}
        return ''.join([inv_vocab.get(t.item(), '') for t in tokens])
    
    def get_ncm_inputs(self, tokens: torch.Tensor):
        """
        Get NCM multi-level inputs from token sequence.
        
        Args:
            tokens: [B, T] BPE token IDs
        Returns:
            x: [B, L, D] multi-level discrete codes
            targets: [B, 128] flattened targets
        """
        with torch.no_grad():
            return self.vqvae.get_ncm_inputs(tokens)
    
    def get_codebooks(self):
        """Get VQ codebooks for head tying."""
        return self.vqvae.get_codebooks()
    
    def get_unigram(self):
        """Get level 0 unigram distribution."""
        return self.vqvae.get_unigram()


# Minimal VQVAE implementation for compatibility
class VQVAE(nn.Module):
    """Minimal VQ-VAE for NCM tokenization."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Embedding(config.in_vocab_size, config.embed_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.embed_dim,
                    nhead=config.embed_dim // 64,
                    dim_feedforward=config.embed_dim * 4,
                    dropout=config.dropout,
                    batch_first=True,
                ),
                num_layers=config.n_layers,
            ),
        )
        
        # Vector Quantizer (simplified)
        self.vq = nn.ModuleList([
            nn.Embedding(vs, config.embed_dim) for vs in config.vqvae_vocab_size
        ])
        
        # Decoder (not needed for encoding)
        
    def get_codebooks(self):
        return [vq.weight.data for vq in self.vq]
    
    def get_unigram(self):
        # Uniform for now
        return torch.ones(self.config.vqvae_vocab_size[0]) / self.config.vqvae_vocab_size[0]
    
    def get_ncm_inputs(self, tokens: torch.Tensor):
        """
        Convert BPE tokens to multi-level VQ codes.
        This is a simplified version - real implementation uses the VQ-VAE encoder.
        """
        B, T = tokens.shape
        
        # For now, just replicate tokens across levels (placeholder)
        # Real implementation: encode through VQ-VAE
        levels = len(self.config.vqvae_vocab_size)
        
        # Create dummy multi-level codes
        x = tokens.unsqueeze(-1).repeat(1, 1, levels)  # [B, T, L]
        
        # Flatten targets for loss
        targets = tokens[:, :128].contiguous()  # First 128 tokens
        
        return x, targets


if __name__ == "__main__":
    # Test tokenizer loading
    config = OmegaConf.load('artifacts/tinystories/config.yaml')
    tokenizer = VQVAETokenizer.from_config(config.tokenizer)
    print("Tokenizer loaded!")
    
    # Test encode
    tokens = tokenizer.encode("Once upon a time")
    print(f"Encoded: {tokens.shape}")
    
    # Test get_ncm_inputs
    x, targets = tokenizer.get_ncm_inputs(tokens.unsqueeze(0))
    print(f"NCM inputs: x={x.shape}, targets={targets.shape}")
    
    print("Tokenizer test passed!")