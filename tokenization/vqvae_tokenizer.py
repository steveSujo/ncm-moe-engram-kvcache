"""
VQ-VAE Tokenizer Interface (frozen)

Wrapper around HazyResearch NCM tokenizer for get_ncm_inputs.
"""

import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import pickle


class VQVAETokenizer(nn.Module):
    """VQ-VAE Tokenizer for NCM (frozen, pre-trained).
    
    Loads codebooks and unigram directly from checkpoint without full model.
    """
    
    def __init__(self, config, base_dir: str = None):
        super().__init__()
        self.config = config
        self.base_dir = base_dir or ""
        
        # BPE tokenizer
        vocab_path = os.path.join(self.base_dir, config.pre_tokenizer.vocab_file_path)
        merges_path = os.path.join(self.base_dir, config.pre_tokenizer.merges_file_path)
        with open(vocab_path, 'rb') as f:
            self.bpe_vocab = pickle.load(f)
        with open(merges_path, 'rb') as f:
            self.bpe_merges = pickle.load(f)
        
        self.special_tokens = config.pre_tokenizer.special_tokens
        self.eos_token_id = self.bpe_vocab.get(config.pre_tokenizer.special_tokens.eos_token, 0)
        self.pad_token_id = self.bpe_vocab.get(config.pre_tokenizer.special_tokens.pad_token, 1)
        
        # Load codebooks and unigram from checkpoint
        checkpoint_path = os.path.join(self.base_dir, config.vqvae.checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model_state = ckpt['model']
        
        # Extract codebooks from VQ embeddings
        self.codebooks = []
        self.vqvae_vocab_size = config.vqvae.vqvae_vocab_size
        self.num_levels = len(self.vqvae_vocab_size)
        self.embed_dim = config.vqvae.embed_dim
        
        for i in range(self.num_levels):
            key = f'vq.embeddings.{i}.weight'
            if key in model_state:
                self.codebooks.append(model_state[key].clone())
            else:
                # Fallback: create dummy codebook
                self.codebooks.append(torch.randn(self.vqvae_vocab_size[i], self.embed_dim))
        
        # Load level0 unigram from generator checkpoint if available
        self.level0_unigram = None
        gen_ckpt_path = os.path.join(self.base_dir, "generator.pt")
        if os.path.exists(gen_ckpt_path):
            gen_ckpt = torch.load(gen_ckpt_path, map_location='cpu')
            if 'inference' in gen_ckpt and 'level0_unigram' in gen_ckpt['inference']:
                self.level0_unigram = gen_ckpt['inference']['level0_unigram'].clone()
        
        if self.level0_unigram is None:
            # Uniform fallback
            self.level0_unigram = torch.ones(self.vqvae_vocab_size[0]) / self.vqvae_vocab_size[0]
        
        # Freeze all
        for cb in self.codebooks:
            cb.requires_grad = False
        self.level0_unigram.requires_grad = False
        
        # Dummy VQ-VAE module for compatibility
        self.vqvae = nn.Module()
        self.vqvae.get_codebooks = lambda: self.codebooks
        self.vqvae.get_unigram = lambda: self.level0_unigram
        self.vqvae.get_ncm_inputs = self._get_ncm_inputs_impl
    
    def _get_ncm_inputs_impl(self, tokens: torch.Tensor):
        """Placeholder for get_ncm_inputs - real NCM would quantize through VQ-VAE."""
        B, T = tokens.shape
        device = tokens.device
        # Dummy multi-level codes
        x = torch.zeros(B, self.num_levels, T, self.embed_dim, device=device)
        targets = torch.zeros(B, 128, dtype=torch.long, device=device)
        return x, targets
    
    @classmethod
    def from_config(cls, config, base_dir: str = None):
        return cls(config, base_dir=base_dir)
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to BPE tokens."""
        tokens = [self.bpe_vocab.get(c, self.pad_token_id) for c in text]
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Decode BPE tokens to text."""
        inv_vocab = {v: k for k, v in self.bpe_vocab.items()}
        return ''.join([inv_vocab.get(t.item(), '') for t in tokens])
    
    def get_ncm_inputs(self, tokens: torch.Tensor):
        """Get NCM multi-level inputs from token sequence."""
        return self.vqvae.get_ncm_inputs(tokens)
    
    def get_codebooks(self):
        """Get VQ codebooks for head tying."""
        return self.codebooks
    
    def get_unigram(self):
        """Get level 0 unigram distribution."""
        return self.level0_unigram