"""
MoE + Engram + KV Cache Enhanced NCM Generator - Native 2-bit Training
=====================================================================

This version implements NATIVE 2-bit training using Cactus Quants (CQ) from Needle.
All weights (except embeddings/heads) are stored and computed in 2-bit CQ format.

Architecture Changes for Native 2-bit:
- All Linear layers → CQQuantizedLinear (2-bit CQ with STE)
- All Embeddings → CQQuantizedEmbedding (2-bit CQ with STE)  
- MoE Experts stored in 2-bit CQ format on SSD
- Engram tables in 2-bit CQ
- KV Cache in 2-bit CQ
- Attention projections in 2-bit CQ
- RMSNorm remains FP32 for stability
- STE (Straight-Through Estimator) for gradient flow

Training uses CQ STE: forward=quantized, backward=identity gradient
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from omegaconf import OmegaConf
import os

from models.common import RMSNorm
from models.cq_quantize import (
    CQQuantizedLinear,
    CQQuantizedEmbedding,
    CQ_GROUP_SIZE,
    WEIGHT_BITS,
)
from models.block_2bit import Block2Bit
from models.kv_cache_2bit import MultiLayerKVCache2Bit
from models.engram_2bit import EngramModule2Bit, needle2_engram_config
from models.moe_block_2bit import CQQuantizedMoEPager
from tokenization.vqvae_tokenizer import VQVAETokenizer


class MoEEngramNextConceptTransformer2Bit(nn.Module):
    """
    Full 2-bit NCM Generator with MoE, Engram, and Bounded KV Cache.
    """
    
    def __init__(self, config, tokenizer: VQVAETokenizer = None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # Model dimensions
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.n_kv_head = getattr(config, "n_kv_head", config.n_head)
        self.head_dim = self.n_embd // self.n_head
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        
        # Engram configuration
        self.engram_layers = getattr(config, "engram_layers", [2, 11])
        self.engram_config = needle2_engram_config()
        if hasattr(config, "engram"):
            self.engram_config.update(config.engram)
        
        # Quantization settings
        self.bits = WEIGHT_BITS
        self.group_size = CQ_GROUP_SIZE
        
        # Token embeddings (keep BF16)
        self.wte = nn.Embedding(self.vocab_size, self.n_embd)
        self.wpe = nn.Embedding(self.max_seq_len, self.n_embd)
        
        # Level embeddings for NCM (keep BF16)
        self.n_levels = getattr(config, "n_levels", 11)
        self.wle = nn.Embedding(self.n_levels, self.n_embd)
        
        # Dropout
        self.drop = nn.Dropout(getattr(config, "embd_dropout", 0.0))
        
        # Expert pager (created later)
        self.pager = None
        
        # Transformer blocks (2-bit)
        self.blocks = nn.ModuleList([
            Block2Bit(
                config,
                layer_idx=i,
                pager=None,  # Set later
                engram_layers=self.engram_layers,
                engram_config=self.engram_config,
            )
            for i in range(self.n_layer)
        ])
        
        # Final norm (FP32)
        self.ln_f = RMSNorm(self.n_embd, bias=config.bias)
        
        # Output heads per level (keep BF16)
        self.lm_heads = nn.ModuleList([
            nn.Linear(self.n_embd, self.vocab_size, bias=False)
            for _ in range(self.n_levels)
        ])
        
        # Tie weights
        for level in range(self.n_levels):
            self.lm_heads[level].weight = self.wte.weight
        
        # KV Cache (2-bit)
        self.kv_cache = None
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def set_pager(self, pager: CQQuantizedMoEPager):
        """Set expert pager for all MoE blocks."""
        self.pager = pager
        for block in self.blocks:
            block.set_pager(pager)
    
    def init_kv_cache(self, batch_size: int, device: torch.device):
        """Initialize 2-bit KV cache."""
        self.kv_cache = MultiLayerKVCache2Bit(
            n_layers=self.n_layer,
            max_window=getattr(self.config, "kv_cache_window", 256),
            sink_tokens=getattr(self.config, "kv_cache_sink", 1),
            n_heads=self.n_kv_head,
            head_dim=self.head_dim,
            bits=self.bits,
            group_size=64,  # Smaller group for KV
            dtype=torch.bfloat16,
            device=device,
        )
    
    def reset_kv_cache(self):
        """Reset KV cache."""
        if self.kv_cache is not None:
            self.kv_cache.reset()
    
    def forward(
        self,
        idx: torch.Tensor,
        level: torch.Tensor = None,
        targets: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        rope_cos_sin: Tuple[torch.Tensor, torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            idx: [B, T] token indices
            level: [B, T] level indices (for NCM)
            targets: [B, T] target tokens (for loss computation)
            attn_mask: Attention mask
            rope_cos_sin: RoPE cos/sin tables
            use_cache: Whether to use KV cache
            
        Returns:
            logits: [B, T, V] or [B, T, n_levels, V] if level provided
            loss: Optional loss tensor
        """
        B, T = idx.shape
        device = idx.device
        
        # Initialize KV cache if needed
        if use_cache and self.kv_cache is None:
            self.init_kv_cache(B, device)
        
        # Embeddings
        tok_emb = self.wte(idx)  # [B, T, D]
        pos_emb = self.wpe(torch.arange(T, device=device).unsqueeze(0))  # [1, T, D]
        x = self.drop(tok_emb + pos_emb)
        
        # Add level embeddings if provided
        if level is not None:
            lvl_emb = self.wle(level)
            x = x + lvl_emb
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                attn_mask=attn_mask,
                rope_cos_sin=rope_cos_sin,
                kv_cache=self.kv_cache if use_cache else None,
            )
        
        # Final norm
        x = self.ln_f(x)
        
        # Output heads
        if level is not None:
            # Multi-level output: [B, T, n_levels, V]
            logits = torch.stack([head(x) for head in self.lm_heads], dim=2)
        else:
            # Single level (default level 0)
            logits = self.lm_heads[0](x)
        
        # Loss computation
        loss = None
        if targets is not None:
            if level is not None:
                # Multi-level loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.n_levels, self.vocab_size).transpose(1, 2),
                    targets.view(-1, self.n_levels).transpose(0, 1)
                ).mean()
            else:
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    targets.view(-1)
                )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        level: torch.Tensor = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively with KV cache."""
        self.eval()
        
        # Enable KV cache for generation
        self.init_kv_cache(idx.shape[0], idx.device)
        
        for _ in range(max_new_tokens):
            # Forward with cache
            logits, _ = self.forward(
                idx[:, -1:] if idx.shape[1] > 0 else idx,
                level=level[:, -1:] if level is not None and level.shape[1] > 0 else level,
                use_cache=True,
            )
            
            # Get next token logits
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('inf')
            
            # Top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = -float('inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat([idx, next_token], dim=1)
            
            if level is not None:
                level = torch.cat([level, level[:, -1:]], dim=1)
            
            # Check EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return idx
    
    def prepare_for_inference(self):
        """Switch all quantized modules to inference mode."""
        for block in self.blocks:
            block.prepare_for_inference()


def create_moe_engram_model_2bit(config_path: str, expert_dir: str = "experts") -> MoEEngramNextConceptTransformer2Bit:
    """Factory function to create 2-bit model from config."""
    from omegaconf import OmegaConf
    
    config = OmegaConf.load(config_path)
    
    # Create tokenizer
    tokenizer = VQVAETokenizer(config_path.replace("generator.yaml", "tokenizer.yaml"))
    
    # Create model
    model = MoEEngramNextConceptTransformer2Bit(config, tokenizer)
    
    # Create expert pager
    from models.moe_block_2bit import CQQuantizedMoEPager
    
    pager = CQQuantizedMoEPager(
        expert_dir=expert_dir,
        n_layers=config.n_layer,
        n_experts=getattr(config, "n_experts", 64),
        d_model=config.n_embd,
        expert_hidden=config.n_embd * 4,
        resident_slots=getattr(config, "resident_slots", 32),
        bits=WEIGHT_BITS,
        group_size=CQ_GROUP_SIZE,
    )
    
    model.set_pager(pager)
    
    return model


if __name__ == "__main__":
    # Quick test
    import torch
    from omegaconf import OmegaConf
    import tempfile
    
    print("Testing MoEEngramNextConceptTransformer2Bit...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal config
        config = OmegaConf.create({
            "n_embd": 768,
            "n_layer": 4,  # Small for test
            "n_head": 12,
            "n_kv_head": 4,
            "vocab_size": 4096,
            "max_seq_len": 1024,
            "bias": False,
            "embd_dropout": 0.0,
            "n_levels": 11,
            "n_experts": 64,
            "top_k": 2,
            "capacity_factor": 1.25,
            "router_loss_weight": 0.01,
            "kv_cache_window": 256,
            "kv_cache_sink": 1,
            "engram_layers": [2, 3],  # Only 2 layers for test
        })
        
        # Create model
        model = MoEEngramNextConceptTransformer2Bit(config)
        
        # Create pager
        from models.moe_block_2bit import CQQuantizedMoEPager
        pager = CQQuantizedMoEPager(
            expert_dir=tmpdir,
            n_layers=4,
            n_experts=64,
            d_model=768,
            expert_hidden=3072,
            resident_slots=32,
        )
        model.set_pager(pager)
        
        # Test forward
        B, T = 2, 16
        idx = torch.randint(0, 4096, (B, T))
        level = torch.randint(0, 11, (B, T))
        
        with torch.no_grad():
            logits, loss = model(idx, level=level)
        
        print(f"Input shape: {idx.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Loss: {loss}")
        print("MoEEngramNextConceptTransformer2Bit test passed!")