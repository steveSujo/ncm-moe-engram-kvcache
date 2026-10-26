"""
Common NCM Model Components

Recreated from HazyResearch NCM checkpoint structure:
- Block: Pre-LN transformer block with SelfAttention + MLP
- SelfAttention: Multi-head attention with QK norm, grouped-query
- MLP: Standard feedforward (replaced by MoEBlock in our model)
- RMSNorm: Root Mean Square Layer Normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm: x * weight / sqrt(mean(x^2) + eps) + bias
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x / rms * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x.to(dtype)


class SelfAttention(nn.Module):
    """
    Multi-head Self-Attention with QK Norm and Grouped-Query Attention.
    
    From NCM checkpoint:
    - c_attn: [3 * n_embd, n_embd] = [2304, 768] for q,k,v
    - c_proj: [n_embd, n_embd] = [768, 768]
    - q_norm, k_norm: [head_dim] = [64]
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        # Grouped-query: use standard MHA for compatibility with pretrained weights
        self.n_kv_heads = getattr(config, 'n_kv_heads', config.n_head)
        self.kv_dim = self.n_kv_heads * self.head_dim
        
        # QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # QK Norm
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.context_length, config.context_length))
            .view(1, 1, config.context_length, config.context_length),
            persistent=False
        )
        
        # RoPE
        self.use_rope = getattr(config, 'use_rope', True)
        self.rope_base = getattr(config, 'rope_base', 100000)
        if self.use_rope:
            cos, sin = self._build_rope_cache(config.context_length)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
    
    def _build_rope_cache(self, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build RoPE cache (cos, sin) for max_len."""
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()
    
    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to x [B, H, T, D]."""
        x1, x2 = x.chunk(2, dim=-1)
        x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_cache: Optional = None,
        layer_idx: int = 0,
        engram_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        
        # QKV projection
        qkv = self.c_attn(x)  # [B, T, 3 * C]
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, n_head, T, head_dim]
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, n_kv_heads, T, head_dim]
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, n_kv_heads, T, head_dim]
        
        # QK Norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # RoPE
        if self.use_rope and rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            cos = cos[:T]
            sin = sin[:T]
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)
        
        # KV Cache update
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)
            # Ensure dtype consistency
            k = k.to(q.dtype)
            v = v.to(q.dtype)
        # Engram KV fusion (add to k, v)
        if engram_kv is not None:
            engram_k, engram_v = engram_kv
            # engram_k/v: [B, T, n_embd] -> reshape
            engram_k = engram_k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            engram_v = engram_v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            
            # Add (or concatenate) - using gated addition
            k = k + engram_k
            v = v + engram_v
        
        # Attention
        # Grouped-query: repeat k, v for each query head group
        if self.n_kv_heads != self.n_head:
            repeat = self.n_head // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        
        # Flash attention or manual
        scale = torch.tensor(1.0 / math.sqrt(self.head_dim), dtype=q.dtype, device=q.device)
        att = (q @ k.transpose(-2, -1)) * scale
        
        # Causal mask
        if attn_mask is not None:
            att = att.masked_fill(~attn_mask, float('-inf'))
        else:
            # Use causal mask - handle KV cache extended length
            kv_len = k.size(2)  # Total KV length including cache
            if kv_len <= self.bias.size(-1):
                # Standard case
                att = att.masked_fill(self.bias[:, :, :T, :kv_len] == 0, float('-inf'))
            else:
                # KV cache extended beyond pre-computed bias
                # Create dynamic mask
                causal_mask = torch.tril(torch.ones(T, kv_len, dtype=torch.bool, device=att.device))
                att = att.masked_fill(~causal_mask[None, None, :, :], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v  # [B, n_head, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class MLP(nn.Module):
    """
    Standard Feedforward Network (replaced by MoEBlock).
    
    From NCM checkpoint:
    - c_fc: [3072, 768] (up projection)
    - c_proj: [768, 3072] (down projection)
    - Uses GELU activation
    """
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer Block with Pre-LN.
    
    Structure:
    x = x + attn(ln_1(x))
    x = x + mlp(ln_2(x))
    """
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_cache: Optional = None,
        layer_idx: int = 0,
        engram_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attn(
            self.ln_1(x),
            attn_mask=attn_mask,
            rope_cos_sin=rope_cos_sin,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            engram_kv=engram_kv,
        )
        
        # MLP with residual
        x = x + self.mlp(self.ln_2(x))
        
        return x


class NextConceptTransformer(nn.Module):
    """
    Base NCM Generator (parallel coarse-to-fine).
    
    This is the original architecture without MoE/Engram/KV-cache enhancements.
    """
    
    def __init__(self, vqvae_local, config):
        super().__init__()
        self.vqvae_local = vqvae_local
        self.config = config
        
        # Model params
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        self.context_length = config.context_length
        self.prefix_len = config.prefix_len
        self.tie_heads_to_codebook = config.tie_heads_to_codebook
        
        # Embeddings
        vocab_0 = config.vocab_size[0] if isinstance(config.vocab_size, list) else config.vocab_size
        self.wte = nn.Embedding(vocab_0, config.n_embd)
        self.wle = nn.Embedding(len(config.vocab_size), config.n_embd) if config.use_level_encoding else None
        self.wpe = nn.Embedding(config.context_length, config.n_embd) if config.use_positional_encoding else None
        
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        
        # Output heads
        self.lm_heads = nn.ModuleList([
            nn.Linear(config.n_embd, vs, bias=False) for vs in config.vocab_size
        ])
        
        self.apply(self._init_weights)
        
        if self.tie_heads_to_codebook:
            self._tie_heads_to_codebook()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _tie_heads_to_codebook(self):
        codebooks = self.vqvae_local.get_codebooks()
        for level, head in enumerate(self.lm_heads):
            if level < len(codebooks):
                head.weight.data = codebooks[level].to(head.weight.dtype).clone()
    
    def forward(self, idx, level=0, targets=None):
        B, T = idx.shape
        device = idx.device
        
        tok_emb = self.wte(idx)
        if self.wle is not None:
            lvl_emb = self.wle(torch.full((B, T), level, dtype=torch.long, device=device))
            tok_emb = tok_emb + lvl_emb
        if self.wpe is not None:
            pos = torch.arange(T, device=device).unsqueeze(0)
            pos_emb = self.wpe(pos)
            tok_emb = tok_emb + pos_emb
        
        x = self.drop(tok_emb)
        
        for i, block in enumerate(self.blocks):
            x = block(x, layer_idx=i)
        
        x = self.ln_f(x)
        logits = self.lm_heads[level](x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def generate(self, idx, l0=None, temperature=1.0, top_k=0, top_p=1.0, cfg_scale=0.0):
        """Autoregressive generation (simplified)."""
        self.eval()
        # ... generation logic ...
        return idx


if __name__ == "__main__":
    # Quick test
    from omegaconf import OmegaConf
    config = OmegaConf.create({
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 768,
        'vocab_size': [4096] * 11,
        'context_length': 129,
        'prefix_len': 1,
        'bias': True,
        'dropout': 0.1,
        'use_level_encoding': True,
        'use_positional_encoding': True,
        'use_rope': True,
        'rope_base': 100000,
        'tie_heads_to_codebook': True,
    })
    
    block = Block(config)
    x = torch.randn(2, 128, 768)
    out = block(x)
    print(f"Block: {x.shape} -> {out.shape}")
    
    attn = SelfAttention(config)
    out = attn(x)
    print(f"Attention: {x.shape} -> {out.shape}")
    
    mlp = MLP(config)
    out = mlp(x)
    print(f"MLP: {x.shape} -> {out.shape}")
    
    print("Common components test passed!")