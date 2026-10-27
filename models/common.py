"""
Common NCM Model Components

Recreated from HazyResearch NCM checkpoint structure:
- Block: Pre-LN transformer block with SelfAttention + MLP
- SelfAttention: Multi-head attention with QK norm, grouped-query
- MLP: Standard feedforward (replaced by MoEBlock in our model)
- RMSNorm: Root Mean Square Layer Normalization
"""

import math
from typing import Optional, tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = x.to(torch.float32)
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

        # QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # QK Norm
        self.q_norm = nn.Parameter(torch.ones(self.n_head, self.head_dim))
        self.k_norm = nn.Parameter(torch.ones(self.n_head, self.head_dim))

        # RoPE
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _build_rope_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Build RoPE cache for given sequence length."""
        if self.rope_cos is not None and self.rope_cos.shape[1] >= seq_len:
            return

        theta = 10000.0 ** (-torch.arange(0, self.head_dim, 2, device=device, dtype=dtype) / self.head_dim)
        pos = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(pos, theta)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # Expand for heads: [seq_len, head_dim//2] -> [1, seq_len, n_head, head_dim//2]
        cos = cos.unsqueeze(0).unsqueeze(2).expand(1, seq_len, self.n_head, self.head_dim // 2)
        sin = sin.unsqueeze(0).unsqueeze(2).expand(1, seq_len, self.n_head, self.head_dim // 2)

        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to x [B, H, T, D]."""
        x1, x2 = x.chunk(2, dim=-1)
        x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache: object | None = None,
        layer_idx: int = 0,
        engram_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # QK Norm
        q = q * self.q_norm.unsqueeze(0).unsqueeze(2)
        k = k * self.k_norm.unsqueeze(0).unsqueeze(2)

        # Apply RoPE
        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
        else:
            self._build_rope_cache(T, x.device, x.dtype)
            cos, sin = self.rope_cos[:, :T], self.rope_sin[:, :T]

        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # Handle Engram KV (prepend engram keys/values)
        if engram_kv is not None:
            engram_k, engram_v = engram_kv
            k = torch.cat([engram_k, k], dim=2)
            v = torch.cat([engram_v, v], dim=2)

        # Handle KV Cache
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, layer_idx)

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        if attn_mask is not None:
            att = att.masked_fill(attn_mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        # Output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y


class MLP(nn.Module):
    """
    Standard Feedforward Network (replaced by MoEBlock in our model).
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
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
        rope_cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        kv_cache: Optional[object] = None,
        layer_idx: int = 0,
        engram_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
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
    Recreated from HazyResearch NCM checkpoint structure.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        self.context_length = config.context_length
        self.prefix_len = config.prefix_len

        # Embeddings
        vocab_size_0 = int(config.vocab_size[0]) if hasattr(config.vocab_size, "__getitem__") else int(config.vocab_size)
        self.wte = nn.Embedding(vocab_size_0, config.n_embd)
        self.wle = nn.Embedding(len(config.vocab_size), config.n_embd) if config.use_level_encoding else None
        self.wpe = nn.Embedding(config.context_length, config.n_embd) if config.use_positional_encoding else None

        # Dropout
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.ln_f = RMSNorm(config.n_embd, bias=config.bias)

        # Output heads
        self.lm_heads = nn.ModuleList([nn.Linear(config.n_embd, vs, bias=False) for vs in config.vocab_size])

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        device = x.device

        tok_emb = self.wte(x)

        if self.wle is not None:
            lvl_emb = self.wle(torch.arange(T, device=device).unsqueeze(0).expand(B, -1))
            tok_emb = tok_emb + lvl_emb

        if self.wpe is not None:
            pos = torch.arange(T, device=device).unsqueeze(0)
            pos_emb = self.wpe(pos)
            tok_emb = tok_emb + pos_emb

        x = self.drop(tok_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = torch.cat([head(x) for head in self.lm_heads], dim=-1)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


if __name__ == "__main__":
    # Quick test
    from omegaconf import OmegaConf

    config = OmegaConf.load("artifacts/tinystories/config.yaml")
    model = NextConceptTransformer(config.model)
    # Model created silently

    # Test forward
    B, T = 2, 128
    x = torch.randint(0, 4096, (B, T))
    model.eval()
    with torch.no_grad():
        logits, loss = model(x, targets=torch.randint(0, 4096, (B, T)))

    # Logits shape and loss printed for debugging