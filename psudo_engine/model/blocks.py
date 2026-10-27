"""
Model Building Blocks for psudo-Engine.

Core transformer components: RMSNorm, Multi-Head Attention, Feed-Forward,
and Transformer Block with support for ternary quantization and paging.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..weight_paging.ternary import TernaryLinear

if TYPE_CHECKING:
    from .kv_cache import KVCache


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Compute RMS
        norm = x.float().pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(norm + self.eps)

        if self.elementwise_affine:
            x_norm = x_norm * self.weight.to(x.dtype)

        return x_norm.to(x.dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


def precompute_rope_cos_sin(
    max_seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE cos/sin tables.

    Returns:
        cos: [max_seq_len, head_dim]
        sin: [max_seq_len, head_dim]
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create inverse frequencies for all dimensions (not just half)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)

    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)

    return cos, sin


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary positional embeddings.

    Args:
        x: [batch, seq_len, num_heads, head_dim] or [batch, num_heads, seq_len, head_dim]
        cos: [seq_len, head_dim]
        sin: [seq_len, head_dim]
    """
    # Handle both [B, T, H, D] and [B, H, T, D] formats
    if x.dim() == 4 and x.shape[1] != cos.shape[0]:
        # Assume [B, H, T, D]
        x = x.transpose(1, 2)
        transposed = True
    else:
        transposed = False

    # x: [B, T, H, D]
    # cos/sin: [T, D]
    batch, seq_len, num_heads, head_dim = x.shape

    # Reshape cos/sin for broadcasting
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, T, 1, D]
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, T, 1, D]

    # Split into pairs
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    # Apply rotation
    cos = cos[..., 0::2]
    sin = sin[..., 0::2]

    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    if transposed:
        rotated = rotated.transpose(1, 2)

    return rotated


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with optional ternary quantization."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.1,
        use_ternary: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim or (d_model // n_heads)
        self.dropout = dropout
        self.use_ternary = use_ternary

        self.scale = self.head_dim ** -0.5

        # Combined QKV projection
        if use_ternary:
            self.qkv_proj = TernaryLinear(
                d_model, 3 * n_heads * self.head_dim, bias=False, device=device, dtype=dtype
            )
            self.out_proj = TernaryLinear(
                n_heads * self.head_dim, d_model, bias=False, device=device, dtype=dtype
            )
        else:
            self.qkv_proj = nn.Linear(
                d_model, 3 * n_heads * self.head_dim, bias=False, device=device, dtype=dtype
            )
            self.out_proj = nn.Linear(
                n_heads * self.head_dim, d_model, bias=False, device=device, dtype=dtype
            )

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # For inference mode
        self.inference_mode = False

    def prepare_for_inference(self):
        """Prepare for inference by quantizing weights."""
        self.inference_mode = True
        if self.use_ternary:
            self.qkv_proj.prepare_for_inference()
            self.out_proj.prepare_for_inference()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: "KVCache | None" = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch, seq_len, d_model]
            cos: [seq_len, head_dim]
            sin: [seq_len, head_dim]
            mask: Optional attention mask [batch, 1, seq_len, seq_len]
            kv_cache: Optional KV cache for incremental decoding

        Returns:
            Output: [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)

        # Split Q, K, V
        qkv = qkv.reshape(batch, seq_len, self.n_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [B, H, T, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)

        # Apply RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Handle KV cache
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # Scaled dot-product attention
        # q: [B, H, T, D], k: [B, H, T_kv, D], v: [B, H, T_kv, D]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # [B, H, T, D]

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, T, H, D]
        attn_output = attn_output.reshape(batch, seq_len, self.n_heads * self.head_dim)

        # Output projection
        output = self.out_proj(attn_output)

        output = self.resid_dropout(output)

        return output

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_heads={self.n_heads}, head_dim={self.head_dim}"


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        use_ternary: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.dropout = dropout
        self.use_ternary = use_ternary

        if use_ternary:
            self.w1 = TernaryLinear(d_model, self.d_ff, bias=False, device=device, dtype=dtype)
            self.w2 = TernaryLinear(self.d_ff, d_model, bias=False, device=device, dtype=dtype)
            self.w3 = TernaryLinear(d_model, self.d_ff, bias=False, device=device, dtype=dtype)
        else:
            self.w1 = nn.Linear(d_model, self.d_ff, bias=False, device=device, dtype=dtype)
            self.w2 = nn.Linear(self.d_ff, d_model, bias=False, device=device, dtype=dtype)
            self.w3 = nn.Linear(d_model, self.d_ff, bias=False, device=device, dtype=dtype)

        self.dropout_layer = nn.Dropout(dropout)
        self.inference_mode = False

    def prepare_for_inference(self):
        """Prepare for inference."""
        self.inference_mode = True
        if self.use_ternary:
            self.w1.prepare_for_inference()
            self.w2.prepare_for_inference()
            self.w3.prepare_for_inference()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: w2(silu(w1(x)) * w3(x))"""
        w1_out = self.w1(x)
        w3_out = self.w3(x)
        hidden = F.silu(w1_out) * w3_out
        output = self.w2(hidden)

        output = self.dropout_layer(output)
        return output

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}"


class TransformerBlock(nn.Module):
    """
    Transformer Block with Pre-Norm architecture.

    Supports:
    - Ternary quantized weights (BitNet 1.58b)
    - Engram memory injection
    - KV Cache for incremental decoding
    - Gradient checkpointing
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int | None = None,
        d_ff: int | None = None,
        dropout: float = 0.1,
        use_ternary: bool = False,
        has_engram: bool = False,
        engram_dim: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim or (d_model // n_heads)
        self.use_ternary = use_ternary
        self.has_engram = has_engram
        self.engram_dim = engram_dim or d_model

        # Layer norms
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        # Attention
        self.attn = MultiHeadAttention(
            d_model, n_heads, self.head_dim, dropout, use_ternary, device, dtype
        )

        # Feed-forward
        self.ffn = SwiGLU(
            d_model, d_ff, dropout, use_ternary, device, dtype
        )

        # Engram projection (if enabled)
        if has_engram:
            if use_ternary:
                self.engram_proj = TernaryLinear(
                    self.engram_dim, d_model, bias=False, device=device, dtype=dtype
                )
            else:
                self.engram_proj = nn.Linear(
                    self.engram_dim, d_model, bias=False, device=device, dtype=dtype
                )

        self.dropout = nn.Dropout(dropout)
        self.inference_mode = False

    def prepare_for_inference(self):
        """Prepare for inference."""
        self.inference_mode = True
        self.attn.prepare_for_inference()
        self.ffn.prepare_for_inference()
        if self.has_engram:
            self.engram_proj.prepare_for_inference()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: "KVCache | None" = None,
        engram: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch, seq_len, d_model]
            cos: [seq_len, head_dim]
            sin: [seq_len, head_dim]
            mask: Optional attention mask
            kv_cache: Optional KV cache
            engram: Optional engram memory [batch, seq_len, engram_dim]
        """
        # Attention block
        residual = x
        x = self.ln1(x)
        x = self.attn(x, cos, sin, mask, kv_cache)
        x = residual + self.dropout(x)

        # Engram injection (additive)
        if self.has_engram and engram is not None:
            engram_proj = self.engram_proj(engram)
            x = x + engram_proj

        # FFN block
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)

        return x

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_heads={self.n_heads}, has_engram={self.has_engram}"


class TransformerBlockCheckpointed(TransformerBlock):
    """TransformerBlock with gradient checkpointing support."""

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: "KVCache | None" = None,
        engram: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward with gradient checkpointing."""
        if self.training and not self.inference_mode:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                x, cos, sin, mask, kv_cache, engram,
                use_reentrant=False
            )
        return self._forward_impl(x, cos, sin, mask, kv_cache, engram)

    def _forward_impl(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None,
        kv_cache: "KVCache | None",
        engram: torch.Tensor | None
    ) -> torch.Tensor:
        """Actual forward implementation."""
        return super().forward(x, cos, sin, mask, kv_cache, engram)


if __name__ == "__main__":
    # Quick test

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16

    # Test RMSNorm
    norm = RMSNorm(512, device=device, dtype=dtype)
    x = torch.randn(2, 128, 512, device=device, dtype=dtype)
    out = norm(x)

    # Test RoPE
    cos, sin = precompute_rope_cos_sin(2048, 64, device=device, dtype=dtype)

    # Test Attention
    attn = MultiHeadAttention(512, 8, 64, use_ternary=True, device=device, dtype=dtype)
    out = attn(x, cos, sin)

    # Test SwiGLU
    ffn = SwiGLU(512, 2048, use_ternary=True, device=device, dtype=dtype)
    out = ffn(x)

    # Test Block
    block = TransformerBlock(512, 8, 64, 2048, use_ternary=True, has_engram=True, device=device, dtype=dtype)
    engram = torch.randn(2, 128, 512, device=device, dtype=dtype)
    out = block(x, cos, sin, engram=engram)

    # Test inference mode
    block.prepare_for_inference()
    out = block(x, cos, sin, engram=engram)

