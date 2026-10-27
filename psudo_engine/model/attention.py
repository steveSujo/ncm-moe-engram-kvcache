"""
Causal Attention with Bounded KV Cache for psudo-Engine.

Implements efficient causal attention with sliding window KV cache
and INT8 quantization for memory efficiency.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import apply_rotary_emb, precompute_rope_cos_sin


class KVCache:
    """
    Bounded KV Cache with sliding window and attention sinks.

    Features:
    - Sliding window of max_window tokens
    - Permanent sink tokens (first N tokens always kept)
    - INT8 quantization for memory efficiency
    - Efficient incremental updates
    """

    def __init__(
        self,
        max_window: int = 256,
        sink_tokens: int = 1,
        n_heads: int = 8,
        head_dim: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None,
        quantize_kv: bool = True
    ):
        self.max_window = max_window
        self.sink_tokens = sink_tokens
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.quantize_kv = quantize_kv

        # Cache storage
        self.k_cache: torch.Tensor | None = None  # [B, H, T, D] or quantized
        self.v_cache: torch.Tensor | None = None

        # Quantization params
        self.k_scales: torch.Tensor | None = None
        self.v_scales: torch.Tensor | None = None

        # Position tracking
        self.current_pos = 0
        self.cache_len = 0

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new K, V and return full sequences.

        Args:
            k: [B, H, T_new, D]
            v: [B, H, T_new, D]

        Returns:
            k_full: [B, H, T_total, D]
            v_full: [B, H, T_total, D]
        """
        batch, heads, seq_new, dim = k.shape

        if self.k_cache is None:
            # First update - initialize cache
            self.k_cache = k
            self.v_cache = v
            self.cache_len = seq_new
            self.current_pos = seq_new
            return k, v

        # Quantize new K, V if needed
        if self.quantize_kv:
            k_q, k_scale = self._quantize_int8(k)
            v_q, v_scale = self._quantize_int8(v)

            # Dequantize existing cache
            k_full = self._dequantize_int8(self.k_cache, self.k_scales)
            v_full = self._dequantize_int8(self.v_cache, self.v_scales)

            # Concatenate
            k_full = torch.cat([k_full, k], dim=2)
            v_full = torch.cat([v_full, v], dim=2)

            # Apply sliding window with sinks
            k_full, v_full = self._apply_window(k_full, v_full)

            # Re-quantize
            self.k_cache, self.k_scales = self._quantize_int8(k_full)
            self.v_cache, self.v_scales = self._quantize_int8(v_full)
        else:
            # No quantization - direct concat
            k_full = torch.cat([self.k_cache, k], dim=2)
            v_full = torch.cat([self.v_cache, v], dim=2)

            # Apply sliding window
            k_full, v_full = self._apply_window(k_full, v_full)

            self.k_cache = k_full
            self.v_cache = v_full

        self.cache_len = k_full.shape[2]
        self.current_pos += seq_new

        return k_full, v_full

    def _apply_window(
        self,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply sliding window with sink tokens."""
        total_len = k.shape[2]

        if total_len <= self.max_window:
            return k, v

        # Keep sink tokens + recent window
        sink_k = k[:, :, :self.sink_tokens]
        sink_v = v[:, :, :self.sink_tokens]

        recent_k = k[:, :, -(self.max_window - self.sink_tokens):]
        recent_v = v[:, :, -(self.max_window - self.sink_tokens):]

        return torch.cat([sink_k, recent_k], dim=2), torch.cat([sink_v, recent_v], dim=2)

    def _quantize_int8(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize to INT8 per-head per-channel."""
        # x: [B, H, T, D]
        scale = x.abs().amax(dim=(0, 2), keepdim=True) / 127.0  # [1, H, 1, D]
        scale = scale.clamp(min=1e-6)
        q = (x / scale).round().clamp(-128, 127).to(torch.int8)
        return q, scale

    def _dequantize_int8(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize INT8 to target dtype."""
        return (q.to(torch.float32) * scale).to(self.dtype)

    def get_mask(self, query_len: int) -> torch.Tensor:
        """
        Generate causal attention mask for current cache state.

        Args:
            query_len: Length of query sequence

        Returns:
            mask: [1, 1, query_len, cache_len]
        """
        cache_len = self.cache_len
        total_len = cache_len

        # Create causal mask
        mask = torch.ones(query_len, total_len, dtype=torch.bool, device=self.device)
        mask = torch.triu(mask, diagonal=1)  # True for positions to mask
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, query_len, total_len]

        return mask

    def reset(self):
        """Reset cache to empty."""
        self.k_cache = None
        self.v_cache = None
        self.k_scales = None
        self.v_scales = None
        self.current_pos = 0
        self.cache_len = 0

    def get_memory_usage(self) -> dict:
        """Get cache memory usage."""
        if self.k_cache is None:
            return {'total_mb': 0, 'k_mb': 0, 'v_mb': 0}

        k_mb = self.k_cache.numel() * self.k_cache.element_size() / (1024 * 1024)
        v_mb = self.v_cache.numel() * self.v_cache.element_size() / (1024 * 1024)

        return {
            'total_mb': k_mb + v_mb,
            'k_mb': k_mb,
            'v_mb': v_mb,
            'cache_len': self.cache_len,
            'quantized': self.quantize_kv,
        }


class MultiLayerKVCache:
    """
    Multi-layer KV cache manager.

    Manages separate KV caches for each transformer layer.
    """

    def __init__(
        self,
        n_layers: int,
        max_window: int = 256,
        sink_tokens: int = 1,
        n_heads: int = 8,
        head_dim: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None,
        quantize_kv: bool = True
    ):
        self.n_layers = n_layers
        self.caches = nn.ModuleList([
            KVCache(
                max_window=max_window,
                sink_tokens=sink_tokens,
                n_heads=n_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
                quantize_kv=quantize_kv
            )
            for _ in range(n_layers)
        ])

    def update_layer(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update specific layer's cache."""
        return self.caches[layer_idx].update(k, v)

    def get_mask(self, query_len: int, layer_idx: int, device: torch.device) -> torch.Tensor:
        """Get attention mask for specific layer."""
        return self.caches[layer_idx].get_mask(query_len).to(device)

    def reset(self):
        """Reset all layer caches."""
        for cache in self.caches:
            cache.reset()

    def get_total_memory_usage(self) -> dict:
        """Get combined memory usage across all layers."""
        total_mb = 0
        for _i, cache in enumerate(self.caches):
            usage = cache.get_memory_usage()
            total_mb += usage['total_mb']

        return {
            'total_mb': total_mb,
            'n_layers': self.n_layers,
        }


class CausalAttention(nn.Module):
    """
    Causal Multi-Head Attention with integrated KV Cache support.

    Combines the attention mechanism with efficient KV caching
    for autoregressive generation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.1,
        max_window: int = 256,
        sink_tokens: int = 1,
        use_ternary: bool = False,
        quantize_kv: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim or (d_model // n_heads)
        self.dropout = dropout
        self.max_window = max_window
        self.sink_tokens = sink_tokens
        self.use_ternary = use_ternary
        self.quantize_kv = quantize_kv

        self.scale = self.head_dim ** -0.5

        # Combined QKV projection
        if use_ternary:
            from ..weight_paging.ternary import TernaryLinear
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

        # KV Cache
        self.kv_cache = KVCache(
            max_window=max_window,
            sink_tokens=sink_tokens,
            n_heads=n_heads,
            head_dim=self.head_dim,
            dtype=dtype,
            device=device,
            quantize_kv=quantize_kv
        )

        # Precomputed RoPE
        self.register_buffer('cos', None, persistent=False)
        self.register_buffer('sin', None, persistent=False)
        self.max_seq_len = 2048  # Will be extended if needed

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.inference_mode = False

    def _ensure_rope(self, seq_len: int, device: torch.device):
        """Ensure RoPE tables are large enough."""
        if self.cos is None or self.cos.shape[0] < seq_len:
            new_len = max(seq_len, self.max_seq_len)
            cos, sin = precompute_rope_cos_sin(
                new_len, self.head_dim, device=device, dtype=self.cos.dtype if self.cos is not None else torch.bfloat16
            )
            self.cos = cos
            self.sin = sin
            self.max_seq_len = new_len

    def prepare_for_inference(self):
        """Prepare for inference mode."""
        self.inference_mode = True
        if self.use_ternary:
            self.qkv_proj.prepare_for_inference()
            self.out_proj.prepare_for_inference()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with optional KV caching.

        Args:
            x: [batch, seq_len, d_model]
            mask: Optional additional mask [batch, 1, seq_len, seq_len]
            use_cache: Whether to use/update KV cache

        Returns:
            Output: [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape
        device = x.device

        # Ensure RoPE tables
        self._ensure_rope(seq_len, device)

        # QKV projection
        qkv = self.qkv_proj(x)

        # Split Q, K, V
        qkv = qkv.reshape(batch, seq_len, self.n_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [B, H, T, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)

        # Apply RoPE
        q = apply_rotary_emb(q, self.cos, self.sin)
        k = apply_rotary_emb(k, self.cos, self.sin)

        # Handle KV cache
        if use_cache and self.inference_mode:
            k, v = self.kv_cache.update(k, v)
        else:
            pass

        # Causal mask
        causal_mask = self.kv_cache.get_mask(seq_len) if use_cache else None

        if mask is not None:
            # Combine with provided mask
            mask = mask & causal_mask if causal_mask is not None else causal_mask
        else:
            mask = causal_mask

        # Scaled dot-product attention
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

    def reset_cache(self):
        """Reset KV cache."""
        self.kv_cache.reset()

    def get_memory_usage(self) -> dict:
        """Get KV cache memory usage."""
        return self.kv_cache.get_memory_usage()


def create_causal_attention(config: dict, device: torch.device | None = None) -> CausalAttention:
    """Create CausalAttention from config."""
    return CausalAttention(
        d_model=config.get('d_model', 512),
        n_heads=config.get('n_heads', 8),
        head_dim=config.get('head_dim'),
        dropout=config.get('dropout', 0.1),
        max_window=config.get('max_window', 256),
        sink_tokens=config.get('sink_tokens', 1),
        use_ternary=config.get('use_ternary', False),
        quantize_kv=config.get('quantize_kv', True),
        device=device,
        dtype=config.get('dtype', torch.bfloat16)
    )


if __name__ == "__main__":
    # Quick test

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16

    # Test CausalAttention
    attn = CausalAttention(
        d_model=512,
        n_heads=8,
        head_dim=64,
        max_window=256,
        sink_tokens=1,
        use_ternary=True,
        quantize_kv=True,
        device=device,
        dtype=dtype
    )

    attn.prepare_for_inference()

    # Test generation loop
    x = torch.randn(1, 1, 512, device=device, dtype=dtype)

    for _i in range(5):
        out = attn(x, use_cache=True)
        x = out  # In real generation, this would be next token embedding


    # Test reset
    attn.reset_cache()

    # Test MultiLayerKVCache
    ml_cache = MultiLayerKVCache(
        n_layers=4,
        max_window=256,
        n_heads=8,
        head_dim=64,
        device=device,
        dtype=dtype
    )

    k = torch.randn(1, 8, 4, 64, device=device, dtype=dtype)
    v = torch.randn(1, 8, 4, 64, device=device, dtype=dtype)

    for layer in range(4):
        k_full, v_full = ml_cache.update_layer(layer, k, v)


