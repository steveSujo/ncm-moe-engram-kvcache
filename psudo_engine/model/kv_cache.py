"""
Bounded KV Cache with Attention Sinks and INT8 Quantization.

Implements a 256-token sliding window with permanently pinned attention sinks
(prefix tokens). KV cache stored in INT8 for memory efficiency.
"""


import torch
import torch.nn as nn


class BoundedKVCache(nn.Module):
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
        quantize: bool = True
    ):
        super().__init__()
        self.max_window = max_window
        self.sink_tokens = sink_tokens
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.quantize = quantize

        # Cache storage
        self.register_buffer('k_cache', torch.empty(0), persistent=False)
        self.register_buffer('v_cache', torch.empty(0), persistent=False)

        # Quantization params
        self.register_buffer('k_scales', torch.empty(0), persistent=False)
        self.register_buffer('v_scales', torch.empty(0), persistent=False)

        # State
        self.current_pos = 0
        self.cache_len = 0
        self.batch_size = 0

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

        if self.k_cache.numel() == 0:
            # First update - initialize cache
            self.batch_size = batch
            if self.quantize:
                self.k_cache, self.k_scales = self._quantize_int8(k)
                self.v_cache, self.v_scales = self._quantize_int8(v)
            else:
                self.k_cache = k
                self.v_cache = v
            self.cache_len = seq_new
            self.current_pos = seq_new
            return k, v

        # Quantize new K, V if needed
        if self.quantize:
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

    def get_mask(self, query_len: int, key_len: int | None = None) -> torch.Tensor:
        """
        Generate causal attention mask for current cache state.
        
        Args:
            query_len: Length of query sequence
            key_len: Length of key sequence (defaults to cache length for autoregressive generation)
            
        Returns:
            mask: [1, 1, query_len, key_len] - True for positions to attend
        """
        if key_len is None:
            # For autoregressive generation, the full key sequence includes
            # cached tokens plus the current query tokens, capped at max_window
            key_len = min(self.cache_len + query_len, self.max_window)
        
        # Create causal mask (True = attend, False = masked)
        mask = torch.ones(query_len, key_len, dtype=torch.bool, device=self.device)
        mask = torch.triu(mask, diagonal=1)  # True for positions to mask (future)
        mask = ~mask  # Invert: True = attend, False = masked
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, query_len, key_len]
        
        return mask

    def get_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get dequantized cache."""
        if self.k_cache.numel() == 0:
            return torch.empty(0), torch.empty(0)

        if self.quantize:
            k = self._dequantize_int8(self.k_cache, self.k_scales)
            v = self._dequantize_int8(self.v_cache, self.v_scales)
        else:
            k = self.k_cache
            v = self.v_cache

        return k, v

    def reset(self):
        """Reset cache to empty."""
        self.k_cache = torch.empty(0, device=self.device)
        self.v_cache = torch.empty(0, device=self.device)
        self.k_scales = torch.empty(0, device=self.device)
        self.v_scales = torch.empty(0, device=self.device)
        self.current_pos = 0
        self.cache_len = 0
        self.batch_size = 0

    def get_memory_usage(self) -> dict:
        """Get cache memory usage."""
        if self.k_cache.numel() == 0:
            return {'total_mb': 0, 'k_mb': 0, 'v_mb': 0}

        k_mb = self.k_cache.numel() * self.k_cache.element_size() / (1024 * 1024)
        v_mb = self.v_cache.numel() * self.v_cache.element_size() / (1024 * 1024)

        if self.quantize:
            # Add scales
            scale_mb = (self.k_scales.numel() + self.v_scales.numel()) * 4 / (1024 * 1024)
        else:
            scale_mb = 0

        return {
            'total_mb': k_mb + v_mb + scale_mb,
            'k_mb': k_mb,
            'v_mb': v_mb,
            'scale_mb': scale_mb,
            'cache_len': self.cache_len,
            'quantized': self.quantize,
        }


class MultiLayerKVCache(nn.Module):
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
        quantize: bool = True
    ):
        super().__init__()
        self.n_layers = n_layers

        self.caches = nn.ModuleList([
            BoundedKVCache(
                max_window=max_window,
                sink_tokens=sink_tokens,
                n_heads=n_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
                quantize=quantize
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

    def get_cache_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cache for specific layer."""
        return self.caches[layer_idx].get_cache()

    def get_mask(self, query_len: int, layer_idx: int, device: torch.device) -> torch.Tensor:
        """Get attention mask for specific layer."""
        # For autoregressive generation, use cache_len as key_len
        return self.caches[layer_idx].get_mask(query_len, key_len=None).to(device)

    def reset(self):
        """Reset all layer caches."""
        for cache in self.caches:
            cache.reset()

    def get_total_memory_usage(self) -> dict:
        """Get combined memory usage across all layers."""
        total_mb = 0
        layer_details = []

        for i, cache in enumerate(self.caches):
            usage = cache.get_memory_usage()
            total_mb += usage['total_mb']
            layer_details.append({
                'layer': i,
                'cache_len': usage['cache_len'],
                'total_mb': usage['total_mb'],
            })

        return {
            'total_mb': total_mb,
            'n_layers': self.n_layers,
            'layers': layer_details,
        }

    def get_layer_memory_usage(self, layer_idx: int) -> dict:
        """Get memory usage for specific layer."""
        return self.caches[layer_idx].get_memory_usage()


def create_kv_cache(config: dict, device: torch.device | None = None) -> BoundedKVCache:
    """Create BoundedKVCache from config."""
    return BoundedKVCache(
        max_window=config.get('max_window', 256),
        sink_tokens=config.get('sink_tokens', 1),
        n_heads=config.get('n_heads', 8),
        head_dim=config.get('head_dim', 64),
        dtype=config.get('dtype', torch.bfloat16),
        device=device,
        quantize=config.get('quantize_kv', True)
    )


def create_multi_layer_kv_cache(config: dict, device: torch.device | None = None) -> MultiLayerKVCache:
    """Create MultiLayerKVCache from config."""
    dtype_str = config.get('dtype', 'bfloat16')
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
        'fp16': torch.float16,
        'fp32': torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)
    
    return MultiLayerKVCache(
        n_layers=config.get('n_layers', 24),
        max_window=config.get('max_window', 256),
        sink_tokens=config.get('sink_tokens', 1),
        n_heads=config.get('n_heads', 8),
        head_dim=config.get('head_dim', 64),
        dtype=dtype,
        device=device,
        quantize=config.get('quantize_kv', True)
    )


if __name__ == "__main__":
    # Quick test

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16

    # Test single layer cache
    cache = BoundedKVCache(
        max_window=256,
        sink_tokens=1,
        n_heads=8,
        head_dim=64,
        dtype=dtype,
        device=device,
        quantize=True
    )

    # Simulate generation
    for _i in range(5):
        k = torch.randn(1, 8, 1, 64, device=device, dtype=dtype)
        v = torch.randn(1, 8, 1, 64, device=device, dtype=dtype)

        k_full, v_full = cache.update(k, v)

    # Test reset
    cache.reset()

    # Test multi-layer
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


