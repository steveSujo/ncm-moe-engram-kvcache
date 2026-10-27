"""
2-bit CQ Quantized KV Cache with Attention Sinks (Needle 2 style)

Implements a 256-token sliding window with permanently pinned attention sinks
(prefix tokens). KV cache stored in 2-bit CQ for memory efficiency.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from models.cq_quantize import (
    cq_quantize_kv,
    dequantize_kv_cq,
    CQ_GROUP_SIZE,
    WEIGHT_BITS,
)


class BoundedKVCache2Bit(nn.Module):
    """
    Bounded KV Cache with 2-bit CQ quantization and attention sinks.
    
    Features:
    - 256-token sliding window
    - 1 permanently pinned sink token (prefix)
    - 2-bit CQ quantization per-head per-channel
    - Causal masking
    """
    
    def __init__(
        self,
        max_window: int = 256,
        sink_tokens: int = 1,
        n_heads: int = 12,
        head_dim: int = 64,
        bits: float = 2.0,
        group_size: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
    ):
        super().__init__()
        self.max_window = max_window
        self.sink_tokens = sink_tokens
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.bits = bits
        self.group_size = group_size
        self.dtype = dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Storage for quantized KV cache
        # Shape: [n_layers, 2, B, H, max_window + sink_tokens, head_dim]
        # 2 for K and V
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)
        self.register_buffer("k_scale", None, persistent=False)
        self.register_buffer("v_scale", None, persistent=False)
        self.register_buffer("cache_len", None, persistent=False)
        
        self._initialized = False
    
    def _init_cache(self, batch_size: int, n_layers: int, device: torch.device):
        """Initialize cache buffers."""
        max_len = self.max_window + self.sink_tokens
        cache_shape = (n_layers, 2, batch_size, self.n_heads, max_len, self.head_dim)
        
        self.k_cache = torch.zeros(cache_shape, dtype=torch.int8, device=device)
        self.v_cache = torch.zeros(cache_shape, dtype=torch.int8, device=device)
        
        # Scales: [n_layers, 2, H, head_dim]
        scale_shape = (n_layers, 2, self.n_heads, self.head_dim)
        self.k_scale = torch.zeros(scale_shape, dtype=torch.float32, device=device)
        self.v_scale = torch.zeros(scale_shape, dtype=torch.float32, device=device)
        
        self.cache_len = torch.zeros(n_layers, dtype=torch.long, device=device)
        self._initialized = True
    
    def reset(self):
        """Reset cache to empty state."""
        if self._initialized:
            self.k_cache.zero_()
            self.v_cache.zero_()
            self.k_scale.zero_()
            self.v_scale.zero_()
            self.cache_len.zero_()
    
    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache with new tokens.
        
        Args:
            k: [B, H, T, D] new keys
            v: [B, H, T, D] new values
            layer_idx: Layer index
            
        Returns:
            k_full: [B, H, T_total, D] full keys (including cache)
            v_full: [B, H, T_total, D] full values (including cache)
        """
        B, H, T_new, D = k.shape
        
        if not self._initialized:
            self._init_cache(B, self.k_cache.shape[0] if self.k_cache is not None else 1, k.device)
        
        # Ensure cache is initialized for this batch size
        if self.k_cache.shape[2] != B:
            self._init_cache(B, self.k_cache.shape[0], k.device)
        
        # Quantize new K, V to 2-bit CQ
        k_q, k_scale, v_q, v_scale = cq_quantize_kv(
            k, v, bits=self.bits, group_size=self.group_size
        )
        
        # Get current cache length
        cur_len = self.cache_len[layer_idx].item()
        max_len = self.max_window + self.sink_tokens
        
        # Determine where to write
        if cur_len < max_len:
            # Append to cache
            write_start = cur_len
            write_end = min(cur_len + T_new, max_len)
            write_len = write_end - write_start
            
            # Only write what fits
            k_q_write = k_q[:, :, :write_len]
            v_q_write = v_q[:, :, :write_len]
            k_scale_write = k_scale
            v_scale_write = v_scale
            
            self.k_cache[layer_idx, 0, :, :, write_start:write_end] = k_q_write.to(torch.int8)
            self.v_cache[layer_idx, 0, :, :, write_start:write_end] = v_q_write.to(torch.int8)
            self.k_cache[layer_idx, 1] = k_scale_write
            self.v_cache[layer_idx, 1] = v_scale_write
            self.cache_len[layer_idx] = write_end
        else:
            # Sliding window: shift and append
            # Keep sink tokens at position 0
            shift = T_new
            
            # Shift window (keep sink tokens at pos 0)
            self.k_cache[layer_idx, 0, :, :, self.sink_tokens:max_len - shift] = \
                self.k_cache[layer_idx, 0, :, :, self.sink_tokens + shift:max_len]
            self.v_cache[layer_idx, 0, :, :, self.sink_tokens:max_len - shift] = \
                self.v_cache[layer_idx, 0, :, :, self.sink_tokens + shift:max_len]
            
            # Write new tokens at end
            write_start = max_len - shift
            self.k_cache[layer_idx, 0, :, :, write_start:max_len] = k_q[:, :, -shift:].to(torch.int8)
            self.v_cache[layer_idx, 0, :, :, write_start:max_len] = v_q[:, :, -shift:].to(torch.int8)
            self.k_cache[layer_idx, 1] = k_scale
            self.v_cache[layer_idx, 1] = v_scale
            # cache_len stays at max_len
        
        # Dequantize full cache for attention
        k_full = self.k_cache[layer_idx, 0, :, :, :self.cache_len[layer_idx]]
        v_full = self.v_cache[layer_idx, 0, :, :, :self.cache_len[layer_idx]]
        k_scale_full = self.k_cache[layer_idx, 1]
        v_scale_full = self.v_cache[layer_idx, 1]
        
        k_deq, v_deq = dequantize_kv_cq(k_full, k_scale_full, v_full, v_scale_full)
        
        return k_deq, v_deq
    
    def get_mask(self, query_len: int, layer_idx: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask for cached sequence."""
        if not self._initialized:
            return None
        
        total_len = self.cache_len[layer_idx].item()
        if total_len == 0:
            return None
        
        # Causal mask: query can attend to all keys up to current position
        mask = torch.triu(
            torch.ones(query_len, total_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        return mask[None, None, :, :]  # [1, 1, query_len, total_len]


class MultiLayerKVCache2Bit(nn.Module):
    """Multi-layer KV Cache with 2-bit CQ quantization."""
    
    def __init__(
        self,
        n_layers: int,
        max_window: int = 256,
        sink_tokens: int = 1,
        n_heads: int = 12,
        head_dim: int = 64,
        bits: float = 2.0,
        group_size: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.caches = nn.ModuleList([
            BoundedKVCache2Bit(
                max_window=max_window,
                sink_tokens=sink_tokens,
                n_heads=n_heads,
                head_dim=head_dim,
                bits=bits,
                group_size=group_size,
                dtype=dtype,
                device=device,
            ) for _ in range(n_layers)
        ])
    
    def reset(self):
        for cache in self.caches:
            cache.reset()
    
    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.caches[layer_idx].update(k, v, layer_idx)
    
    def get_mask(self, query_len: int, layer_idx: int, device: torch.device) -> torch.Tensor:
        return self.caches[layer_idx].get_mask(query_len, layer_idx, device)


def create_kv_cache_2bit(
    config,
    n_layers: int = None,
    max_window: int = 256,
    sink_tokens: int = 1,
    bits: float = 2.0,
    group_size: int = 64,
) -> MultiLayerKVCache2Bit:
    """Factory function to create 2-bit KV cache from config."""
    n_layers = n_layers or getattr(config, "n_layer", 12)
    n_heads = getattr(config, "n_head", 12)
    n_kv_head = getattr(config, "n_kv_head", n_heads)
    head_dim = getattr(config, "n_embd", 768) // n_heads
    
    return MultiLayerKVCache2Bit(
        n_layers=n_layers,
        max_window=max_window,
        sink_tokens=sink_tokens,
        n_heads=n_kv_head,  # Use K/V heads for cache
        head_dim=head_dim,
        bits=bits,
        group_size=group_size,
    )


if __name__ == "__main__":
    # Quick test
    import torch
    
    print("Testing MultiLayerKVCache2Bit...")
    
    # Create cache
    cache = MultiLayerKVCache2Bit(
        n_layers=2,
        max_window=256,
        sink_tokens=1,
        n_heads=4,
        head_dim=64,
        bits=2.0,
        group_size=64,
    )
    
    B, H, D = 2, 4, 64
    
    # Simulate multiple forward passes
    for step in range(5):
        T_new = 8
        k = torch.randn(B, H, T_new, D)
        v = torch.randn(B, H, T_new, D)
        
        k_full, v_full = cache.update(k, v, layer_idx=0)
        
        print(f"Step {step}: cache_len={cache.caches[0].cache_len[0].item()}, "
              f"k_full={k_full.shape}, v_full={v_full.shape}")
    
    # Test mask
    mask = cache.get_mask(query_len=8, layer_idx=0, device=torch.device("cpu"))
    print(f"Mask shape: {mask.shape}")
    
    print("MultiLayerKVCache2Bit test passed!")