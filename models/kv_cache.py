"""
Bounded KV Cache with Attention Sinks and INT8 Quantization (Needle 2 style)

Implements a 256-token sliding window with permanently pinned attention sinks
(prefix tokens). KV cache stored in INT8 for memory efficiency.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List


def quantize_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-channel INT8 quantization for KV cache.
    
    Args:
        x: [..., C] or [..., H, C] tensor (BF16/FP16/FP32)
    Returns:
        q: INT8 quantized tensor
        scale: per-channel scale [C] or [H, C]
    """
    # Compute per-channel scale
    if x.dim() >= 3 and x.shape[-2] > 1:  # has head dim
        # Per-head per-channel: [B, H, T, C] -> scale [H, C]
        scale = x.abs().amax(dim=(0, 2)) / 127.0  # [H, C]
        scale = scale.clamp(min=1e-6)
        q = (x / scale.unsqueeze(0).unsqueeze(2)).round().clamp(-128, 127).to(torch.int8)
    else:
        # Per-channel: [..., C] -> scale [C]
        scale = x.abs().amax(dim=tuple(range(x.dim() - 1))) / 127.0
        scale = scale.clamp(min=1e-6)
        q = (x / scale.view(*([1] * (x.dim() - 1)), -1)).round().clamp(-128, 127).to(torch.int8)
    
    return q, scale


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Dequantize INT8 tensor using per-channel scale.
    
    Args:
        q: INT8 quantized tensor
        scale: per-channel scale
    Returns:
        Dequantized tensor in original dtype
    """
    if scale.dim() == 2:  # [H, C]
        scale = scale.unsqueeze(0).unsqueeze(2)  # [1, H, 1, C]
    else:  # [C]
        scale = scale.view(*([1] * (q.dim() - 1)), -1)
    
    return (q.to(torch.float32) * scale).to(torch.bfloat16)


class BoundedKVCache(nn.Module):
    """
    Bounded Key-Value Cache with Attention Sinks (Needle 2 style).
    
    Features:
    - Fixed-size sliding window (default 256 tokens)
    - Permanently pinned attention sinks (prefix tokens)
    - INT8 quantization for KV cache storage
    - Per-layer cache management
    """
    
    def __init__(
        self,
        max_window: int = 256,
        sink_tokens: int = 1,
        n_layers: int = 12,
        n_kv_heads: int = 3,  # grouped-query attention
        head_dim: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.max_window = max_window
        self.sink_tokens = sink_tokens
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device or torch.device('cpu')
        
        # Cache state - using lists per layer for dynamic sizing
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []
        self.k_scale: List[torch.Tensor] = []
        self.v_scale: List[torch.Tensor] = []
        self.sink_k: List[torch.Tensor] = []
        self.sink_v: List[torch.Tensor] = []
        self.sink_k_scale: List[torch.Tensor] = []
        self.sink_v_scale: List[torch.Tensor] = []
        self.initialized = False
        self.seq_len = 0
        
    def _init_cache(self, batch_size: int, device: torch.device):
        """Initialize cache buffers for all layers."""
        self.k_cache = []
        self.v_cache = []
        self.k_scale = []
        self.v_scale = []
        self.sink_k = []
        self.sink_v = []
        self.sink_k_scale = []
        self.sink_v_scale = []
        
        for _ in range(self.n_layers):
            self.k_cache.append(torch.zeros(
                batch_size, self.n_kv_heads, 0, self.head_dim,
                dtype=torch.int8, device=device
            ))
            self.v_cache.append(torch.zeros(
                batch_size, self.n_kv_heads, 0, self.head_dim,
                dtype=torch.int8, device=device
            ))
            self.k_scale.append(torch.zeros(
                self.n_kv_heads, self.head_dim,
                dtype=torch.float32, device=device
            ))
            self.v_scale.append(torch.zeros(
                self.n_kv_heads, self.head_dim,
                dtype=torch.float32, device=device
            ))
            self.sink_k.append(torch.zeros(
                batch_size, self.n_kv_heads, self.sink_tokens, self.head_dim,
                dtype=torch.int8, device=device
            ))
            self.sink_v.append(torch.zeros(
                batch_size, self.n_kv_heads, self.sink_tokens, self.head_dim,
                dtype=torch.int8, device=device
            ))
            self.sink_k_scale.append(torch.zeros(
                self.n_kv_heads, self.head_dim,
                dtype=torch.float32, device=device
            ))
            self.sink_v_scale.append(torch.zeros(
                self.n_kv_heads, self.head_dim,
                dtype=torch.float32, device=device
            ))
        self.initialized = True
        self.seq_len = 0
    
    def reset(self):
        """Reset cache state."""
        self.k_cache = []
        self.v_cache = []
        self.k_scale = []
        self.v_scale = []
        self.sink_k = []
        self.sink_v = []
        self.sink_k_scale = []
        self.sink_v_scale = []
        self.initialized = False
        self.seq_len = 0
    
    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,  # [B, n_kv_heads, T_new, head_dim]
        v: torch.Tensor,  # [B, n_kv_heads, T_new, head_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a specific layer and return full K, V for attention.
        
        Returns:
            k_full, v_full: [B, n_kv_heads, T_total, head_dim] in bfloat16
        """
        B, H, T_new, D = k.shape
        device = k.device
        
        # Initialize if needed
        if not self.initialized:
            self._init_cache(B, device)
        
        # Quantize incoming K, V to INT8
        k_q, k_scale = quantize_int8(k)  # [B, H, T_new, D] int8, [H, D] float32
        v_q, v_scale = quantize_int8(v)
        
        if self.seq_len == 0:
            # First forward: split sink + window
            sink_k_q = k_q[:, :, :self.sink_tokens]
            sink_v_q = v_q[:, :, :self.sink_tokens]
            window_k_q = k_q[:, :, self.sink_tokens:]
            window_v_q = v_q[:, :, self.sink_tokens:]
            
            self.sink_k[layer_idx] = sink_k_q
            self.sink_v[layer_idx] = sink_v_q
            self.sink_k_scale[layer_idx] = k_scale
            self.sink_v_scale[layer_idx] = v_scale
            
            self.k_cache[layer_idx] = window_k_q
            self.v_cache[layer_idx] = window_v_q
            self.k_scale[layer_idx] = k_scale
            self.v_scale[layer_idx] = v_scale
        else:
            # Append to window
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], k_q], dim=2)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], v_q], dim=2)
            
            # Update scales (use max for safety)
            self.k_scale[layer_idx] = torch.maximum(self.k_scale[layer_idx], k_scale)
            self.v_scale[layer_idx] = torch.maximum(self.v_scale[layer_idx], v_scale)
            
            # Evict oldest if window exceeded
            window_len = self.k_cache[layer_idx].size(2)
            if window_len > self.max_window:
                self.k_cache[layer_idx] = self.k_cache[layer_idx][:, :, -self.max_window:]
                self.v_cache[layer_idx] = self.v_cache[layer_idx][:, :, -self.max_window:]
        
        self.seq_len += T_new
        
        # Dequantize for attention computation
        # Sink
        sink_k = dequantize_int8(self.sink_k[layer_idx], self.sink_k_scale[layer_idx])
        sink_v = dequantize_int8(self.sink_v[layer_idx], self.sink_v_scale[layer_idx])
        
        # Window
        window_k = dequantize_int8(self.k_cache[layer_idx], self.k_scale[layer_idx])
        window_v = dequantize_int8(self.v_cache[layer_idx], self.v_scale[layer_idx])
        
        # Concat: [sink, window]
        k_full = torch.cat([sink_k, window_k], dim=2)  # [B, H, sink+window, D]
        v_full = torch.cat([sink_v, window_v], dim=2)
        
        return k_full, v_full
    
    def get_window_len(self, layer_idx: int) -> int:
        """Get current window length (excluding sink)."""
        if not self.initialized:
            return 0
        return self.k_cache[layer_idx].size(2)
    
    def get_total_len(self, layer_idx: int) -> int:
        """Get total KV length (sink + window)."""
        return self.sink_tokens + self.get_window_len(layer_idx)
    
    def get_mask(self, query_len: int, layer_idx: int, device: torch.device) -> torch.Tensor:
        """
        Get causal attention mask for sink + window.
        
        Returns:
            mask: [1, 1, query_len, total_kv_len] boolean
        """
        if not self.initialized:
            # No cache yet
            total_len = self.sink_tokens + query_len
        else:
            window_len = self.k_cache[layer_idx].size(2)
            total_len = self.sink_tokens + window_len + query_len
        
        # Causal mask: query can attend to sink + past window + current
        mask = torch.tril(
            torch.ones(query_len, total_len, dtype=torch.bool, device=device)
        )
        return mask[None, None, :, :]  # [1, 1, query_len, total_len]
    
class MultiLayerKVCache(nn.Module):
    """
    Manages KV caches for all layers.
    """
    
    def __init__(
        self,
        n_layers: int,
        max_window: int = 256,
        sink_tokens: int = 1,
        n_kv_heads: int = 3,
        head_dim: int = 64,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        # Single BoundedKVCache that manages all layers internally
        self.cache = BoundedKVCache(
            max_window=max_window,
            sink_tokens=sink_tokens,
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        self.n_layers = n_layers
    
    def reset(self):
        self.cache.reset()
    
    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cache.update(layer_idx, k, v)
    
    def get_mask(self, query_len: int, layer_idx: int, device: torch.device) -> torch.Tensor:
        return self.cache.get_mask(query_len, layer_idx, device)


def fake_quant_act(x: torch.Tensor, bit: int = 8) -> torch.Tensor:
    """
    Fake quantization for activations (STE during training).
    Used to simulate INT8 activation quantization.
    """
    if not x.requires_grad:
        # Inference: real quantization
        q, scale = quantize_int8(x)
        return dequantize_int8(q, scale)
    
    # Training: STE
    scale = x.detach().abs().amax(dim=tuple(range(x.dim() - 1))) / (2 ** (bit - 1) - 1)
    scale = scale.clamp(min=1e-6).view(*([1] * (x.dim() - 1)), -1)
    q = (x / scale).round().clamp(-(2 ** (bit - 1)), 2 ** (bit - 1) - 1)
    return (q * scale).to(x.dtype)


if __name__ == "__main__":
    # Quick test
    import torch
    
    cache = BoundedKVCache(max_window=256, sink_tokens=1, n_layers=12, n_kv_heads=3, head_dim=64)
    
    # Simulate 5 forward steps
    B, H, T, D = 2, 3, 10, 64
    for i in range(5):
        k = torch.randn(B, H, T, D, dtype=torch.bfloat16)
        v = torch.randn(B, H, T, D, dtype=torch.bfloat16)
        k_full, v_full = cache.update(0, k, v)
        print(f"Step {i}: window_len={cache.get_window_len(0)}, total_len={cache.get_total_len(0)}, k_full={k_full.shape}")
    
    # Test mask
    mask = cache.get_mask(5, 0, torch.device('cpu'))
    print(f"Mask shape: {mask.shape}")  # [1, 1, 5, total_len]
    
    # Test multi-layer
    ml_cache = MultiLayerKVCache(n_layers=12, max_window=256, sink_tokens=1)
    k_full, v_full = ml_cache.update(5, torch.randn(B, H, T, D), torch.randn(B, H, T, D))
    print(f"Multi-layer: {k_full.shape}")
    
    print("KV Cache test passed!")