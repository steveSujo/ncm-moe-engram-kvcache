"""
Engram Key-Value Memory Module with 2-bit CQ Quantization (Needle 2 style)

2-bit quantized embeddings and projections, FP32 RMSNorm and Conv1D.
Matches EngramModule interface exactly for drop-in replacement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from models.cq_quantize import (
    CQQuantizedEmbedding,
    CQQuantizedLinear,
    CQ_GROUP_SIZE,
    WEIGHT_BITS,
    configure_quantization,
)
from models.engram import fnv1a_hash, engram_indices, engram_geometry, needle2_engram_config


class EngramModule2Bit(nn.Module):
    """
    Engram with 2-bit quantized embeddings and projections.
    
    Uses CQQuantizedEmbedding for tables and CQQuantizedLinear for k_proj/v_proj.
    Keeps RMSNorm and Conv1D in FP32 for stability.
    """
    
    def __init__(
        self,
        d_model: int,
        num_slots: int = 8192,
        orders: List[int] = [2, 3],
        num_heads: int = 4,
        sub_dim: int = 128,
        conv_kernel: int = 4,
        conv_dilation: int = 3,
        gate_temperature: float = 1.0,
        bits: float = 2.0,
        group_size: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.orders = orders
        self.num_heads = num_heads
        self.sub_dim = sub_dim
        self.gate_temperature = gate_temperature
        self.bits = bits
        self.group_size = group_size
        
        total_dim = num_heads * sub_dim
        assert total_dim <= d_model, f"Engram total dim {total_dim} > d_model {d_model}"
        
        # 2-bit quantized embedding tables (keys and values)
        self.k_tables = nn.ModuleList([
            CQQuantizedEmbedding(num_slots, sub_dim, bits=bits, group_size=group_size)
            for _ in orders
        ])
        self.v_tables = nn.ModuleList([
            CQQuantizedEmbedding(num_slots, sub_dim, bits=bits, group_size=group_size)
            for _ in orders
        ])
        
        # 2-bit quantized projections
        self.k_proj = CQQuantizedLinear(d_model, total_dim, bias=False, bits=bits, group_size=group_size)
        self.v_proj = CQQuantizedLinear(d_model, total_dim, bias=False, bits=bits, group_size=group_size)
        self.out_proj = CQQuantizedLinear(total_dim, d_model, bias=False, bits=bits, group_size=group_size)
        
        # Gate projection (FP32 for stability)
        self.gate_proj = nn.Linear(d_model, num_heads, bias=False)
        
        # RMSNorm per head (FP32)
        self.norms = nn.ModuleList([
            nn.RMSNorm(sub_dim, eps=1e-6) for _ in range(num_heads)
        ])
        
        # Dilated conv for context (FP32)
        self.context_conv = nn.Conv1d(
            total_dim, total_dim,
            kernel_size=conv_kernel,
            dilation=conv_dilation,
            padding=(conv_kernel - 1) * conv_dilation // 2,
            groups=num_heads,
            bias=False,
        )
        
        # Temperature for gating
        self.register_buffer("temperature", torch.tensor(gate_temperature))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling for 2-bit quantization."""
        # Gate projection: small initialization
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        
        # Conv: identity-like initialization
        with torch.no_grad():
            for i in range(self.num_heads):
                start = i * self.sub_dim
                end = start + self.sub_dim
                weight = self.context_conv.weight[start:end]
                weight.zero_()
                weight[:, :, self.context_conv.kernel_size[0] // 2].fill_(1.0 / self.num_heads)
    
    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int = 0,
        engram_orders: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with 2-bit quantized tables and projections.
        
        Args:
            x: [B, T, D] input
            layer_idx: Current layer index (for logging)
            engram_orders: Override orders for this layer
            
        Returns:
            out: [B, T, D] output with engram memory added
            gates: [B, T, H] gate values for monitoring
        """
        B, T, D = x.shape
        orders = engram_orders if engram_orders is not None else self.orders
        
        # Project to key/value space (2-bit quantized weights, STE gradient)
        k = self.k_proj(x)  # [B, T, H * sub_dim]
        v = self.v_proj(x)  # [B, T, H * sub_dim]
        
        # Reshape for multi-head: [B, T, H, sub_dim]
        k = k.view(B, T, self.num_heads, self.sub_dim)
        v = v.view(B, T, self.num_heads, self.sub_dim)
        
        # Per-head RMSNorm (FP32)
        k_norm = torch.stack([self.norms[h](k[:, :, h]) for h in range(self.num_heads)], dim=2)
        v_norm = torch.stack([self.norms[h](v[:, :, h]) for h in range(self.num_heads)], dim=2)
        
        # Compute gates (FP32)
        gates = torch.sigmoid(self.gate_proj(x) / self.temperature)  # [B, T, H]
        
        # Hash lookup for each order
        engram_out = torch.zeros_like(v_norm)
        
        for order_idx, order in enumerate(orders):
            if order_idx >= len(self.k_tables):
                continue
                
            # Compute n-gram hashes
            indices = engram_indices(k_norm[:, :, :, :order_idx + 1].reshape(B, T, -1), 
                                    self.num_slots, order)
            # indices: [B, T, H]
            
            # Lookup keys and values (2-bit quantized embeddings)
            k_mem = self.k_tables[order_idx](indices)  # [B, T, H, sub_dim]
            v_mem = self.v_tables[order_idx](indices)  # [B, T, H, sub_dim]
            
            # Similarity scoring
            sim = (k_norm * k_mem).sum(dim=-1) / (self.sub_dim ** 0.5)  # [B, T, H]
            sim = sim * gates  # Gate by learned importance
            
            # Softmax over heads for each order
            attn = F.softmax(sim, dim=-1)  # [B, T, H]
            
            # Weighted value retrieval
            engram_out = engram_out + (attn.unsqueeze(-1) * v_mem)
        
        # Apply dilated conv for context (FP32)
        engram_out = engram_out.reshape(B, T, -1).transpose(1, 2)  # [B, H*sub_dim, T]
        conv_out = self.context_conv(engram_out)
        conv_out = conv_out.transpose(1, 2).reshape(B, T, self.num_heads, self.sub_dim)
        
        # Output projection (2-bit quantized)
        out = conv_out.reshape(B, T, -1)  # [B, T, H*sub_dim]
        out = self.out_proj(out)  # [B, T, D]
        
        return out, gates
    
    def prepare_for_inference(self):
        """Switch all quantized modules to inference mode (pre-quantize weights)."""
        self.k_proj.prepare_for_inference()
        self.v_proj.prepare_for_inference()
        self.out_proj.prepare_for_inference()
        for table in self.k_tables:
            table.prepare_for_inference()
        for table in self.v_tables:
            table.prepare_for_inference()


def create_engram_2bit(config: dict = None, **kwargs) -> EngramModule2Bit:
    """Factory function to create EngramModule2Bit from config."""
    if config is None:
        config = needle2_engram_config()
    
    # Merge config with kwargs
    merged = {**config, **kwargs}
    
    # Extract quant-specific params
    bits = merged.pop("bits", WEIGHT_BITS)
    group_size = merged.pop("group_size", CQ_GROUP_SIZE)
    
    return EngramModule2Bit(bits=bits, group_size=group_size, **merged)


if __name__ == "__main__":
    # Quick test
    import torch
    
    print("Testing EngramModule2Bit...")
    
    # Create module
    engram = EngramModule2Bit(
        d_model=768,
        num_slots=8192,
        orders=[2, 3],
        num_heads=4,
        sub_dim=128,
        bits=2.0,
        group_size=128,
    )
    
    # Test forward
    B, T, D = 2, 16, 768
    x = torch.randn(B, T, D)
    
    with torch.no_grad():
        out, gates = engram(x, layer_idx=2)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Gates shape: {gates.shape}")
    print(f"Gate stats: min={gates.min():.4f}, max={gates.max():.4f}, mean={gates.mean():.4f}")
    
    # Test with gradients
    x = torch.randn(B, T, D, requires_grad=True)
    out, gates = engram(x, layer_idx=2)
    loss = out.sum()
    loss.backward()
    
    print(f"Gradient norm: {x.grad.norm():.4f}")
    print("EngramModule2Bit test passed!")