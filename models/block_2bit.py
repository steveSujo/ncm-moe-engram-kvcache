"""
2-bit Transformer Block

Block2Bit: Pre-LN block with CQQuantizedAttention + CQQuantizedMoEBlock.
Supports Engram injection at specified layers.
Gradient checkpointing compatible.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from models.common import RMSNorm
from models.cq_quantize import CQQuantizedLinear, CQ_GROUP_SIZE, WEIGHT_BITS
from models.moe_block_2bit import CQQuantizedMoEBlock
from models.engram_2bit import EngramModule2Bit


class CQQuantizedAttention(nn.Module):
    """Multi-head Attention with 2-bit CQ quantized projections."""
    
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_kv_head = getattr(config, "n_kv_head", config.n_head)
        self.head_dim = self.n_embd // self.n_head
        self.dropout = getattr(config, "attn_dropout", 0.0)
        self.bias = getattr(config, "bias", False)
        
        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0
        
        # 2-bit quantized projections
        self.c_attn = CQQuantizedLinear(
            self.n_embd, self.n_embd + 2 * self.n_kv_head * self.head_dim,
            bias=self.bias,
            bits=WEIGHT_BITS,
            group_size=CQ_GROUP_SIZE,
        )
        self.c_proj = CQQuantizedLinear(
            self.n_embd, self.n_embd,
            bias=self.bias,
            bits=WEIGHT_BITS,
            group_size=CQ_GROUP_SIZE,
        )
        
        # QK Norm (FP32)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(getattr(config, "resid_dropout", 0.0))
        
        # For RoPE
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)
    
    def _rope(self, q, k, cos, sin):
        """Apply rotary positional embeddings."""
        # q, k: [B, H, T, D]
        B, H, T, D = q.shape
        cos = cos[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, D/2]
        sin = sin[:T].unsqueeze(0).unsqueeze(0)
        
        q_rot = q[..., :D//2]
        q_pass = q[..., D//2:]
        k_rot = k[..., :D//2]
        k_pass = k[..., D//2:]
        
        q_rot = q_rot * cos + self._rotate_half(q_rot) * sin
        k_rot = k_rot * cos + self._rotate_half(k_rot) * sin
        
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
        return q, k
    
    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
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
        
        # Combined QKV projection (2-bit quantized)
        qkv = self.c_attn(x)
        
        # Split Q, K, V
        q = qkv[:, :, :self.n_embd]
        kv = qkv[:, :, self.n_embd:]
        
        # Reshape for multi-head
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = kv[:, :, :self.n_kv_head * self.head_dim].view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = kv[:, :, self.n_kv_head * self.head_dim:].view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        # QK Norm (FP32)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # RoPE
        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            q, k = self._rope(q, k, cos, sin)
        
        # Engram KV injection
        if engram_kv is not None:
            ek, ev = engram_kv
            if ek is not None:
                ek = ek.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
                ev = ev.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
                k = torch.cat([ek, k], dim=2)
                v = torch.cat([ev, v], dim=2)
        
        # KV Cache
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, layer_idx)
        
        # Repeat K,V for GQA
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        
        # Attention
        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        attn_out = attn_probs @ v  # [B, H, T, D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection (2-bit quantized)
        attn_out = self.c_proj(attn_out)
        attn_out = self.resid_dropout(attn_out)
        
        return attn_out
    
    def prepare_for_inference(self):
        """Switch to inference mode (pre-quantize weights)."""
        self.c_attn.prepare_for_inference()
        self.c_proj.prepare_for_inference()


# Import F here
import torch.nn.functional as F


class Block2Bit(nn.Module):
    """Transformer block with 2-bit attention and MoE."""
    
    def __init__(
        self,
        config,
        layer_idx: int = 0,
        pager=None,
        engram_layers: List[int] = None,
        engram_config: dict = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.engram_layers = engram_layers or [2, 11]
        self.has_engram = layer_idx in self.engram_layers
        
        self.ln_1 = RMSNorm(config.n_embd, bias=config.bias)
        self.attn = CQQuantizedAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, bias=config.bias)
        
        # MoE Block (pager set externally)
        self.mlp = CQQuantizedMoEBlock(
            layer_idx=layer_idx,
            d_model=config.n_embd,
            expert_hidden=config.n_embd * 4,  # 4x expansion
            pager=pager,
            n_experts=getattr(config, "n_experts", 64),
            top_k=getattr(config, "top_k", 2),
            capacity_factor=getattr(config, "capacity_factor", 1.25),
            router_loss_weight=getattr(config, "router_loss_weight", 0.01),
            bits=WEIGHT_BITS,
            group_size=CQ_GROUP_SIZE,
        )
        
        # Engram module (only at specified layers)
        if self.has_engram and engram_config is not None:
            self.engram = EngramModule2Bit(
                d_model=config.n_embd,
                num_slots=engram_config.get("num_slots", 8192),
                orders=engram_config.get("orders", [2, 3]),
                num_heads=engram_config.get("num_heads", 4),
                sub_dim=engram_config.get("sub_dim", 128),
                conv_kernel=engram_config.get("conv_kernel", 4),
                conv_dilation=engram_config.get("conv_dilation", 3),
                gate_temperature=engram_config.get("gate_temperature", 1.0),
                bits=WEIGHT_BITS,
                group_size=CQ_GROUP_SIZE,
            )
        else:
            self.engram = None
    
    def set_pager(self, pager):
        """Set pager after model creation."""
        self.mlp.pager = pager
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_cache: Optional = None,
        engram_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attn(
            self.ln_1(x),
            attn_mask=attn_mask,
            rope_cos_sin=rope_cos_sin,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
            engram_kv=engram_kv,
        )
        x = x + attn_out
        
        # Engram injection (if this layer has engram)
        if self.has_engram and self.engram is not None:
            engram_out, _ = self.engram(x, layer_idx=self.layer_idx)
            x = x + engram_out
        
        # MoE with residual
        mlp_out = self.mlp(self.ln_2(x))
        x = x + mlp_out
        
        return x
    
    def prepare_for_inference(self):
        """Switch to inference mode."""
        self.attn.prepare_for_inference()
        if self.engram is not None:
            self.engram.prepare_for_inference()


if __name__ == "__main__":
    # Quick test
    import torch
    from omegaconf import OmegaConf
    
    print("Testing Block2Bit...")
    
    config = OmegaConf.create({
        "n_embd": 768,
        "n_head": 12,
        "n_kv_head": 4,
        "bias": False,
        "attn_dropout": 0.0,
        "resid_dropout": 0.0,
        "n_experts": 64,
        "top_k": 2,
        "capacity_factor": 1.25,
        "router_loss_weight": 0.01,
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from models.moe_block_2bit import create_cq_quantized_moe_block
        
        # Create a pager for testing
        block_moe, pager = create_cq_quantized_moe_block(
            layer_idx=0,
            d_model=768,
            expert_hidden=3072,
            expert_dir=tmpdir,
            n_layers=12,
            n_experts=64,
            resident_slots=32,
        )
        
        block = Block2Bit(config, layer_idx=2, pager=pager)
        
        B, T, D = 2, 16, 768
        x = torch.randn(B, T, D)
        
        with torch.no_grad():
            out = block(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
        print("Block2Bit test passed!")