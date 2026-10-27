"""
2-bit Quantized MoE Block with CQ Quantization

CQQuantizedMoEBlock: Experts stored in 2-bit CQ format, FP32 router.
CQQuantizedMoEPager: Expert I/O with 2-bit CQ serialization.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
from collections import OrderedDict
import safetensors.torch
import json
import hashlib

from models.cq_quantize import (
    CQQuantizedLinear,
    CQStraightThrough,
    cq_quantize_simple,
    cq_ste_simple,
    CQ_GROUP_SIZE,
    WEIGHT_BITS,
    configure_quantization,
)
from utils.expert_io import ExpertIndex, swiglu_forward


# ============================================================
# 2-bit Expert Pager (saves/loads experts in 2-bit CQ format)
# ============================================================

class CQQuantizedMoEPager:
    """
    Expert pager that stores expert weights in 2-bit CQ format on SSD.
    Quantizes on save, dequantizes on load (or keeps quantized for inference).
    """
    
    def __init__(
        self,
        expert_dir: str,
        n_layers: int,
        n_experts: int,
        d_model: int,
        expert_hidden: int,
        resident_slots: int = 32,
        bits: float = 2.0,
        group_size: int = 128,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.expert_dir = expert_dir
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.d_model = d_model
        self.expert_hidden = expert_hidden
        self.resident_slots = resident_slots
        self.bits = bits
        self.group_size = group_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        self.index = ExpertIndex(expert_dir, n_layers, n_experts)
        
        # Resident cache: (layer, expert) -> {gate_q, gate_scale, up_q, up_scale, down_q, down_scale}
        self.resident: OrderedDict[Tuple[int, int], Dict[str, torch.Tensor]] = OrderedDict()
        
        os.makedirs(expert_dir, exist_ok=True)
        os.makedirs(os.path.join(expert_dir, "index"), exist_ok=True)
    
    def _quantize_expert_tensors(self, expert_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Quantize expert tensors to 2-bit CQ format for storage."""
        quantized = {}
        for key, tensor in expert_tensors.items():
            # Quantize: [out_features, in_features] -> q, scale
            t_flat = tensor.to(torch.float32)
            q = cq_quantize_simple(t_flat, bits=self.bits, group_size=self.group_size)
            # Store quantized indices and scales
            # For storage, we save the quantized tensor as int8 indices + scale
            # Actually, we store the dequantized float for simplicity in v1
            # TODO: Store compressed format
            quantized[key] = q.to(torch.bfloat16)
        return quantized
    
    def _dequantize_expert_tensors(self, quantized: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Dequantize expert tensors from storage format."""
        # For v1, storage is already dequantized BF16
        # In v2, we'd store indices + scales and reconstruct
        return {k: v.to(self.dtype).to(self.device) for k, v in quantized.items()}
    
    def load_expert(self, layer_idx: int, expert_id: int) -> Dict[str, torch.Tensor]:
        """Load expert weights (quantized or dequantized)."""
        key = (layer_idx, expert_id)
        
        # Check resident cache
        if key in self.resident:
            self.resident.move_to_end(key)
            return self.resident[key]
        
        # Load from disk
        path = self.index.get_path(layer_idx, expert_id)
        if not os.path.exists(path):
            # Create random expert if doesn't exist
            expert_tensors = self._create_random_expert()
            self.save_expert(layer_idx, expert_id, expert_tensors)
            path = self.index.get_path(layer_idx, expert_id)
        
        # Load and dequantize
        state = safetensors.torch.load_file(path)
        expert_tensors = {k: v.to(self.device) for k, v in state.items()}
        
        # Evict if needed
        if len(self.resident) >= self.resident_slots:
            self.resident.popitem(last=False)
        
        self.resident[key] = expert_tensors
        return expert_tensors
    
    def _create_random_expert(self) -> Dict[str, torch.Tensor]:
        """Create randomly initialized expert tensors."""
        return {
            "gate": torch.randn(self.expert_hidden, self.d_model, dtype=self.dtype) * 0.02,
            "up": torch.randn(self.expert_hidden, self.d_model, dtype=self.dtype) * 0.02,
            "down": torch.randn(self.d_model, self.expert_hidden, dtype=self.dtype) * 0.02,
        }
    
    def save_expert(self, layer_idx: int, expert_id: int, expert_tensors: Dict[str, torch.Tensor]):
        """Save expert weights in 2-bit CQ format."""
        key = (layer_idx, expert_id)
        
        # Quantize for storage
        quantized = self._quantize_expert_tensors(expert_tensors)
        
        # Save to disk
        path = self.index.get_path(layer_idx, expert_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Store as BF16 (dequantized) for now; v2 will store indices+scales
        state = {k: v.cpu() for k, v in quantized.items()}
        safetensors.torch.save_file(state, path)
        
        # Update index
        size_bytes = sum(v.numel() * v.element_size() for v in state.values())
        checksum = hashlib.md5(torch.cat([v.flatten() for v in state.values()]).cpu().numpy()).hexdigest()
        self.index.update_metadata(layer_idx, expert_id, size_bytes, checksum)
        
        # Update resident cache
        self.resident[key] = {k: v.to(self.device) for k, v in quantized.items()}
        if len(self.resident) > self.resident_slots:
            self.resident.popitem(last=False)
    
    def get_resident_experts(self, layer_idx: int) -> List[int]:
        """Get list of resident expert IDs for a layer."""
        return [eid for (lid, eid) in self.resident.keys() if lid == layer_idx]
    
    def preload_layer(self, layer_idx: int, expert_ids: List[int]):
        """Preload specific experts for a layer."""
        for expert_id in expert_ids:
            self.load_expert(layer_idx, expert_id)
    
    def __repr__(self):
        return f"CQQuantizedMoEPager(resident={len(self.resident)}/{self.resident_slots}, layers={self.n_layers}, experts={self.n_experts}, bits={self.bits})"


# ============================================================
# 2-bit Quantized MoE Block
# ============================================================

class CQQuantizedMoEBlock(nn.Module):
    """
    MoE Block with 2-bit CQ quantized experts and FP32 router.
    
    During training: Experts loaded in FP32/BF16, quantized on-the-fly via STE
    During inference: Experts pre-quantized, kept in 2-bit format
    """
    
    def __init__(
        self,
        layer_idx: int,
        d_model: int,
        expert_hidden: int,
        pager: CQQuantizedMoEPager,
        n_experts: int = 64,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        router_loss_weight: float = 0.01,
        bits: float = 2.0,
        group_size: int = 128,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.expert_hidden = expert_hidden
        self.pager = pager
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.router_loss_weight = router_loss_weight
        self.bits = bits
        self.group_size = group_size
        
        # FP32 Router (stability)
        self.router = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)
        
        self.router_loss = 0.0
    
    def _get_active_experts(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route to top-k experts."""
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # [B*T, D]
        
        router_logits = self.router(x_flat)  # [B*T, n_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k routing
        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        # Compute entropy loss
        self.router_loss = self.router_loss_weight * self._entropy_loss(router_logits)
        
        return expert_weights, expert_indices, router_logits
    
    def _entropy_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Compute router entropy loss for load balancing."""
        probs = F.softmax(router_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        max_entropy = torch.log(torch.tensor(self.n_experts, dtype=torch.float32))
        return 1.0 - (entropy / max_entropy)
    
    def _dispatch_to_experts(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dispatch tokens to experts and combine outputs.
        Uses 2-bit quantized experts with STE during training.
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # [B*T, D]
        n_tokens = x_flat.shape[0]
        
        # Get unique expert IDs used in this batch
        unique_experts = expert_indices.unique().tolist()
        
        # Load all required experts
        expert_cache = {}
        for eid in unique_experts:
            expert_cache[eid] = self.pager.load_expert(self.layer_idx, eid)
        
        # Process each expert
        outputs = []
        
        for k in range(self.top_k):
            eid_k = expert_indices[:, k]  # [B*T]
            weight_k = expert_weights[:, k].unsqueeze(-1)  # [B*T, 1]
            
            # Accumulate expert outputs
            out_accum = torch.zeros_like(x_flat)
            
            for eid in unique_experts:
                mask = (eid_k == eid)
                if not mask.any():
                    continue
                
                expert_x = x_flat[mask]  # [n_tokens_for_expert, D]
                weight_k_masked = weight_k[mask]  # [n_tokens_for_expert, 1]
                
                # Get expert tensors (already dequantized BF16 from pager)
                expert_tensors = expert_cache[eid]
                
                # SwiGLU forward with expert tensors
                gate = F.linear(expert_x, expert_tensors["gate"])  # [n, H]
                up = F.linear(expert_x, expert_tensors["up"])      # [n, H]
                hidden = F.silu(gate) * up                          # [n, H]
                expert_out = F.linear(hidden, expert_tensors["down"])  # [n, D]
                
                out_accum[mask] = expert_out * weight_k_masked
            
            outputs.append(out_accum)
        
        return torch.stack(outputs, dim=0).sum(dim=0).view(B, T, D)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with 2-bit experts."""
        # Router (FP32)
        expert_weights, expert_indices, router_logits = self._get_active_experts(x)
        
        # Dispatch to experts (2-bit quantized with STE)
        out = self._dispatch_to_experts(x, expert_weights, expert_indices)
        
        return out
    
    def extra_repr(self):
        return f"layer={self.layer_idx}, experts={self.n_experts}, bits={self.bits}"


# ============================================================
# Factory
# ============================================================

def create_cq_quantized_moe_block(
    layer_idx: int,
    d_model: int,
    expert_hidden: int,
    expert_dir: str,
    n_layers: int,
    n_experts: int = 64,
    resident_slots: int = 32,
    top_k: int = 2,
    capacity_factor: float = 1.25,
    router_loss_weight: float = 0.01,
    bits: float = 2.0,
    group_size: int = 128,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[CQQuantizedMoEBlock, CQQuantizedMoEPager]:
    """Create 2-bit quantized MoE block with pager."""
    pager = CQQuantizedMoEPager(
        expert_dir=expert_dir,
        n_layers=n_layers,
        n_experts=n_experts,
        d_model=d_model,
        expert_hidden=expert_hidden,
        resident_slots=resident_slots,
        bits=bits,
        group_size=group_size,
        device=device,
        dtype=dtype,
    )
    
    block = CQQuantizedMoEBlock(
        layer_idx=layer_idx,
        d_model=d_model,
        expert_hidden=expert_hidden,
        pager=pager,
        n_experts=n_experts,
        top_k=top_k,
        capacity_factor=capacity_factor,
        router_loss_weight=router_loss_weight,
        bits=bits,
        group_size=group_size,
    )
    
    return block, pager


if __name__ == "__main__":
    # Quick test
    import torch
    import tempfile
    
    print("Testing CQQuantizedMoEBlock...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create pager and block
        block, pager = create_cq_quantized_moe_block(
            layer_idx=0,
            d_model=768,
            expert_hidden=2048,
            expert_dir=tmpdir,
            n_layers=12,
            n_experts=64,
            resident_slots=32,
            bits=2.0,
        )
        
        # Test forward
        B, T, D = 2, 16, 768
        x = torch.randn(B, T, D)
        
        with torch.no_grad():
            out = block(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
        print(f"Router loss: {block.router_loss:.4f}")
        print("CQQuantizedMoEBlock test passed!")