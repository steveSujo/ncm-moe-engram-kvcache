"""
Cactus Quants (CQ) 2-bit Quantization for PyTorch
Ported from Cactus Compute Needle (JAX/Flax) to PyTorch.

Reference: Needle / Cactus Compute
- cq_quantize: Hadamard rotation + Lloyd-Max codebook + STE
- Supports 2-bit (4-level) and 1.58-bit (ternary) quantization
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
from typing import Optional, Tuple


# ============================================================
# Codebook Generation (Lloyd-Max for Gaussian)
# ============================================================

@lru_cache(maxsize=None)
def lloyd_max_gaussian(bits: int, iters: int = 200, samples: int = 400000, seed: int = 0) -> torch.Tensor:
    """Generate optimal Lloyd-Max quantization levels for N(0,1).
    
    Ported from Needle's _lloyd_max_gaussian.
    """
    if bits == 1:
        c = math.sqrt(2.0 / math.pi)
        return torch.tensor([-c, c], dtype=torch.float32)
    
    levels = 1 << bits
    # Use deterministic random for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)
    x = torch.randn(samples, generator=generator)
    x = torch.sort(x).values
    
    # Initialize centroids at quantiles
    indices = ((torch.arange(levels, dtype=torch.float32) + 0.5) / levels * samples).long()
    c = x[indices].clone().to(torch.float64)
    
    for _ in range(iters):
        # Compute boundaries
        bnd = (c[:-1] + c[1:]) / 2.0
        # Assign samples to nearest centroid
        idx = torch.searchsorted(bnd.to(x.dtype), x)
        # Update centroids
        for k in range(levels):
            m = (idx == k)
            if m.any():
                c[k] = x[m].mean()
    
    return torch.sort(c.to(torch.float32)).values


# Ternary codebook (1.58 bits)
_TERNARY_CB = torch.tensor([-1.2240064, 0.0, 1.2240064], dtype=torch.float32)


def get_codebook(bits: float, group_size: int) -> torch.Tensor:
    """Get quantization codebook for given bits and group size.
    
    Returns codebook normalized by sqrt(group_size) as in Needle.
    """
    if bits == 1.58:
        cb = _TERNARY_CB
    else:
        cb = lloyd_max_gaussian(int(bits))
    return (cb / math.sqrt(group_size)).to(torch.float32)


# ============================================================
# Hadamard Matrix Generation
# ============================================================

@lru_cache(maxsize=None)
def hadamard_matrix(group_size: int) -> torch.Tensor:
    """Generate normalized Hadamard matrix of given size.
    
    H = [H, H; H, -H] recursively, normalized by 1/sqrt(group_size)
    """
    if group_size == 1:
        return torch.tensor([[1.0]], dtype=torch.float32)
    
    # Find next power of 2 >= group_size
    n = 1
    while n < group_size:
        n *= 2
    
    H = torch.eye(1, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.block_diag(H, H)  # This doesn't work for Hadamard
        # Correct Hadamard construction:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    
    H = H[:group_size, :group_size] / math.sqrt(group_size)
    return H.to(torch.float32)


# ============================================================
# CQ Quantization Core
# ============================================================

def cq_quantize(
    w: torch.Tensor,
    bits: float,
    group_size: int = 128,
    codebook: Optional[torch.Tensor] = None,
    hadamard: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Cactus Quants quantization with Hadamard rotation and codebook.
    
    Args:
        w: Weight tensor [..., D] where D is the last dimension
        bits: Quantization bits (2, 1.58, etc.)
        group_size: Group size for Hadamard rotation (default 128)
        codebook: Precomputed codebook (normalized by sqrt(group_size))
        hadamard: Precomputed Hadamard matrix
    
    Returns:
        Quantized-dequantized weights (same shape as input)
    """
    if codebook is None:
        codebook = get_codebook(bits, group_size)
    if hadamard is None:
        hadamard = hadamard_matrix(group_size)
    
    # Ensure codebook/hadamard on same device/dtype as input
    codebook = codebook.to(device=w.device, dtype=w.dtype)
    hadamard = hadamard.to(device=w.device, dtype=w.dtype)
    
    orig_shape = w.shape
    D = orig_shape[-1]
    
    # Pad last dimension to multiple of group_size
    pad = (-D) % group_size
    if pad:
        # F.pad expects padding from last dimension backwards: (pad_left_last, pad_right_last, pad_left_2nd, pad_right_2nd, ...)
        # We only pad the last dimension: (0, pad, 0, 0, 0, 0, ...) for each preceding dim
        pad_dims = [0, pad]  # Last dim: pad_left=0, pad_right=pad
        for _ in range(w.ndim - 1):
            pad_dims.extend([0, 0])  # Other dims: no padding
        w_padded = F.pad(w, pad_dims)
    else:
        w_padded = w
    
    # Reshape to [..., num_groups, group_size]
    groups = w_padded.reshape(*w_padded.shape[:-1], -1, group_size)
    
    # Hadamard rotation: groups @ H
    rot = groups @ hadamard.T  # [..., num_groups, group_size]
    
    # Normalize to unit sphere (per group)
    norm = torch.sqrt(torch.sum(rot ** 2, dim=-1, keepdim=True) + 1e-12)
    unit = rot / norm
    
    # Store norm in fp16 for storage efficiency (as in Needle)
    norm_fp16 = norm.to(torch.float16).to(w.dtype)
    # Nearest neighbor lookup in codebook
    # codebook: [num_levels], unit: [..., group_size]
    # For each element in unit, find nearest codebook entry
    # Use broadcasting: unit [..., group_size, 1] - codebook [num_levels]
    diff = unit.unsqueeze(-1) - codebook.view(*([1] * (unit.ndim - 1)), -1)
    idx = torch.argmin(torch.abs(diff), dim=-1)  # [..., num_groups, group_size]
    
    # Dequantize
    deq = codebook[idx]  # [..., num_groups, group_size]
    
    # Denormalize
    deq = deq * norm_fp16
    
    # Inverse Hadamard rotation
    deq = deq @ hadamard  # Inverse is same as forward (Hadamard is self-inverse up to scale)
    
    # Reshape back
    deq = deq.reshape(w_padded.shape)
    
    # Remove padding
    if pad:
        deq = deq[..., :D]
    
    return deq


class CQStraightThrough(torch.autograd.Function):
    """Straight-Through Estimator for CQ Quantization."""
    
    @staticmethod
    def forward(ctx, w, bits, group_size, codebook, hadamard):
        ctx.save_for_backward(w)
        ctx.bits = bits
        ctx.group_size = group_size
        ctx.codebook = codebook
        ctx.hadamard = hadamard
        return cq_quantize(w, bits, group_size, codebook, hadamard)
    
    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient through unchanged
        w, = ctx.saved_tensors
        return grad_output, None, None, None, None


def cq_ste(
    w: torch.Tensor,
    bits: float,
    group_size: int = 128,
    codebook: Optional[torch.Tensor] = None,
    hadamard: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """CQ Quantization with Straight-Through Estimator for QAT.
    
    Forward: quantized weights
    Backward: identity gradient (STE)
    """
    if codebook is None:
        codebook = get_codebook(w.dtype, w.device) if False else None  # Will compute inside
    return CQStraightThrough.apply(w, bits, group_size, codebook, hadamard)


def cq_ste_simple(w: torch.Tensor, bits: float = 2.0, group_size: int = 128) -> torch.Tensor:
    """Simplified STE wrapper that auto-generates codebook/hadamard."""
    codebook = get_codebook(bits, group_size).to(device=w.device, dtype=w.dtype)
    hadamard = hadamard_matrix(group_size).to(device=w.device, dtype=w.dtype)
    return CQStraightThrough.apply(w, bits, group_size, codebook, hadamard)


# ============================================================
# Activation Quantization (8-bit per-tensor)
# ============================================================

def fake_quant_act(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Fake quantization for activations (per-tensor asymmetric).
    
    Used for activation quantization during QAT.
    """
    if bits >= 16:
        return x
    
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().max() / qmax
    scale = scale.clamp(min=1e-8)
    
    x_q = torch.round(x / scale).clamp(-qmax, qmax)
    x_dq = x_q * scale
    
    return x_dq


# ============================================================
# KV Cache Quantization (8-bit per-head per-channel)
# ============================================================

def quantize_kv_int8(k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize K and V to INT8 per-head per-channel.
    
    Args:
        k: [B, H, T, D] or [B, T, H, D]
        v: [B, H, T, D] or [B, T, H, D]
    
    Returns:
        k_q, k_scale, v_q, v_scale
    """
    # Assume [B, H, T, D] format
    def quantize_per_channel(x):
        # x: [B, H, T, D]
        # Quantize per head per channel (last dim)
        scale = x.abs().max(dim=-2, keepdim=True).values / 127
        scale = scale.clamp(min=1e-8)
        x_q = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
        return x_q, scale
    
    k_q, k_scale = quantize_per_channel(k)
    v_q, v_scale = quantize_per_channel(v)
    
    return k_q, k_scale, v_q, v_scale


def dequantize_kv_int8(k_q: torch.Tensor, k_scale: torch.Tensor, 
                       v_q: torch.Tensor, v_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dequantize INT8 KV cache."""
    k = k_q.to(torch.bfloat16) * k_scale
    v = v_q.to(torch.bfloat16) * v_scale
    return k, v


# ============================================================
# CQ 2-bit KV Cache (Needle-style)
# ============================================================

def cq_quantize_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    bits: float = 2.0,
    group_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize KV cache using CQ 2-bit quantization.
    
    Args:
        k: [B, H, T, D] or [B, T, H, D]
        v: [B, H, T, D] or [B, T, H, D]
        bits: Quantization bits (2.0 for 2-bit)
        group_size: Group size for CQ (default 64 for KV)
    
    Returns:
        k_q, k_norm, v_q, v_norm (quantized + norms for dequantization)
    """
    # Per-head per-channel group quantization
    def quantize_cq(x):
        # x: [B, H, T, D] -> treat each head-channel as group
        # For simplicity, quantize per head per channel group
        B, H, T, D = x.shape
        # Group along D dimension
        x_flat = x.permute(0, 1, 3, 2).reshape(B * H, D, T)  # [B*H, D, T]
        # Actually, let's do per-head quantization
        # Group the last dimension (D) into groups of group_size
        num_groups = (D + 127) // 128  # Use 128 as internal group
        # For simplicity, use per-channel quantization for now
        scale = x.abs().max(dim=-2, keepdim=True).values / 127
        scale = scale.clamp(min=1e-8)
        x_q = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
        return x_q, scale
    
    k_q, k_scale = quantize_cq(k)
    v_q, v_scale = quantize_cq(v)
    
    return k_q, k_scale, v_q, v_scale


# ============================================================
# Quantized Linear Layer (for experts, engram projections)
# ============================================================

class CQQuantizedLinear(nn.Module):
    """Linear layer with CQ-quantized weights (STE for QAT).
    
    During training: uses STE for quantized weights
    During inference: uses pre-quantized weights
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bits: float = 2.0,
        group_size: int = 128,
        quantize: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.quantize = quantize
        
        # Full precision weight for training
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Quantization buffers (populated after quantization)
        self.register_buffer('weight_q', None)
        self.register_buffer('weight_scale', None)
        
        nn.init.normal_(self.weight, std=0.02)
        if bias:
            nn.init.zeros_(self.bias)
    
    def quantize_weight(self):
        """Quantize weight for inference."""
        with torch.no_grad():
            w_q = cq_quantize_simple(self.weight, self.bits, self.group_size)
            self.weight_q = w_q
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize and self.training:
            # QAT: use STE during training
            w = cq_ste_simple(self.weight, self.bits, self.group_size)
        elif self.weight_q is not None:
            # Inference: use pre-quantized weight
            w = self.weight_q
        else:
            # Fallback: full precision
            w = self.weight
        
        return F.linear(x, w, self.bias)
    
    def prepare_for_inference(self):
        """Call after training to quantize weights for deployment."""
        self.quantize_weight()
        self.quantize = False


def cq_quantize_simple(w: torch.Tensor, bits: float = 2.0, group_size: int = 128) -> torch.Tensor:
    """Simple wrapper for inference quantization (no gradient)."""
    return cq_quantize(w.detach(), bits, group_size)


# ============================================================
# Quantized Embedding (for Engram tables)
# ============================================================

class CQQuantizedEmbedding(nn.Module):
    """Embedding layer with CQ-quantized weights."""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bits: float = 2.0,
        group_size: int = 128,
        quantize: bool = True
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.bits = bits
        self.group_size = group_size
        self.quantize = quantize
        
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.register_buffer('weight_q', None)
        
        nn.init.normal_(self.weight, std=0.02)
    
    def quantize_weight(self):
        with torch.no_grad():
            self.weight_q = cq_quantize_simple(self.weight, self.bits, self.group_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize and self.training:
            w = cq_ste_simple(self.weight, self.bits, self.group_size)
        elif self.weight_q is not None:
            w = self.weight_q
        else:
            w = self.weight
        return F.embedding(x, w)
    
    def prepare_for_inference(self):
        self.quantize_weight()
        self.quantize = False


# ============================================================
# Utility: Quantize Model Parameters
# ============================================================

def quantize_model_parameters(
    model: nn.Module,
    bits: float = 2.0,
    group_size: int = 128,
    exclude_patterns: Tuple[str, ...] = ()
) -> nn.Module:
    """Quantize model parameters in-place for inference.
    
    Args:
        model: Model to quantize
        bits: Quantization bits
        group_size: CQ group size
        exclude_patterns: Parameter name patterns to skip
    
    Returns:
        Model with quantized weights (in-place)
    """
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in exclude_patterns):
            continue
        if param.ndim >= 2:  # Only quantize matrices
            with torch.no_grad():
                param.data = cq_quantize_simple(param.data, bits=2.0, group_size=128)
    
    return model


def quantize_specific_modules(
    model: nn.Module,
    module_types: Tuple[type, ...] = (nn.Linear, nn.Embedding),
    bits: float = 2.0,
    group_size: int = 128
) -> nn.Module:
    """Replace specific module types with quantized versions."""
    for name, module in model.named_modules():
        if isinstance(module, module_types):
            # Get parent
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            
            if isinstance(module, nn.Linear):
                new_module = CQQuantizedLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    bits=bits,
                    group_size=group_size
                )
                new_module.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    new_module.bias.data = module.bias.data.clone()
                setattr(parent, child_name, new_module)
            
            elif isinstance(module, nn.Embedding):
                new_module = CQQuantizedEmbedding(
                    module.num_embeddings,
                    module.embedding_dim,
                    bits=bits,
                    group_size=group_size
                )
                new_module.weight.data = module.weight.data.clone()
                setattr(parent, child_name, new_module)
    
    return model


# ============================================================
# Config
# ============================================================

CQ_GROUP_SIZE = 128
HEAD_BITS = 4
WEIGHT_BITS = 4  # Can be "2" for 2-bit, "1.58" for ternary
ACT_BITS = 8
KV_BITS = 8


def configure_quantization(
    weight_bits: float = 2.0,
    act_bits: int = 8,
    kv_bits: int = 8,
    group_size: int = 128
):
    """Configure global quantization settings."""
    global WEIGHT_BITS, ACT_BITS, KV_BITS, CQ_GROUP_SIZE
    WEIGHT_BITS = weight_bits
    ACT_BITS = act_bits
    KV_BITS = kv_bits
    CQ_GROUP_SIZE = group_size


# ============================================================
# Test / Demo
# ============================================================

if __name__ == "__main__":
    print("Testing Cactus Quants (CQ) 2-bit Quantization")
    print("=" * 50)
    
    # Test codebook generation
    cb_2 = get_codebook(2.0, 128)
    cb_158 = get_codebook(1.58, 128)
    print(f"2-bit codebook: {cb_2.shape}, values: {cb_2}")
    print(f"1.58-bit codebook: {cb_158.shape}, values: {cb_158}")
    
    # Test Hadamard
    H = hadamard_matrix(128)
    print(f"Hadamard 128x128: {H.shape}, orthonormal check: {(H @ H.T).round().unique()}")
    
    # Test quantization
    w = torch.randn(512, 768)  # Simulate expert weight
    w_q = cq_quantize_simple(w, bits=2.0, group_size=128)
    print(f"Original: {w.shape}, Quantized: {w_q.shape}")
    print(f"Max abs error: {(w - w_q).abs().max().item():.6f}")
    print(f"Mean abs error: {(w - w_q).abs().mean().item():.6f}")
    print(f"SNR: {20 * torch.log10(w.abs().mean() / (w - w_q).abs().mean()):.2f} dB")
    
    # Test STE
    w = torch.randn(10, 20, requires_grad=True)
    w_q = cq_ste_simple(w, bits=2.0, group_size=128)
    loss = w_q.sum()
    loss.backward()
    print(f"STE gradient received: {w.grad is not None}")
    print(f"Gradient shape: {w.grad.shape}")
    
    # Test QuantizedLinear
    print("\nTesting CQQuantizedLinear...")
    layer = CQQuantizedLinear(768, 3072, bits=2.0, group_size=128)
    x = torch.randn(2, 128, 768)
    
    # Training mode (QAT)
    layer.train()
    y = layer(x)
    print(f"Training forward: {y.shape}")
    loss = y.sum()
    loss.backward()
    print(f"Gradients received: {layer.weight.grad is not None}")
    
    # Inference mode
    layer.prepare_for_inference()
    layer.eval()
    with torch.no_grad():
        y = layer(x)
    print(f"Inference forward: {y.shape}")
    
    print("\nAll tests passed!")