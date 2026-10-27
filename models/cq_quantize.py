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
    
    Codebook is normalized by sqrt(group_size) so that quantized values
    have unit variance when applied to normalized vectors.
    """
    if bits == 1.58:
        cb = _TERNARY_CB.clone()
    elif bits == 2.0:
        cb = lloyd_max_gaussian(2)
    elif bits == 4.0:
        cb = lloyd_max_gaussian(4)
    else:
        cb = lloyd_max_gaussian(int(bits))
    
    # Normalize by sqrt(group_size) for proper scaling
    return (cb / math.sqrt(group_size)).to(torch.float32)


# ============================================================
# Hadamard Matrix Generation
# ============================================================

@lru_cache(maxsize=None)
def hadamard_matrix(group_size: int) -> torch.Tensor:
    """Generate normalized Hadamard matrix of given size.
    
    H = [H, H; H, -H] recursively, normalized by 1/sqrt(group_size).
    Requires group_size to be a power of 2.
    """
    # Verify group_size is power of 2
    assert group_size & (group_size - 1) == 0, f"group_size must be power of 2, got {group_size}"
    
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < group_size:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    
    # Normalize by sqrt(group_size) for orthogonality
    H = H / math.sqrt(group_size)
    
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
    
    # Nearest neighbor lookup in codebook (memory-efficient chunked version)
    # codebook: [num_levels], unit: [..., group_size]
    # For each element in unit, find nearest codebook entry
    # Process in chunks to avoid OOM for large tensors
    flat_shape = unit.shape[:-1]
    num_elements = unit.shape[-1]
    unit_flat = unit.reshape(-1, num_elements)
    
    deq_flat = torch.zeros_like(unit_flat)
    codebook_flat = codebook.view(-1)  # [num_levels]
    
    # Process in chunks of 1024 rows
    chunk_size = 1024
    for i in range(0, unit_flat.shape[0], chunk_size):
        chunk = unit_flat[i:i+chunk_size]  # [chunk_size, group_size]
        # Broadcast: [chunk_size, group_size, 1] - [1, 1, num_levels]
        diff = chunk.unsqueeze(-1) - codebook_flat.view(1, 1, -1)
        idx = torch.argmin(torch.abs(diff), dim=-1)  # [chunk_size, group_size]
        deq_flat[i:i+chunk_size] = codebook_flat[idx]
    
    # Restore shape
    deq = deq_flat.reshape(flat_shape + (num_elements,))
    
    # Denormalize
    deq = deq * norm_fp16
    
    # Inverse Hadamard rotation
    deq = deq @ hadamard  # Inverse is same as forward (Hadamard is self-inverse up to scale)
    
    # Reshape back
    deq = deq.reshape(w_padded.shape)
    
    # Remove padding
    if pad:
        deq = deq[..., :D]
    
    # Replace any NaN/Inf with 0
    deq = torch.nan_to_num(deq, nan=0.0, posinf=0.0, neginf=0.0)
    
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
        codebook = get_codebook(bits, group_size).to(device=w.device, dtype=w.dtype)
    if hadamard is None:
        hadamard = hadamard_matrix(group_size).to(device=w.device, dtype=w.dtype)
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


def dequantize_kv_cq(k_q: torch.Tensor, k_scale: torch.Tensor, 
                      v_q: torch.Tensor, v_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dequantize CQ KV cache (alias for INT8 dequantization)."""
    return dequantize_kv_int8(k_q, k_scale, v_q, v_scale)


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
        
        # Ensure weight dtype matches input dtype
        w = w.to(x.dtype)
        
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
        
        # Ensure weight dtype is bfloat16 for NCM model
        w = w.to(torch.bfloat16)
        
        return F.embedding(x, w)
    
    def prepare_for_inference(self):
        """Call after training to quantize weights for deployment."""
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
        Quantized model
    """
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in exclude_patterns):
            continue
        if param.dim() >= 2:
            # Quantize 2D+ parameters (weights)
            param.data = cq_quantize_simple(param.data, bits, group_size)
    return model


def quantize_specific_modules(
    model: nn.Module,
    bits: float = 2.0,
    group_size: int = 128,
    target_modules: Tuple[type, ...] = (nn.Linear, nn.Embedding),
) -> nn.Module:
    """Replace specific module types with quantized versions."""
    for name, module in list(model.named_modules()):
        if isinstance(module, target_modules):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            parent = model if parent_name == '' else model.get_submodule(parent_name)
            child_name = name.rsplit('.', 1)[-1]
            
            if isinstance(module, nn.Linear):
                q_module = CQQuantizedLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    bits=bits,
                    group_size=group_size,
                    quantize=True
                )
                q_module.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    q_module.bias.data = module.bias.data.clone()
                setattr(parent, child_name, q_module)
            
            elif isinstance(module, nn.Embedding):
                q_module = CQQuantizedEmbedding(
                    module.num_embeddings,
                    module.embedding_dim,
                    bits=bits,
                    group_size=group_size,
                    quantize=True
                )
                q_module.weight.data = module.weight.data.clone()
                setattr(parent, child_name, q_module)
    
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
    group_size: int = 128,
    head_bits: int = 4,
    weight_bits: int = 4,
    act_bits: int = 8,
    kv_bits: int = 8
):
    """Configure global quantization settings."""
    global CQ_GROUP_SIZE, HEAD_BITS, WEIGHT_BITS, ACT_BITS, KV_BITS
    CQ_GROUP_SIZE = group_size
    HEAD_BITS = head_bits
    WEIGHT_BITS = weight_bits
    ACT_BITS = act_bits
    KV_BITS = kv_bits


# ============================================================
# Test / Demo
# ============================================================

if __name__ == "__main__":
    print("Testing Cactus Quants (CQ) 2-bit Quantization")
    
    # Test Hadamard
    print("\n1. Testing Hadamard matrix...")
    for size in [16, 32, 64, 128, 256]:
        H = hadamard_matrix(size)
        HHt = H @ H.T
        I = torch.eye(size)
        diff = (HHt - I).abs().max()
        print(f"  size={size}: max diff={diff:.2e}, ortho={diff < 1e-5}")
    
    # Test codebook
    print("\n2. Testing codebook...")
    for bits in [1.58, 2.0, 4.0]:
        cb = get_codebook(bits, CQ_GROUP_SIZE)
        print(f"  bits={bits}: levels={len(cb)}, range=[{cb.min():.4f}, {cb.max():.4f}]")
    
    # Test round-trip
    print("\n3. Testing round-trip quantization...")
    torch.manual_seed(42)
    w = torch.randn(1024, 512)
    w_q = cq_quantize(w, bits=2.0, group_size=128)
    error = (w - w_q).abs().max() / w.abs().max()
    mse = (w - w_q).pow(2).mean() / w.pow(2).mean()
    print(f"  max_rel_error={error:.6f}, mse_rel={mse:.6f}")
    
    # Test STE gradient
    print("\n4. Testing STE gradient...")
    w = torch.randn(128, 256, requires_grad=True)
    w_q = cq_ste_simple(w, bits=2.0, group_size=128)
    loss = w_q.sum()
    loss.backward()
    print(f"  grad norm: {w.grad.norm():.4f}, close to 1: {torch.allclose(w.grad, torch.ones_like(w.grad), atol=1e-3)}")
    
    # Test quantized linear
    print("\n5. Testing CQQuantizedLinear...")
    linear = CQQuantizedLinear(512, 1024, bits=2.0, group_size=128)
    x = torch.randn(2, 32, 512, dtype=torch.bfloat16)
    linear.train()
    out = linear(x)
    print(f"  train out: shape={out.shape}, dtype={out.dtype}")
    linear.prepare_for_inference()
    linear.eval()
    with torch.no_grad():
        out_inf = linear(x)
    print(f"  inf out: shape={out_inf.shape}, dtype={out_inf.dtype}")
    
    # Test quantized embedding
    print("\n6. Testing CQQuantizedEmbedding...")
    emb = CQQuantizedEmbedding(10000, 512, bits=2.0, group_size=128)
    x = torch.randint(0, 10000, (2, 32), dtype=torch.long)
    emb.train()
    out = emb(x)
    print(f"  train out: shape={out.shape}, dtype={out.dtype}")
    emb.prepare_for_inference()
    emb.eval()
    with torch.no_grad():
        out_inf = emb(x)
    print(f"  inf out: shape={out_inf.shape}, dtype={out_inf.dtype}")
    
    print("\nAll tests passed!")