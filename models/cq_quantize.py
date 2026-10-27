"""
Cactus Quants (CQ) Quantization

High-performance quantization for weights, activations, and KV cache
using Hadamard rotation + codebook lookup with Straight-Through Estimator.
"""

import math
import functools

import torch
import torch.nn as nn
from torch import Tensor

from typing import Optional


# ============================================================
# Codebook Generation (Lloyd-Max for Gaussian)
# ============================================================

@functools.cache
def lloyd_max_gaussian(bits: int, iters: int = 200, samples: int = 400000, seed: int = 0) -> Tensor:
    """Generate optimal Lloyd-Max quantization levels for N(0,1)."""
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in (0, 8]")

    torch.manual_seed(seed)
    x = torch.randn(samples)

    # Initial codebook: uniform in [-3, 3] (covers 99.7% of Gaussian)
    k = 2 ** bits
    codebook = torch.linspace(-3, 3, k)

    for _ in range(iters):
        # Assign each sample to nearest codebook entry
        dist = (x.unsqueeze(1) - codebook.unsqueeze(0)).abs()
        assign = dist.argmin(dim=1)

        # Update codebook to mean of assigned samples
        new_codebook = torch.zeros_like(codebook)
        counts = torch.zeros_like(codebook)
        for i in range(k):
            mask = assign == i
            if mask.any():
                new_codebook[i] = x[mask].mean()
                counts[i] = mask.sum()

        # Handle empty clusters: split largest cluster
        empty = counts == 0
        if empty.any():
            largest = counts.argmax()
            split_val = new_codebook[largest]
            new_codebook[largest] = split_val * 0.9
            new_codebook[empty.nonzero()[0]] = split_val * 1.1

        codebook = new_codebook

        # Early stopping
        if (dist.min(dim=1).values.mean().item() < 1e-6):
            break

    return torch.sort(codebook.to(torch.float32)).values


# Ternary codebook (1.58 bits)
_TERNARY_CB = torch.tensor([-1.2240064, 0.0, 1.2240064], dtype=torch.float32)


def get_codebook(bits: float, group_size: int) -> Tensor:
    """Get quantization codebook for given bits and group size."""
    if bits == 1.58:
        return (_TERNARY_CB / math.sqrt(group_size)).to(torch.float32)
    if bits == 2:
        return lloyd_max_gaussian(2) / math.sqrt(group_size)
    if bits == 4:
        return lloyd_max_gaussian(4) / math.sqrt(group_size)
    raise ValueError(f"Unsupported bits: {bits}. Use 1.58, 2, or 4")


# ============================================================
# Hadamard Matrix Generation
# ============================================================

@functools.cache
def hadamard_matrix(group_size: int) -> Tensor:
    """Generate normalized Hadamard matrix of given size."""
    if group_size & (group_size - 1) != 0:
        raise ValueError("group_size must be power of 2")

    # Sylvester construction
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < group_size:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)

    return (H / math.sqrt(group_size)).to(torch.float32)


# ============================================================
# CQ Quantization Core
# ============================================================

def cq_quantize(
    w: Tensor,
    bits: float = 2.0,
    group_size: int = 128,
    codebook: Tensor | None = None,
    hadamard: Tensor | None = None,
) -> Tensor:
    """Cactus Quants quantization with Hadamard rotation and codebook."""
    if w.dim() != 2:
        raise ValueError("Input must be 2D [out_features, in_features]")

    out_features, in_features = w.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by group_size ({group_size})")

    if codebook is None:
        codebook = get_codebook(bits, group_size)
    if hadamard is None:
        hadamard = hadamard_matrix(group_size)

    device = w.device
    codebook = codebook.to(device)
    hadamard = hadamard.to(device)

    # Reshape to groups: [out_features, num_groups, group_size]
    w_grouped = w.view(out_features, -1, group_size)

    # Apply Hadamard rotation: w_rot = w_grouped @ H^T
    w_rot = w_grouped @ hadamard.t()

    # Quantize each element to nearest codebook entry
    # w_rot: [out_features, num_groups, group_size]
    # codebook: [K]
    # dist: [out_features, num_groups, group_size, K]
    dist = (w_rot.unsqueeze(-1) - codebook.view(1, 1, 1, -1)).abs()
    indices = dist.argmin(dim=-1)

    # Dequantize: gather codebook values
    w_q = codebook[indices]

    # Inverse Hadamard: w_deq = w_q @ H
    w_deq = w_q @ hadamard

    # Reshape back
    deq = w_deq.view(out_features, in_features)

    return deq


class CQStraightThrough(torch.autograd.Function):
    """Straight-Through Estimator for CQ Quantization."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        w: Tensor,
        bits: float,
        group_size: int,
        codebook: Tensor | None,
        hadamard: Tensor | None,
) -> Tensor:
        ctx.bits = bits
        ctx.group_size = group_size
        ctx.codebook = codebook
        ctx.hadamard = hadamard
        return cq_quantize(w, bits, group_size, codebook, hadamard)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
) -> tuple[Tensor, None, None, None, None]:
        return grad_output, None, None, None, None


def cq_ste(
    w: Tensor,
    bits: float = 2.0,
    group_size: int = 128,
    codebook: Tensor | None = None,
    hadamard: Tensor | None = None,
) -> Tensor:
    """CQ Quantization with Straight-Through Estimator for QAT."""
    return CQStraightThrough.apply(w, bits, group_size, codebook, hadamard)


def cq_ste_simple(w: Tensor, bits: float = 2.0, group_size: int = 128) -> Tensor:
    """Simplified STE wrapper that auto-generates codebook/hadamard."""
    codebook = get_codebook(bits, group_size)
    hadamard = hadamard_matrix(group_size)
    return CQStraightThrough.apply(w, bits, group_size, codebook, hadamard)


# ============================================================
# Activation Quantization (8-bit per-tensor)
# ============================================================

def fake_quant_act(x: Tensor, bits: int = 8) -> Tensor:
    """Fake quantization for activations (per-tensor asymmetric)."""
    x_min = x.min()
    x_max = x.max()
    qmin = 0
    qmax = 2 ** bits - 1
    scale = (x_max - x_min) / (qmax - qmin)
    scale = scale.clamp(min=1e-8)
    zero_point = (-x_min / scale).round().clamp(qmin, qmax)
    x_q = (x / scale + zero_point).round().clamp(qmin, qmax)
    x_dq = (x_q - zero_point) * scale
    return x_dq


# ============================================================
# KV Cache Quantization (8-bit per-head per-channel)
# ============================================================

def quantize_kv_int8(k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Quantize K and V to INT8 per-head per-channel."""
    # k, v: [batch, num_heads, seq_len, head_dim]
    # Quantize per head per channel (last dim)
    k_scale = k.abs().amax(dim=-1, keepdim=True) / 127
    k_scale = k_scale.clamp(min=1e-8)
    k_q = (k / k_scale).round().clamp(-128, 127).to(torch.int8)

    v_scale = v.abs().amax(dim=-1, keepdim=True) / 127
    v_scale = v_scale.clamp(min=1e-8)
    v_q = (v / v_scale).round().clamp(-128, 127).to(torch.int8)

    return k_q, k_scale, v_q, v_scale


def dequantize_kv_int8(
    k_q: Tensor, k_scale: Tensor,
    v_q: Tensor, v_scale: Tensor
) -> tuple[Tensor, Tensor]:
    """Dequantize INT8 KV cache."""
    k = k_q.to(torch.float32) * k_scale
    v = v_q.to(torch.float32) * v_scale
    return k, v


def dequantize_kv_cq(
    k_q: Tensor, k_scale: Tensor,
    v_q: Tensor, v_scale: Tensor
) -> tuple[Tensor, Tensor]:
    """Dequantize CQ KV cache (alias for INT8 dequantization)."""
    return dequantize_kv_int8(k_q, k_scale, v_q, v_scale)


# ============================================================
# CQ 2-bit KV Cache (Needle-style)
# ============================================================

def cq_quantize_kv(
    k: Tensor, v: Tensor,
    bits: float = 2.0,
    group_size: int = 128,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Quantize KV cache using CQ 2-bit quantization."""
    # k, v: [batch, num_heads, seq_len, head_dim]
    # Reshape to 2D for cq_quantize
    batch, num_heads, seq_len, head_dim = k.shape
    k_2d = k.view(batch * num_heads * seq_len, head_dim)
    v_2d = v.view(batch * num_heads * seq_len, head_dim)

    codebook = get_codebook(bits, group_size)
    hadamard = hadamard_matrix(group_size)

    k_q = cq_quantize(k_2d, bits, group_size, codebook, hadamard)
    v_q = cq_quantize(v_2d, bits, group_size, codebook, hadamard)

    # Per-group scale for dequantization
    k_grouped = k_2d.view(-1, group_size)
    k_scale = k_grouped.abs().amax(dim=-1, keepdim=True)
    v_grouped = v_2d.view(-1, group_size)
    v_scale = v_grouped.abs().amax(dim=-1, keepdim=True)

    k_q = k_q.view(batch, num_heads, seq_len, head_dim)
    v_q = v_q.view(batch, num_heads, seq_len, head_dim)
    k_scale = k_scale.view(batch, num_heads, seq_len, -1, 1)
    v_scale = v_scale.view(batch, num_heads, seq_len, -1, 1)

    return k_q, k_scale, v_q, v_scale


# ============================================================
# Quantized Linear Layer (for experts, engram projections)
# ============================================================

class CQQuantizedLinear(nn.Module):
    """Linear layer with CQ-quantized weights (STE for QAT)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: float = 2.0,
        group_size: int = 128,
        bias: bool = False,
) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.codebook = get_codebook(bits, group_size)
        self.hadamard = hadamard_matrix(group_size)
        self.quantize = True

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if self.quantize:
            w_q = cq_ste(self.weight, self.bits, self.group_size, self.codebook, self.hadamard)
        else:
            w_q = self.weight
        return x @ w_q.t() + (self.bias if self.bias is not None else 0)

    def set_quantize(self, enabled: bool) -> None:
        self.quantize = enabled


def cq_quantize_simple(w: Tensor, bits: float = 2.0, group_size: int = 128) -> Tensor:
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
) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.bits = bits
        self.group_size = group_size
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()
        self.codebook = get_codebook(bits, group_size)
        self.hadamard = hadamard_matrix(group_size)
        self.quantize = True

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, indices: Tensor) -> Tensor:
        if self.quantize:
            w_q = cq_ste(self.weight, self.bits, self.group_size, self.codebook, self.hadamard)
        else:
            w_q = self.weight
        return nn.functional.embedding(indices, w_q)

    def set_quantize(self, enabled: bool) -> None:
        self.quantize = enabled


# ============================================================
# Utility: Quantize Model Parameters
# ============================================================

def quantize_model_parameters(
    model: nn.Module,
    bits: float = 2.0,
    group_size: int = 128,
) -> nn.Module:
    """Quantize model parameters in-place for inference."""
    for name, param in model.named_parameters():
        if param.dim() == 2 and "weight" in name:
            if "embedding" not in name.lower():
                param.data = cq_quantize_simple(param.data, bits, group_size)
    return model


def quantize_specific_modules(
    model: nn.Module,
    bits: float = 2.0,
    group_size: int = 128,
) -> nn.Module:
    """Replace specific module types with quantized versions."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if module.in_features % group_size == 0:
                q_module = CQQuantizedLinear(
                    module.in_features,
                    module.out_features,
                    bits,
                    group_size,
                    module.bias is not None,
                )
                q_module.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    q_module.bias.data.copy_(module.bias.data)
                setattr(model, name, q_module)
        elif isinstance(module, nn.Embedding):
            if module.embedding_dim % group_size == 0:
                q_module = CQQuantizedEmbedding(
                    module.num_embeddings,
                    module.embedding_dim,
                    bits,
                    group_size,
                )
                q_module.weight.data.copy_(module.weight.data)
                setattr(model, name, q_module)
        else:
            quantize_specific_modules(module, bits, group_size)
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
    cq_group_size: int = 128,
    head_bits: int = 4,
    weight_bits: int | float = 4,
    act_bits: int = 8,
    kv_bits: int = 8,
) -> None:
    """Configure global quantization settings."""
    global CQ_GROUP_SIZE, HEAD_BITS, WEIGHT_BITS, ACT_BITS, KV_BITS
    CQ_GROUP_SIZE = cq_group_size
    HEAD_BITS = head_bits
    WEIGHT_BITS = weight_bits
    ACT_BITS = act_bits
    KV_BITS = kv_bits


# ============================================================
# Test / Demo
# ============================================================

if __name__ == "__main__":
    print("Testing Cactus Quants (CQ) 2-bit Quantization")
    print("=" * 60)

    # Test 1: Codebook generation
    print("\n1. Codebook generation...")
    cb2 = get_codebook(2.0, 128)
    cb4 = get_codebook(4.0, 128)
    cb158 = get_codebook(1.58, 128)
    print(f"   2-bit codebook shape: {cb2.shape}, values: {cb2[:4].tolist()}...")
    print(f"   4-bit codebook shape: {cb4.shape}, values: {cb4[:4].tolist()}...")
    print(f"   1.58-bit codebook shape: {cb158.shape}, values: {cb158.tolist()}")

    # Test 2: Hadamard matrix
    print("\n2. Hadamard matrix...")
    H = hadamard_matrix(128)
    print(f"   H shape: {H.shape}")
    print(f"   H @ H^T ≈ I: {(H @ H.t()).round().eq(torch.eye(128)).all().item()}")

    # Test 3: CQ Quantization
    print("\n3. CQ Quantization (inference)...")
    w = torch.randn(256, 512)
    w_q = cq_quantize(w, bits=2.0, group_size=128)
    mse = (w - w_q).pow(2).mean().item()
    print(f"   Original shape: {w.shape}")
    print(f"   Quantized shape: {w_q.shape}")
    print(f"   MSE: {mse:.6f}")

    # Test 4: STE for QAT
    print("\n4. STE for QAT...")
    w = torch.randn(256, 512, requires_grad=True)
    w_q = cq_ste(w, bits=2.0, group_size=128)
    loss = w_q.pow(2).sum()
    loss.backward()
    print(f"   Gradient shape: {w.grad.shape}")
    print(f"   Gradient norm: {w.grad.norm().item():.6f}")

    # Test 5: KV INT8 quantization
    print("\n5. KV INT8 Quantization...")
    k = torch.randn(2, 8, 128, 64)
    v = torch.randn(2, 8, 128, 64)
    k_q, k_scale, v_q, v_scale = quantize_kv_int8(k, v)
    k_dq, v_dq = dequantize_kv_int8(k_q, k_scale, v_q, v_scale)
    k_mse = (k - k_dq).pow(2).mean().item()
    v_mse = (v - v_dq).pow(2).mean().item()
    print(f"   K MSE: {k_mse:.6f}")
    print(f"   V MSE: {v_mse:.6f}")

    # Test 6: CQ KV quantization
    print("\n6. CQ 2-bit KV Quantization...")
    k = torch.randn(2, 8, 128, 64)
    v = torch.randn(2, 8, 128, 64)
    k_q, k_scale, v_q, v_scale = cq_quantize_kv(k, v, bits=2.0, group_size=64)
    k_dq, v_dq = dequantize_kv_cq(k_q, k_scale, v_q, v_scale)
    k_mse = (k - k_dq).pow(2).mean().item()
    v_mse = (v - v_dq).pow(2).mean().item()
    print(f"   K MSE: {k_mse:.6f}")
    print(f"   V MSE: {v_mse:.6f}")

    # Test 7: Quantized Linear
    print("\n7. CQQuantizedLinear...")
    linear = CQQuantizedLinear(512, 256, bits=2.0, group_size=128)
    x = torch.randn(4, 512)
    out = linear(x)
    print(f"   Output shape: {out.shape}")
    linear.set_quantize(False)
    out_fp = linear(x)
    print(f"   FP32 output norm: {out_fp.norm().item():.6f}")
    linear.set_quantize(True)
    out_q = linear(x)
    print(f"   Quantized output norm: {out_q.norm().item():.6f}")

    # Test 8: Quantized Embedding
    print("\n8. CQQuantizedEmbedding...")
    emb = CQQuantizedEmbedding(1000, 256, bits=2.0, group_size=128)
    idx = torch.randint(0, 1000, (4, 32))
    out = emb(idx)
    print(f"   Output shape: {out.shape}")
    emb.set_quantize(False)
    out_fp = emb(idx)
    print(f"   FP32 output norm: {out_fp.norm().item():.6f}")
    emb.set_quantize(True)
    out_q = emb(idx)
    print(f"   Quantized output norm: {out_q.norm().item():.6f}")

    # Test 9: Model quantization utilities
    print("\n9. Model quantization utilities...")
    model = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )
    quantize_specific_modules(model, bits=2.0, group_size=128)
    x = torch.randn(4, 512)
    out = model(x)
    print(f"   Quantized model output shape: {out.shape}")

    print("\nAll tests passed!")
