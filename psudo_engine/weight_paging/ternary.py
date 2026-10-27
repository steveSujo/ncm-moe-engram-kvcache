"""
Ternary Quantization (BitNet 1.58b) for psudo-Engine.

Implements ternary weight quantization with values {-1, 0, +1} and
per-channel scaling factors following BitNet 1.58b methodology.
Uses Straight-Through Estimator (STE) for gradient flow during training.
"""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ternary quantization codebook (BitNet 1.58b optimal scaling)
TERNARY_VALUES = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
TERNARY_SCALE = 1.58  # Approximate bits per weight for ternary


def ternary_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weight tensor to ternary values {-1, 0, +1} with per-channel scaling.

    Following BitNet 1.58b: w_q = round(weight / scale) clipped to {-1, 0, 1}
    where scale = mean(|weight|) per output channel.

    Args:
        weight: Input weight tensor [..., out_features, in_features]

    Returns:
        quantized: Ternary quantized weights (int8 with values -1, 0, 1)
        scales: Per-channel scaling factors [out_features]
    """
    original_shape = weight.shape
    weight = weight.float()  # Compute in FP32 for numerical stability

    # Reshape to [out_features, -1] for per-output-channel scaling
    if weight.dim() >= 2:
        out_features = weight.shape[0]
        weight_flat = weight.reshape(out_features, -1)
    else:
        weight_flat = weight.reshape(1, -1)
        out_features = 1

    # Compute per-channel scale: mean of absolute values
    # This is the BitNet 1.58b scaling factor
    scale = weight_flat.abs().mean(dim=1, keepdim=True)  # [out_features, 1]
    scale = scale.clamp(min=1e-6)  # Avoid division by zero

    # Quantize: round(weight / scale) and clip to {-1, 0, 1}
    quantized = (weight_flat / scale).round().clamp(-1, 1)

    # Convert to int8 for storage efficiency
    quantized = quantized.to(torch.int8)
    scale = scale.squeeze(1)  # [out_features]

    # Restore original shape for quantized weights
    quantized = quantized.reshape(original_shape)

    return quantized, scale


def ternary_dequantize(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Dequantize ternary weights back to float.

    Args:
        quantized: Ternary quantized weights (int8, values -1, 0, 1)
        scales: Per-channel scaling factors [out_features]

    Returns:
        Dequantized weight tensor (float)
    """
    original_shape = quantized.shape
    quantized = quantized.float()
    scales = scales.float()

    if quantized.dim() >= 2:
        out_features = quantized.shape[0]
        quantized_flat = quantized.reshape(out_features, -1)
        scales = scales.reshape(out_features, 1)
    else:
        quantized_flat = quantized.reshape(1, -1)
        scales = scales.reshape(1, 1)

    dequantized = quantized_flat * scales
    dequantized = dequantized.reshape(original_shape)

    return dequantized


class TernaryQuantFunction(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for ternary quantization.

    Forward: quantize to ternary
    Backward: pass gradients through as if identity (with scale)
    """

    @staticmethod
    def forward(ctx: Any, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quantized, scales = ternary_quantize(weight)
        ctx.save_for_backward(weight, scales)
        return quantized, scales

    @staticmethod
    def backward(ctx: Any, grad_quantized: torch.Tensor, grad_scales: torch.Tensor) -> torch.Tensor:
        weight, scales = ctx.saved_tensors

        # STE: gradient flows through as if weight = quantized * scale
        # But we need to account for the scaling
        # grad_weight = grad_quantized * scale (broadcasted)

        original_shape = weight.shape
        if weight.dim() >= 2:
            out_features = weight.shape[0]
            scales_broadcast = scales.reshape(out_features, 1)
        else:
            scales_broadcast = scales.reshape(1, 1)

        # Gradient wrt weight: STE passes through with scale
        grad_weight = grad_quantized.float() * scales_broadcast
        grad_weight = grad_weight.reshape(original_shape)

        return grad_weight


def ternary_quantize_ste(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ternary quantization with Straight-Through Estimator for training.

    Returns quantized weights and scales, with STE gradient support.
    """
    return TernaryQuantFunction.apply(weight)


class TernaryLinear(nn.Module):
    """
    Linear layer with ternary quantized weights (BitNet 1.58b style).

    During training: uses STE for gradient flow
    During inference: uses pre-quantized weights
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Full precision weights for training (STE)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter('bias', None)

        # Quantized weights for inference (populated by prepare_for_inference)
        self.register_buffer('weight_quantized', None, persistent=False)
        self.register_buffer('weight_scales', None, persistent=False)
        self.inference_mode = False

        # Initialize
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def prepare_for_inference(self):
        """Quantize weights for inference mode."""
        self.inference_mode = True
        with torch.no_grad():
            quantized, scales = ternary_quantize(self.weight)
            self.weight_quantized = quantized
            self.weight_scales = scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inference_mode and self.weight_quantized is not None:
            # Inference: use pre-quantized weights
            weight = ternary_dequantize(self.weight_quantized, self.weight_scales)
            weight = weight.to(x.dtype)
        else:
            # Training: use STE
            weight_quantized, scales = ternary_quantize_ste(self.weight)
            weight = ternary_dequantize(weight_quantized, scales)
            weight = weight.to(x.dtype)

        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def pack_ternary_weights(quantized: torch.Tensor) -> torch.Tensor:
    """
    Pack ternary weights ({-1, 0, 1}) into 2-bit format for storage.

    Each weight uses 2 bits: 00=0, 01=+1, 10=-1, 11=reserved
    4 weights per byte.

    Args:
        quantized: Ternary quantized weights (int8 with values -1, 0, 1)

    Returns:
        Packed uint8 tensor (4x compression)
    """
    # Flatten
    flat = quantized.flatten().to(torch.int8)
    num_weights = flat.numel()

    # Pack 4 weights per byte
    packed_size = (num_weights + 3) // 4
    packed = torch.zeros(packed_size, dtype=torch.uint8)

    for i in range(0, num_weights, 4):
        byte_val = 0
        for j in range(4):
            if i + j < num_weights:
                w = flat[i + j].item()
                # 00=0, 01=+1, 10=-1
                if w == 1:
                    bits = 0b01
                elif w == -1:
                    bits = 0b10
                else:  # w == 0
                    bits = 0b00
                byte_val |= (bits << (j * 2))
        packed[i // 4] = byte_val

    return packed


def unpack_ternary_weights(packed: torch.Tensor, num_weights: int) -> torch.Tensor:
    """
    Unpack 2-bit ternary weights back to int8.

    Args:
        packed: Packed uint8 tensor
        num_weights: Number of weights to unpack

    Returns:
        Ternary quantized weights (int8 with values -1, 0, 1)
    """
    flat = packed.flatten()
    unpacked = torch.zeros(num_weights, dtype=torch.int8)

    for i in range(num_weights):
        byte_idx = i // 4
        bit_pos = (i % 4) * 2
        byte_val = flat[byte_idx].item()
        bits = (byte_val >> bit_pos) & 0b11

        if bits == 0b01:
            unpacked[i] = 1
        elif bits == 0b10:
            unpacked[i] = -1
        else:  # 0b00 or 0b11
            unpacked[i] = 0

    return unpacked


def compute_ternary_compression_ratio(original_bits: int = 16) -> float:
    """Compute compression ratio for ternary vs original precision."""
    # Ternary uses ~1.58 bits per weight (log2(3))
    ternary_bits = math.log2(3)
    return original_bits / ternary_bits


if __name__ == "__main__":
    # Quick test

    # Test quantization/dequantization
    weight = torch.randn(256, 512, dtype=torch.bfloat16)
    quantized, scales = ternary_quantize(weight)
    dequantized = ternary_dequantize(quantized, scales)

    # Check values are ternary
    unique_vals = torch.unique(quantized)
    assert set(unique_vals.tolist()).issubset({-1, 0, 1})

    # Check compression
    ratio = compute_ternary_compression_ratio(16)

    # Test STE
    weight_ste = torch.randn(64, 128, dtype=torch.float32, requires_grad=True)
    quantized_ste, scales_ste = ternary_quantize_ste(weight_ste)
    dequantized_ste = ternary_dequantize(quantized_ste, scales_ste)
    loss = dequantized_ste.sum()
    loss.backward()
    assert weight_ste.grad is not None

    # Test TernaryLinear
    layer = TernaryLinear(128, 256)
    x = torch.randn(2, 32, 128)
    out = layer(x)

    # Test inference mode
    layer.prepare_for_inference()
    out_inf = layer(x)

    # Test packing
    packed = pack_ternary_weights(quantized)
    unpacked = unpack_ternary_weights(packed, quantized.numel())
    assert torch.equal(quantized.flatten(), unpacked)

