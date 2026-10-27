"""
Batch Dequantizer for psudo-Engine.

Handles efficient batch dequantization of ternary/quantized weights
to INT8/FP16 upon VRAM entry, eliminating per-operation bit-shifting overhead
on GTX 1660 Ti (no native sub-byte ALUs).
"""

from typing import Any

import torch


class BatchDequantizer:
    """
    Batch dequantization manager for weight paging.

    When layers are paged into VRAM, their quantized weights are
    batch-dequantized once to the target compute dtype (INT8/FP16/BF16),
    avoiding repeated dequantization during forward passes.
    """

    def __init__(
        self,
        target_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None
    ):
        self.target_dtype = target_dtype
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._dequant_cache: dict[str, torch.Tensor] = {}
        self._scale_cache: dict[str, torch.Tensor] = {}

    def dequantize_layer(
        self,
        layer_tensors: dict[str, torch.Tensor],
        quantization_params: dict[str, Any] | None = None,
        layer_idx: int = 0
    ) -> dict[str, torch.Tensor]:
        """
        Dequantize all tensors in a layer batch.

        Args:
            layer_tensors: Dict of tensor_name -> quantized tensor
            quantization_params: Optional quantization parameters per tensor
            layer_idx: Layer index for cache key

        Returns:
            Dict of tensor_name -> dequantized tensor in target_dtype
        """
        dequantized = {}

        for name, tensor in layer_tensors.items():
            cache_key = f"layer_{layer_idx}_{name}"

            # Check cache first
            if cache_key in self._dequant_cache:
                dequantized[name] = self._dequant_cache[cache_key]
                continue

            # Get quantization params for this tensor
            q_params = quantization_params.get(name, {}) if quantization_params else {}
            q_type = q_params.get('type', 'ternary')

            if q_type == 'ternary':
                # Ternary dequantization
                scales = q_params.get('scales')
                if scales is not None:
                    scales_tensor = scales if isinstance(scales, torch.Tensor) else torch.tensor(scales)
                    dequant = self._dequantize_ternary(tensor, scales_tensor)
                else:
                    # No scales stored, assume already dequantized or use default
                    dequant = tensor.to(self.target_dtype)
            elif q_type == 'int8':
                # INT8 per-channel dequantization
                scales = q_params.get('scales')
                zero_points = q_params.get('zero_points')
                if scales is not None:
                    scales_tensor = scales if isinstance(scales, torch.Tensor) else torch.tensor(scales)
                    zp_tensor = zero_points if isinstance(zero_points, torch.Tensor) else torch.tensor(zero_points) if zero_points is not None else None
                    dequant = self._dequantize_int8(tensor, scales_tensor, zp_tensor)
                else:
                    dequant = tensor.to(self.target_dtype)
            else:
                # Unknown type, just convert dtype
                dequant = tensor.to(self.target_dtype)

            # Move to target device
            dequant = dequant.to(self.device)

            # Cache for reuse
            self._dequant_cache[cache_key] = dequant
            dequantized[name] = dequant

        return dequantized

    def _dequantize_ternary(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize ternary weights to target dtype."""
        # quantized: int8 with values {-1, 0, 1}
        # scales: [out_features] per-output-channel scales

        original_shape = quantized.shape
        quantized = quantized.float()
        scales = scales.float().to(quantized.device)

        if quantized.dim() >= 2:
            out_features = quantized.shape[0]
            quantized_flat = quantized.reshape(out_features, -1)
            scales = scales.reshape(out_features, 1)
        else:
            quantized_flat = quantized.reshape(1, -1)
            scales = scales.reshape(1, 1)

        dequantized = quantized_flat * scales
        dequantized = dequantized.reshape(original_shape)

        return dequantized.to(self.target_dtype)

    def _dequantize_int8(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Dequantize INT8 weights to target dtype."""
        # quantized: int8
        # scales: per-channel scales
        # zero_points: optional per-channel zero points

        original_shape = quantized.shape
        quantized = quantized.float()
        scales = scales.float().to(quantized.device)

        if zero_points is not None:
            zero_points = zero_points.float().to(quantized.device)

        if quantized.dim() >= 2:
            out_features = quantized.shape[0]
            quantized_flat = quantized.reshape(out_features, -1)
            scales = scales.reshape(out_features, 1)
            if zero_points is not None:
                zero_points = zero_points.reshape(out_features, 1)
        else:
            quantized_flat = quantized.reshape(1, -1)
            scales = scales.reshape(1, 1)
            if zero_points is not None:
                zero_points = zero_points.reshape(1, 1)

        if zero_points is not None:
            dequantized = (quantized_flat - zero_points) * scales
        else:
            dequantized = quantized_flat * scales

        dequantized = dequantized.reshape(original_shape)

        return dequantized.to(self.target_dtype)

    def clear_cache(self):
        """Clear dequantization cache (e.g., on layer eviction)."""
        self._dequant_cache.clear()
        self._scale_cache.clear()

    def evict_layer(self, layer_idx: int):
        """Remove cached dequantized tensors for a specific layer."""
        keys_to_remove = [k for k in self._dequant_cache if k.startswith(f"layer_{layer_idx}_")]
        for key in keys_to_remove:
            del self._dequant_cache[key]


class AsyncDequantizer:
    """
    Asynchronous dequantization using CUDA streams for overlap with compute.

    Allows dequantization of next layer while current layer is computing.
    """

    def __init__(
        self,
        target_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None,
        num_streams: int = 2
    ):
        self.target_dtype = target_dtype
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.streams = [torch.cuda.Stream(device=self.device) for _ in range(num_streams)]
        self.current_stream = 0
        self.pending_dequant: list[tuple[str, torch.Tensor, dict]] = []
        self.completed_dequant: dict[str, torch.Tensor] = {}

    def submit_dequantize(
        self,
        name: str,
        tensor: torch.Tensor,
        q_params: dict[str, Any]
    ):
        """Submit a tensor for async dequantization."""
        self.pending_dequant.append((name, tensor, q_params))

    def process_pending(self):
        """Process pending dequantizations on next available stream."""
        if not self.pending_dequant:
            return

        stream = self.streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % len(self.streams)

        with torch.cuda.stream(stream):
            for name, tensor, q_params in self.pending_dequant:
                q_type = q_params.get('type', 'ternary')

                if q_type == 'ternary':
                    scales = q_params.get('scales')
                    if scales is not None:
                        scales_tensor = scales if isinstance(scales, torch.Tensor) else torch.tensor(scales, device=self.device)
                        dequant = self._dequantize_ternary_async(tensor, scales_tensor)
                    else:
                        dequant = tensor.to(self.target_dtype)
                elif q_type == 'int8':
                    scales = q_params.get('scales')
                    zero_points = q_params.get('zero_points')
                    if scales is not None:
                        scales_tensor = scales if isinstance(scales, torch.Tensor) else torch.tensor(scales, device=self.device)
                        zp_tensor = zero_points if isinstance(zero_points, torch.Tensor) else torch.tensor(zero_points, device=self.device) if zero_points is not None else None
                        dequant = self._dequantize_int8_async(tensor, scales_tensor, zp_tensor)
                    else:
                        dequant = tensor.to(self.target_dtype)
                else:
                    dequant = tensor.to(self.target_dtype)

                self.completed_dequant[name] = dequant

        self.pending_dequant.clear()

    def _dequantize_ternary_async(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor
    ) -> torch.Tensor:
        original_shape = quantized.shape
        quantized = quantized.float()

        if quantized.dim() >= 2:
            out_features = quantized.shape[0]
            quantized_flat = quantized.reshape(out_features, -1)
            scales = scales.reshape(out_features, 1)
        else:
            quantized_flat = quantized.reshape(1, -1)
            scales = scales.reshape(1, 1)

        dequantized = quantized_flat * scales
        dequantized = dequantized.reshape(original_shape)

        return dequantized.to(self.target_dtype)

    def _dequantize_int8_async(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor | None = None
    ) -> torch.Tensor:
        original_shape = quantized.shape
        quantized = quantized.float()

        if quantized.dim() >= 2:
            out_features = quantized.shape[0]
            quantized_flat = quantized.reshape(out_features, -1)
            scales = scales.reshape(out_features, 1)
            if zero_points is not None:
                zero_points = zero_points.reshape(out_features, 1)
        else:
            quantized_flat = quantized.reshape(1, -1)
            scales = scales.reshape(1, 1)
            if zero_points is not None:
                zero_points = zero_points.reshape(1, 1)

        if zero_points is not None:
            dequantized = (quantized_flat - zero_points) * scales
        else:
            dequantized = quantized_flat * scales

        dequantized = dequantized.reshape(original_shape)

        return dequantized.to(self.target_dtype)

    def get_completed(self, name: str) -> torch.Tensor | None:
        """Get a completed dequantized tensor, waiting if necessary."""
        if name in self.completed_dequant:
            return self.completed_dequant.pop(name)
        return None

    def wait_all(self):
        """Wait for all streams to complete."""
        for stream in self.streams:
            stream.synchronize()

    def clear_completed(self):
        """Clear completed dequantizations."""
        self.completed_dequant.clear()


def create_dequantizer_from_config(config: dict) -> BatchDequantizer:
    """Create BatchDequantizer from config dict."""
    dtype_str = config.get('target_dtype', 'bfloat16')
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'int8': torch.int8,
    }
    target_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    device_str = config.get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    return BatchDequantizer(target_dtype=target_dtype, device=device)


if __name__ == "__main__":
    # Quick test

    dequantizer = BatchDequantizer(target_dtype=torch.bfloat16)

    # Simulate ternary quantized layer tensors
    weight = torch.randn(512, 512)
    quantized, scales = weight.sign().to(torch.int8), weight.abs().mean(dim=1)

    layer_tensors = {
        'attn.q_proj.weight': quantized,
        'attn.k_proj.weight': quantized.clone(),
        'attn.v_proj.weight': quantized.clone(),
        'mlp.w1.weight': quantized[:2048],
    }

    quantization_params = {
        'attn.q_proj.weight': {'type': 'ternary', 'scales': scales},
        'attn.k_proj.weight': {'type': 'ternary', 'scales': scales},
        'attn.v_proj.weight': {'type': 'ternary', 'scales': scales},
        'mlp.w1.weight': {'type': 'ternary', 'scales': scales[:2048]},
    }

    # Dequantize
    dequantized = dequantizer.dequantize_layer(layer_tensors, quantization_params, layer_idx=0)

    for _name, tensor in dequantized.items():
        assert tensor.dtype == torch.bfloat16

    # Test cache hit
    dequantized2 = dequantizer.dequantize_layer(layer_tensors, quantization_params, layer_idx=0)
    assert dequantized['attn.q_proj.weight'] is dequantized2['attn.q_proj.weight']

    # Test eviction
    dequantizer.evict_layer(0)
    assert 'layer_0_attn.q_proj.weight' not in dequantizer._dequant_cache

