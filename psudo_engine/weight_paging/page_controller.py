"""
Page Controller for psudo-Engine Weight Paging.

Manages sequential layer loading/eviction from disk to VRAM with LRU eviction policy.
Integrates with WeightStore and BatchDequantizer for efficient weight paging.
"""

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from .dequantizer import BatchDequantizer
from .weight_store import WeightStore


class PageController:
    """
    Runtime controller for sequential layer loading with LRU eviction.

    Manages the active layer buffer in VRAM, loading layers on-demand
    and evicting least recently used layers when VRAM budget is exceeded.
    """

    def __init__(
        self,
        weight_store: WeightStore,
        dequantizer: BatchDequantizer,
        max_vram_bytes: int = 2_000_000_000,  # 2GB budget for active layers
        device: torch.device | None = None
    ):
        self.weight_store = weight_store
        self.dequantizer = dequantizer
        self.max_vram_bytes = max_vram_bytes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # LRU cache: layer_idx -> {tensor_name: tensor}
        self._active_layers: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self._layer_sizes: dict[int, int] = {}  # layer_idx -> VRAM bytes
        self._current_vram = 0

        # Statistics
        self.load_count = 0
        self.evict_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def load_layer(
        self,
        layer_idx: int,
        quantization_params: dict[str, Any] | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Load a layer into VRAM, evicting if necessary.

        Args:
            layer_idx: Layer index to load
            quantization_params: Optional quantization parameters

        Returns:
            Dict of tensor_name -> dequantized tensor on device
        """
        # Check if already in cache
        if layer_idx in self._active_layers:
            # Move to end (most recently used)
            self._active_layers.move_to_end(layer_idx)
            self.cache_hits += 1
            return self._active_layers[layer_idx]

        self.cache_misses += 1

        # Load from disk
        layer_tensors = self.weight_store.load_layer(layer_idx, device=self.device)

        if not layer_tensors:
            raise ValueError(f"Layer {layer_idx} not found in weight store")

        # Dequantize
        dequantized = self.dequantizer.dequantize_layer(
            layer_tensors, quantization_params, layer_idx
        )

        # Calculate VRAM usage
        layer_size = sum(t.numel() * t.element_size() for t in dequantized.values())

        # Evict if needed
        self._ensure_space(layer_size)

        # Add to cache
        self._active_layers[layer_idx] = dequantized
        self._layer_sizes[layer_idx] = layer_size
        self._current_vram += layer_size
        self.load_count += 1

        return dequantized

    def _ensure_space(self, needed_bytes: int):
        """Evict LRU layers until enough space is available."""
        while self._current_vram + needed_bytes > self.max_vram_bytes and self._active_layers:
            self._evict_lru()

    def _evict_lru(self) -> int:
        """Evict least recently used layer."""
        if not self._active_layers:
            return 0

        lru_layer_idx, lru_tensors = self._active_layers.popitem(last=False)
        layer_size = self._layer_sizes.pop(lru_layer_idx)
        self._current_vram -= layer_size
        self.evict_count += 1

        # Clear dequantizer cache for this layer
        self.dequantizer.evict_layer(lru_layer_idx)

        # Evict from disk store if configured
        # self.weight_store.evict_layer(lru_layer_idx)  # Optional

        return layer_size

    def get_layer(self, layer_idx: int) -> dict[str, torch.Tensor] | None:
        """Get layer if in cache, None otherwise (no loading)."""
        if layer_idx in self._active_layers:
            self._active_layers.move_to_end(layer_idx)
            self.cache_hits += 1
            return self._active_layers[layer_idx]
        self.cache_misses += 1
        return None

    def forward_layer(
        self,
        layer_idx: int,
        x: torch.Tensor,
        layer_module: nn.Module,
        quantization_params: dict[str, Any] | None = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Load layer, run forward pass, optionally keep in cache.

        This is the main entry point for sequential layer execution.
        """
        # Load layer weights
        layer_weights = self.load_layer(layer_idx, quantization_params)

        # Temporarily replace module parameters with loaded weights
        # This avoids copying - we use the loaded tensors directly
        original_params = {}
        for name, param in layer_module.named_parameters():
            if name in layer_weights:
                original_params[name] = param.data
                param.data = layer_weights[name]

        try:
            # Run forward pass
            output = layer_module(x, **kwargs)
        finally:
            # Restore original parameters (or keep loaded ones in module)
            for name, param in layer_module.named_parameters():
                if name in original_params:
                    param.data = original_params[name]

        return output

    def prefetch_next(self, layer_idx: int, quantization_params: dict[str, Any] | None = None):
        """Prefetch next layer in background (if using async dequantizer)."""
        # For synchronous dequantizer, just load
        if layer_idx not in self._active_layers:
            self.load_layer(layer_idx, quantization_params)

    def get_memory_stats(self) -> dict[str, Any]:
        """Get current memory usage statistics."""
        return {
            'active_layers': len(self._active_layers),
            'current_vram_bytes': self._current_vram,
            'current_vram_mb': self._current_vram / (1024 * 1024),
            'max_vram_mb': self.max_vram_bytes / (1024 * 1024),
            'vram_utilization': self._current_vram / self.max_vram_bytes if self.max_vram_bytes > 0 else 0,
            'layer_indices': list(self._active_layers.keys()),
            'layer_sizes_mb': {k: v / (1024 * 1024) for k, v in self._layer_sizes.items()},
            'load_count': self.load_count,
            'evict_count': self.evict_count,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }

    def clear(self):
        """Clear all active layers."""
        self._active_layers.clear()
        self._layer_sizes.clear()
        self._current_vram = 0
        self.dequantizer.clear_cache()

    def get_active_layer_indices(self) -> list[int]:
        """Get list of currently loaded layer indices (MRU first)."""
        return list(reversed(list(self._active_layers.keys())))


class SequentialPageController(PageController):
    """
    Specialized controller for sequential forward passes.

    Optimized for the case where layers are executed in order (0, 1, 2, ...),
    with automatic prefetching of the next layer.
    """

    def __init__(self, *args, prefetch_distance: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefetch_distance = prefetch_distance
        self._last_layer = -1

    def forward_layer(
        self,
        layer_idx: int,
        x: torch.Tensor,
        layer_module: nn.Module,
        quantization_params: dict[str, Any] | None = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward with automatic next-layer prefetch."""
        # Prefetch next layer if sequential
        if layer_idx == self._last_layer + 1 and self.prefetch_distance > 0:
            next_idx = layer_idx + 1
            if self.weight_store.layer_exists(next_idx):
                # Non-blocking prefetch
                self.prefetch_next(next_idx, quantization_params)

        self._last_layer = layer_idx
        return super().forward_layer(layer_idx, x, layer_module, quantization_params, **kwargs)

    def reset_sequence(self):
        """Reset sequence tracking (e.g., at sequence boundary)."""
        self._last_layer = -1


class MemoryAwarePageController(PageController):
    """
    Page controller that monitors actual GPU memory and adjusts.

    Uses torch.cuda.memory_allocated() to track real VRAM usage
    and evicts proactively before OOM.
    """

    def __init__(
        self,
        *args,
        safety_margin_bytes: int = 512 * 1024 * 1024,  # 512MB safety margin
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.safety_margin = safety_margin_bytes
        self._peak_vram = 0

    def _get_actual_vram(self) -> int:
        """Get actual GPU memory allocated."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device)
        return self._current_vram

    def _ensure_space(self, needed_bytes: int):
        """Evict based on actual VRAM usage."""
        actual_vram = self._get_actual_vram()
        available = self.max_vram_bytes - actual_vram - self.safety_margin

        while needed_bytes > available and self._active_layers:
            self._evict_lru()
            actual_vram = self._get_actual_vram()
            available = self.max_vram_bytes - actual_vram - self.safety_margin

    def get_memory_stats(self) -> dict[str, Any]:
        stats = super().get_memory_stats()
        if torch.cuda.is_available():
            actual = torch.cuda.memory_allocated(self.device)
            stats['actual_vram_bytes'] = actual
            stats['actual_vram_mb'] = actual / (1024 * 1024)
            stats['peak_vram_mb'] = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        return stats


def create_page_controller(
    weight_store: WeightStore,
    dequantizer: BatchDequantizer,
    config: dict
) -> PageController:
    """Create page controller from config."""
    controller_type = config.get('type', 'sequential')
    max_vram_mb = config.get('max_vram_mb', 2048)
    max_vram_bytes = max_vram_mb * 1024 * 1024

    if controller_type == 'memory_aware':
        return MemoryAwarePageController(
            weight_store, dequantizer, max_vram_bytes,
            safety_margin_bytes=config.get('safety_margin_mb', 512) * 1024 * 1024
        )
    elif controller_type == 'sequential':
        return SequentialPageController(
            weight_store, dequantizer, max_vram_bytes,
            prefetch_distance=config.get('prefetch_distance', 1)
        )
    else:
        return PageController(weight_store, dequantizer, max_vram_bytes)


if __name__ == "__main__":
    # Quick test with mock objects

    import tempfile

    from .dequantizer import BatchDequantizer
    from .weight_store import WeightStore

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create weight store with test data
        store = WeightStore(tmpdir, "test_model")

        # Create test layers
        for layer_idx in range(5):
            tensors = {
                'attn.q_proj.weight': torch.randn(512, 512, dtype=torch.bfloat16),
                'attn.k_proj.weight': torch.randn(512, 512, dtype=torch.bfloat16),
                'mlp.w1.weight': torch.randn(2048, 512, dtype=torch.bfloat16),
            }
            store.save_layer(layer_idx, tensors)

        # Create dequantizer and controller
        dequantizer = BatchDequantizer(target_dtype=torch.bfloat16)
        controller = PageController(store, dequantizer, max_vram_bytes=500_000_000)  # 500MB

        # Load some layers
        layer0 = controller.load_layer(0)
        layer1 = controller.load_layer(1)
        layer2 = controller.load_layer(2)


        # Load more to trigger eviction
        layer3 = controller.load_layer(3)
        layer4 = controller.load_layer(4)


        # Test cache hit
        layer0_again = controller.load_layer(0)

