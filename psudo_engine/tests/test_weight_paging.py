"""
Unit tests for Weight Paging Subsystem.
"""

import tempfile

import pytest
import torch

from psudo_engine.weight_paging.dequantizer import BatchDequantizer
from psudo_engine.weight_paging.page_controller import PageController
from psudo_engine.weight_paging.ternary import (
    TernaryLinear,
    pack_ternary_weights,
    ternary_dequantize,
    ternary_quantize,
    unpack_ternary_weights,
)
from psudo_engine.weight_paging.weight_store import WeightStore


class TestWeightStore:
    """Tests for WeightStore."""

    def test_save_and_load_layer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WeightStore(tmpdir, "test_model")

            tensors = {
                'attn.q_proj.weight': torch.randn(512, 512, dtype=torch.bfloat16),
                'attn.k_proj.weight': torch.randn(512, 512, dtype=torch.bfloat16),
                'mlp.w1.weight': torch.randn(2048, 512, dtype=torch.bfloat16),
            }

            store.save_layer(0, tensors)
            loaded = store.load_layer(0)

            for name, orig in tensors.items():
                assert torch.allclose(orig, loaded[name]), f"Mismatch in {name}"

    def test_load_single_tensor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WeightStore(tmpdir, "test_model")

            tensor = torch.randn(256, 512, dtype=torch.bfloat16)
            store.save_layer(0, {'test_tensor': tensor})

            loaded = store.load_tensor(0, 'test_tensor')
            assert torch.allclose(tensor, loaded)

    def test_layer_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WeightStore(tmpdir, "test_model")

            assert not store.layer_exists(0)

            store.save_layer(0, {'w': torch.randn(10, 10)})
            assert store.layer_exists(0)
            assert not store.layer_exists(1)

    def test_evict_layer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WeightStore(tmpdir, "test_model")

            store.save_layer(0, {'w': torch.randn(100, 100)})
            freed = store.evict_layer(0)

            assert freed > 0
            assert not store.layer_exists(0)

    def test_memory_usage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WeightStore(tmpdir, "test_model")

            tensors = {'w': torch.randn(512, 512, dtype=torch.bfloat16)}
            store.save_layer(0, tensors)
            store.save_layer(1, tensors)

            usage = store.get_memory_usage()
            assert usage['num_layers'] == 2
            assert usage['total_size_bytes'] > 0

    def test_verify_integrity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WeightStore(tmpdir, "test_model")

            tensors = {'w': torch.randn(64, 64, dtype=torch.bfloat16)}
            store.save_layer(0, tensors)

            valid, errors = store.verify_integrity()
            assert valid
            assert len(errors) == 0


class TestTernaryQuantization:
    """Tests for ternary quantization (BitNet 1.58b)."""

    def test_quantize_dequantize(self):
        weight = torch.randn(256, 512, dtype=torch.bfloat16)

        quantized, scales = ternary_quantize(weight)
        ternary_dequantize(quantized, scales)

        # Check quantized values are ternary
        unique_vals = torch.unique(quantized)
        assert set(unique_vals.tolist()).issubset({-1, 0, 1})

        # Check shapes
        assert quantized.shape == weight.shape
        assert scales.shape == (weight.shape[0],)

    def test_ste_gradient(self):
        weight = torch.randn(64, 128, dtype=torch.float32, requires_grad=True)

        from psudo_engine.weight_paging.ternary import ternary_quantize_ste
        quantized, scales = ternary_quantize_ste(weight)
        dequantized = ternary_dequantize(quantized, scales)

        loss = dequantized.sum()
        loss.backward()

        assert weight.grad is not None
        assert weight.grad.shape == weight.shape

    def test_ternary_linear(self):
        layer = TernaryLinear(128, 256)
        x = torch.randn(2, 32, 128)

        # Training mode
        out = layer(x)
        assert out.shape == (2, 32, 256)

        # Inference mode
        layer.prepare_for_inference()
        out = layer(x)
        assert out.shape == (2, 32, 256)

    def test_packing(self):
        quantized = torch.randint(-1, 2, (1000,), dtype=torch.int8)

        packed = pack_ternary_weights(quantized)
        unpacked = unpack_ternary_weights(packed, quantized.numel())

        assert torch.equal(quantized, unpacked)
        # Check compression (4 weights per byte)
        assert packed.numel() == (quantized.numel() + 3) // 4

    def test_compression_ratio(self):
        from psudo_engine.weight_paging.ternary import compute_ternary_compression_ratio

        ratio = compute_ternary_compression_ratio(16)  # FP16
        assert ratio > 9  # ~9x compression for ternary vs FP16


class TestBatchDequantizer:
    """Tests for BatchDequantizer."""

    def test_dequantize_ternary_layer(self):
        dequantizer = BatchDequantizer(target_dtype=torch.bfloat16)

        # Create ternary quantized tensors
        weight = torch.randn(512, 512)
        quantized, scales = ternary_quantize(weight)

        layer_tensors = {
            'attn.q_proj.weight': quantized,
            'attn.k_proj.weight': quantized.clone(),
        }

        quantization_params = {
            'attn.q_proj.weight': {'type': 'ternary', 'scales': scales},
            'attn.k_proj.weight': {'type': 'ternary', 'scales': scales},
        }

        dequantized = dequantizer.dequantize_layer(layer_tensors, quantization_params, layer_idx=0)

        for name, tensor in dequantized.items():
            assert tensor.dtype == torch.bfloat16
            assert tensor.shape == layer_tensors[name].shape

    def test_cache_hit(self):
        dequantizer = BatchDequantizer(target_dtype=torch.bfloat16)

        weight = torch.randn(256, 256)
        quantized, scales = ternary_quantize(weight)

        layer_tensors = {'test.weight': quantized}
        quantization_params = {'test.weight': {'type': 'ternary', 'scales': scales}}

        # First call - cache miss
        result1 = dequantizer.dequantize_layer(layer_tensors, quantization_params, layer_idx=0)

        # Second call - cache hit
        result2 = dequantizer.dequantize_layer(layer_tensors, quantization_params, layer_idx=0)

        assert result1['test.weight'] is result2['test.weight']

    def test_evict_layer(self):
        dequantizer = BatchDequantizer(target_dtype=torch.bfloat16)

        weight = torch.randn(256, 256)
        quantized, scales = ternary_quantize(weight)

        layer_tensors = {'test.weight': quantized}
        quantization_params = {'test.weight': {'type': 'ternary', 'scales': scales}}

        dequantizer.dequantize_layer(layer_tensors, quantization_params, layer_idx=0)
        assert 'layer_0_test.weight' in dequantizer._dequant_cache

        dequantizer.evict_layer(0)
        assert 'layer_0_test.weight' not in dequantizer._dequant_cache


class TestPageController:
    """Tests for PageController."""

    def test_load_layer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WeightStore(tmpdir, "test_model")

            tensors = {'w': torch.randn(256, 256, dtype=torch.bfloat16)}
            store.save_layer(0, tensors)
            store.save_layer(1, tensors)

            dequantizer = BatchDequantizer()
            controller = PageController(store, dequantizer, max_vram_bytes=10_000_000)

            loaded = controller.load_layer(0)
            assert 'w' in loaded
            assert loaded['w'].dtype == torch.bfloat16

    def test_lru_eviction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WeightStore(tmpdir, "test_model")

            # Create multiple layers
            for i in range(5):
                store.save_layer(i, {'w': torch.randn(1000, 1000, dtype=torch.bfloat16)})

            dequantizer = BatchDequantizer()
            # Small VRAM budget to force eviction
            controller = PageController(store, dequantizer, max_vram_bytes=5_000_000)

            # Load layers sequentially
            for i in range(5):
                controller.load_layer(i)

            stats = controller.get_memory_stats()
            # Should have evicted some layers
            assert stats['active_layers'] < 5
            assert stats['evict_count'] > 0

    def test_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WeightStore(tmpdir, "test_model")
            store.save_layer(0, {'w': torch.randn(256, 256, dtype=torch.bfloat16)})

            dequantizer = BatchDequantizer()
            controller = PageController(store, dequantizer)

            controller.load_layer(0)
            controller.load_layer(0)  # Second access

            stats = controller.get_memory_stats()
            assert stats['cache_hits'] == 1

    def test_get_memory_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WeightStore(tmpdir, "test_model")
            store.save_layer(0, {'w': torch.randn(256, 256, dtype=torch.bfloat16)})

            dequantizer = BatchDequantizer()
            controller = PageController(store, dequantizer)

            controller.load_layer(0)
            stats = controller.get_memory_stats()

            assert 'active_layers' in stats
            assert 'current_vram_mb' in stats
            assert 'hit_rate' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
