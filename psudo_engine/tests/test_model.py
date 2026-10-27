"""
Unit tests for Model Architecture.
"""

import pytest
import torch
from omegaconf import OmegaConf

from psudo_engine.model.attention import CausalAttention, KVCache
from psudo_engine.model.blocks import (
    MultiHeadAttention,
    RMSNorm,
    SwiGLU,
    TransformerBlock,
    apply_rotary_emb,
    precompute_rope_cos_sin,
)
from psudo_engine.model.engram import create_engram_module
from psudo_engine.model.moe import MoEConfig, MoELayer, TopKRouter
from psudo_engine.model.psudo_transformer import create_model


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_forward(self):
        norm = RMSNorm(512)
        x = torch.randn(2, 32, 512)
        out = norm(x)

        assert out.shape == x.shape
        # RMS should be close to 1
        rms = out.float().pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_with_weight(self):
        norm = RMSNorm(256, elementwise_affine=True)
        x = torch.randn(1, 16, 256)
        out = norm(x)
        assert out.shape == x.shape


class TestRoPE:
    """Tests for Rotary Positional Embeddings."""

    def test_precompute(self):
        cos, sin = precompute_rope_cos_sin(128, 64)

        assert cos.shape == (128, 64)
        assert sin.shape == (128, 64)

        # Check that cos^2 + sin^2 = 1
        assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos))

    def test_apply_rotary_emb(self):
        x = torch.randn(2, 8, 16, 64)  # [B, H, T, D]
        cos, sin = precompute_rope_cos_sin(16, 64)

        out = apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""

    def test_forward(self):
        attn = MultiHeadAttention(512, 8, 64)
        x = torch.randn(2, 32, 512)
        cos, sin = precompute_rope_cos_sin(32, 64)

        out = attn(x, cos, sin)
        assert out.shape == x.shape

    def test_with_mask(self):
        attn = MultiHeadAttention(256, 4, 64)
        x = torch.randn(1, 16, 256)
        cos, sin = precompute_rope_cos_sin(16, 64)

        # Causal mask
        mask = torch.triu(torch.ones(16, 16), diagonal=1).bool()
        mask = ~mask
        mask = mask.unsqueeze(0).unsqueeze(0)

        out = attn(x, cos, sin, mask=mask)
        assert out.shape == x.shape

    def test_prepare_for_inference(self):
        attn = MultiHeadAttention(256, 4, 64, use_ternary=True)
        attn.prepare_for_inference()
        assert attn.inference_mode


class TestSwiGLU:
    """Tests for SwiGLU."""

    def test_forward(self):
        ffn = SwiGLU(256, 1024)
        x = torch.randn(2, 32, 256)
        out = ffn(x)

        assert out.shape == x.shape

    def test_prepare_for_inference(self):
        ffn = SwiGLU(256, 1024, use_ternary=True)
        ffn.prepare_for_inference()
        assert ffn.inference_mode


class TestTransformerBlock:
    """Tests for TransformerBlock."""

    def test_forward(self):
        block = TransformerBlock(256, 4, 64, 1024)
        x = torch.randn(2, 32, 256)
        cos, sin = precompute_rope_cos_sin(32, 64)

        out = block(x, cos, sin)
        assert out.shape == x.shape

    def test_with_engram(self):
        block = TransformerBlock(256, 4, 64, 1024, has_engram=True)
        x = torch.randn(2, 32, 256)
        cos, sin = precompute_rope_cos_sin(32, 64)
        engram = torch.randn(2, 32, 256)

        out = block(x, cos, sin, engram=engram)
        assert out.shape == x.shape

    def test_prepare_for_inference(self):
        block = TransformerBlock(256, 4, 64, 1024, use_ternary=True)
        block.prepare_for_inference()
        assert block.inference_mode


class TestCausalAttention:
    """Tests for CausalAttention with KV Cache."""

    def test_forward(self):
        attn = CausalAttention(256, 4, 64, max_window=128)
        x = torch.randn(1, 1, 256)

        out = attn(x)
        assert out.shape == x.shape

    def test_kv_cache(self):
        attn = CausalAttention(256, 4, 64, max_window=128)
        attn.prepare_for_inference()

        # First token
        x = torch.randn(1, 1, 256)
        attn(x)
        assert attn.kv_cache.cache_len == 1

        # Second token
        x2 = torch.randn(1, 1, 256)
        attn(x2)
        assert attn.kv_cache.cache_len == 2

    def test_reset_cache(self):
        attn = CausalAttention(256, 4, 64, max_window=128)
        attn.prepare_for_inference()

        x = torch.randn(1, 1, 256)
        attn(x)
        assert attn.kv_cache.cache_len == 1

        attn.reset_cache()
        assert attn.kv_cache.cache_len == 0


class TestKVCache:
    """Tests for BoundedKVCache."""

    def test_update(self):
        cache = KVCache(max_window=4, sink_tokens=1)

        # Add tokens
        k = torch.randn(1, 2, 1, 16)
        v = torch.randn(1, 2, 1, 16)

        k_full, v_full = cache.update(k, v)
        assert cache.cache_len == 1

        # Add more
        for _ in range(3):
            k = torch.randn(1, 2, 1, 16)
            v = torch.randn(1, 2, 1, 16)
            k_full, v_full = cache.update(k, v)

        assert cache.cache_len == 4  # Max window

    def test_sliding_window(self):
        cache = KVCache(max_window=4, sink_tokens=1)

        # Fill beyond window
        for _i in range(6):
            k = torch.randn(1, 2, 1, 16)
            v = torch.randn(1, 2, 1, 16)
            k_full, v_full = cache.update(k, v)

        # Should keep sink + recent window
        assert cache.cache_len == 4
        # Sink token should be first
        assert k_full[0, 0, 0, 0] == cache.k_cache[0, 0, 0, 0]


class TestMoE:
    """Tests for Mixture of Experts."""

    def test_topk_router(self):
        router = TopKRouter(256, 8, top_k=2)
        x = torch.randn(2, 32, 256)

        weights, indices, logits = router(x)

        assert weights.shape == (2, 32, 2)
        assert indices.shape == (2, 32, 2)
        assert logits.shape == (2, 32, 8)

        # Weights should sum to 1
        assert torch.allclose(weights.sum(-1), torch.ones(2, 32))

    def test_router_losses(self):
        router = TopKRouter(256, 8)
        x = torch.randn(2, 32, 256)

        router(x)
        losses = router.get_losses()

        assert 'aux_loss' in losses
        assert 'z_loss' in losses

    def test_moe_layer(self):
        config = MoEConfig(d_model=256, n_experts=4, top_k=2, expert_hidden=512)
        moe = MoELayer(config, layer_idx=0)

        x = torch.randn(2, 16, 256)
        out = moe(x)

        assert out.shape == x.shape

    def test_moe_losses(self):
        config = MoEConfig(d_model=256, n_experts=4)
        moe = MoELayer(config, layer_idx=0)

        x = torch.randn(2, 16, 256)
        _ = moe(x)

        losses = moe.get_router_losses()
        assert 'aux_loss' in losses
        assert 'z_loss' in losses


class TestEngram:
    """Tests for Engram Module."""

    def test_forward(self):
        config = {
            'd_model': 256,
            'engram_slots': 1024,
            'engram_heads': 2,
            'engram_orders': [2, 3],
            'engram_gating': True,
            'use_ternary': False,
        }

        engram = create_engram_module(config)
        x = torch.randn(2, 32, 256)
        tokens = torch.randint(0, 1000, (2, 32))

        out = engram(x, tokens)
        assert out.shape == x.shape

    def test_prepare_for_inference(self):
        config = {
            'd_model': 256,
            'engram_slots': 1024,
            'engram_heads': 2,
            'engram_orders': [2, 3],
            'use_ternary': True,
        }

        engram = create_engram_module(config)
        engram.prepare_for_inference()
        assert engram.inference_mode


class TestPsudoTransformer:
    """Tests for PsudoTransformer."""

    def test_forward(self):
        config = OmegaConf.create({
            'model': {
                'vocab_size': 1000,
                'd_model': 256,
                'n_layers': 2,
                'n_heads': 4,
                'head_dim': 32,
                'd_ff': 1024,
                'n_experts': 4,
                'top_k': 2,
                'resident_slots': 2,
                'engram_layers': [1],
                'engram_slots': 1024,
                'engram_heads': 2,
                'engram_orders': [2, 3],
                'max_window': 128,
                'sink_tokens': 1,
                'weight_bits': 1.58,
                'tie_embeddings': True,
                'max_seq_len': 256,
                'dropout': 0.1,
            }
        })

        model = create_model(config)
        input_ids = torch.randint(0, 1000, (2, 32))

        outputs = model(input_ids)

        assert outputs['logits'].shape == (2, 32, 1000)
        if outputs['loss'] is not None:
            assert outputs['loss'].item() >= 0

    def test_generate(self):
        config = OmegaConf.create({
            'model': {
                'vocab_size': 1000,
                'd_model': 256,
                'n_layers': 2,
                'n_heads': 4,
                'head_dim': 32,
                'd_ff': 1024,
                'n_experts': 4,
                'top_k': 2,
                'resident_slots': 2,
                'engram_layers': [1],
                'engram_slots': 1024,
                'engram_heads': 2,
                'engram_orders': [2, 3],
                'max_window': 128,
                'sink_tokens': 1,
                'weight_bits': 1.58,
                'tie_embeddings': True,
                'max_seq_len': 256,
                'dropout': 0.1,
            }
        })

        model = create_model(config)
        model.prepare_for_inference()

        generated = model.generate(
            torch.tensor([[1]]),  # BOS
            max_new_tokens=10,
            temperature=1.0
        )

        assert generated.shape[1] == 11  # 1 initial + 10 generated

    def test_prepare_for_inference(self):
        config = OmegaConf.create({
            'model': {
                'vocab_size': 1000,
                'd_model': 256,
                'n_layers': 2,
                'n_heads': 4,
                'head_dim': 32,
                'd_ff': 1024,
                'n_experts': 4,
                'top_k': 2,
                'resident_slots': 2,
                'engram_layers': [1],
                'engram_slots': 1024,
                'engram_heads': 2,
                'engram_orders': [2, 3],
                'max_window': 128,
                'sink_tokens': 1,
                'weight_bits': 1.58,
                'tie_embeddings': True,
                'max_seq_len': 256,
            }
        })

        model = create_model(config)
        model.prepare_for_inference()

        assert model.inference_mode
        for block in model.blocks:
            assert block.inference_mode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
