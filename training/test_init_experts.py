"""
Test cases for init_experts.py to find runtime errors.
Run with: PYTHONPATH=. .venv/Scripts/python.exe -m pytest training/test_init_experts.py -v
"""

import os
import tempfile
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import pytest

# Import the functions to test
from training.init_experts import init_experts_from_dense
from utils.expert_io import MoEPager, ExpertIndex


class TestExpertInit:
    """Test expert initialization logic."""
    
    @pytest.fixture
    def mock_checkpoint(self):
        """Create a mock dense checkpoint with MLP weights."""
        d_model = 768
        hidden = 3072
        n_layers = 12
        
        state_dict = {}
        for layer_idx in range(n_layers):
            # MLP weights
            state_dict[f'transformer.blocks.{layer_idx}.mlp.c_fc.weight'] = torch.randn(hidden, d_model)
            state_dict[f'transformer.blocks.{layer_idx}.mlp.c_fc.bias'] = torch.randn(hidden)
            state_dict[f'transformer.blocks.{layer_idx}.mlp.c_proj.weight'] = torch.randn(d_model, hidden)
            state_dict[f'transformer.blocks.{layer_idx}.mlp.c_proj.bias'] = torch.randn(d_model)
            
            # Attention weights (needed for model loading)
            state_dict[f'transformer.blocks.{layer_idx}.attn.q_proj.weight'] = torch.randn(d_model, d_model)
            state_dict[f'transformer.blocks.{layer_idx}.attn.k_proj.weight'] = torch.randn(d_model, d_model)
            state_dict[f'transformer.blocks.{layer_idx}.attn.v_proj.weight'] = torch.randn(d_model, d_model)
            state_dict[f'transformer.blocks.{layer_idx}.attn.out_proj.weight'] = torch.randn(d_model, d_model)
        
        # Embeddings
        state_dict['transformer.wte.weight'] = torch.randn(4096, d_model)
        state_dict['transformer.wpe.weight'] = torch.randn(129, d_model)
        
        # Output heads
        state_dict['lm_head.weight'] = torch.randn(4096, d_model)
        
        return {'model': state_dict}
    
    @pytest.fixture
    def temp_expert_dir(self):
        """Create temporary directory for experts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_kmeans_clustering_shapes(self, mock_checkpoint):
        """Test that k-means produces correct centroid shapes."""
        from sklearn.cluster import KMeans
        
        d_model = 768
        hidden = 3072
        n_experts = 64
        
        # Use layer 0 weights
        c_fc = mock_checkpoint['model']['transformer.blocks.0.mlp.c_fc.weight']
        c_proj = mock_checkpoint['model']['transformer.blocks.0.mlp.c_proj.weight']
        
        fc_weight = c_fc.numpy().astype(np.float32)
        proj_weight = c_proj.numpy().astype(np.float32)
        proj_weight_t = proj_weight.T
        
        combined = np.concatenate([fc_weight, proj_weight_t], axis=1)
        assert combined.shape == (hidden, 2 * d_model)
        
        kmeans = KMeans(n_clusters=n_experts, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(combined)
        centroids = kmeans.cluster_centers_
        
        assert centroids.shape == (n_experts, 2 * d_model)
        assert cluster_labels.shape == (hidden,)
        assert cluster_labels.min() >= 0
        assert cluster_labels.max() < n_experts
    
    def test_centroid_split(self, mock_checkpoint):
        """Test splitting centroids back into fc and proj."""
        from sklearn.cluster import KMeans
        
        d_model = 768
        hidden = 3072
        n_experts = 64
        
        c_fc = mock_checkpoint['model']['transformer.blocks.0.mlp.c_fc.weight']
        c_proj = mock_checkpoint['model']['transformer.blocks.0.mlp.c_proj.weight']
        
        fc_weight = c_fc.numpy().astype(np.float32)
        proj_weight = c_proj.numpy().astype(np.float32)
        proj_weight_t = proj_weight.T
        combined = np.concatenate([fc_weight, proj_weight_t], axis=1)
        
        kmeans = KMeans(n_clusters=n_experts, random_state=42, n_init=10)
        kmeans.fit(combined)
        centroids = kmeans.cluster_centers_
        
        fc_centroids = centroids[:, :d_model]
        proj_centroids = centroids[:, d_model:].T
        
        assert fc_centroids.shape == (n_experts, d_model)
        assert proj_centroids.shape == (d_model, n_experts)
    
    def test_expert_weight_expansion(self):
        """Test expanding centroids to full expert weights."""
        d_model = 768
        expert_hidden = 3072
        n_experts = 64
        
        # Mock centroids
        fc_centroids = np.random.randn(n_experts, d_model).astype(np.float32)
        proj_centroids = np.random.randn(d_model, n_experts).astype(np.float32)
        
        for e in range(n_experts):
            gate_base = fc_centroids[e:e+1]
            up_base = fc_centroids[e:e+1]
            down_base = proj_centroids[:, e:e+1]
            
            gate_weight = gate_base.repeat(expert_hidden, axis=0) + np.random.randn(expert_hidden, d_model).astype(np.float32) * 0.01
            up_weight = up_base.repeat(expert_hidden, axis=0) + np.random.randn(expert_hidden, d_model).astype(np.float32) * 0.01
            down_weight = down_base.repeat(expert_hidden, axis=1) + np.random.randn(d_model, expert_hidden).astype(np.float32) * 0.01
            
            assert gate_weight.shape == (expert_hidden, d_model)
            assert up_weight.shape == (expert_hidden, d_model)
            assert down_weight.shape == (d_model, expert_hidden)
    
    def test_moe_pager_save_load(self, temp_expert_dir):
        """Test MoEPager save/load functionality."""
        pager = MoEPager(
            expert_dir=temp_expert_dir,
            resident_slots=0,
            n_layers=2,
            n_experts=4,
            d_model=768,
            expert_hidden=3072,
        )
        
        # Save an expert
        expert_tensors = {
            "gate": torch.randn(3072, 768, dtype=torch.bfloat16),
            "up": torch.randn(3072, 768, dtype=torch.bfloat16),
            "down": torch.randn(768, 3072, dtype=torch.bfloat16),
        }
        
        pager.save_expert(0, 0, expert_tensors)
        
        # Load it back
        loaded = pager.load_expert(0, 0)
        
        assert "gate" in loaded
        assert "up" in loaded
        assert "down" in loaded
        assert loaded["gate"].shape == (3072, 768)
        assert loaded["gate"].dtype == torch.bfloat16
        
        # Check file exists
        path = pager.index.get_path(0, 0)
        assert os.path.exists(path)
    
    def test_expert_index_json(self, temp_expert_dir):
        """Test expert index JSON creation and loading."""
        index = ExpertIndex(expert_dir=temp_expert_dir, n_layers=2, n_experts=4)
        
        path = index.get_path(0, 0)
        assert "layer0_expert0.safetensors" in path
        
        # Save index
        index.save()
        
        # Load index
        loaded_index = ExpertIndex.load(temp_expert_dir)
        assert loaded_index.n_layers == 2
        assert loaded_index.n_experts == 4
        
        loaded_path = loaded_index.get_path(1, 3)
        assert "layer1_expert3.safetensors" in loaded_path
    
    @pytest.mark.slow
    def test_full_init_single_layer(self, mock_checkpoint, temp_expert_dir):
        """Test full initialization for a single layer (slow)."""
        # Save mock checkpoint
        checkpoint_path = os.path.join(temp_expert_dir, "mock_generator.pt")
        torch.save(mock_checkpoint, checkpoint_path)
        
        # Run init for just 1 layer, 4 experts
        init_experts_from_dense(
            dense_checkpoint=checkpoint_path,
            expert_dir=temp_expert_dir,
            n_experts=4,
            n_layers=1,
            d_model=768,
            expert_hidden=3072,
            random_state=42,
        )
        
        # Verify all experts created
        for e in range(4):
            path = os.path.join(temp_expert_dir, f"layer0_expert{e}.safetensors")
            assert os.path.exists(path), f"Missing expert: {path}"
        
        # Verify index exists
        index_path = os.path.join(temp_expert_dir, "expert_index.json")
        assert os.path.exists(index_path)
    
    def test_skip_completed_layers(self, mock_checkpoint, temp_expert_dir):
        """Test that already-completed layers are skipped."""
        checkpoint_path = os.path.join(temp_expert_dir, "mock_generator.pt")
        torch.save(mock_checkpoint, checkpoint_path)
        
        # First run: 1 layer, 2 experts
        init_experts_from_dense(
            dense_checkpoint=checkpoint_path,
            expert_dir=temp_expert_dir,
            n_experts=2,
            n_layers=1,
            d_model=768,
            expert_hidden=3072,
            random_state=42,
        )
        
        # Second run: should skip layer 0
        init_experts_from_dense(
            dense_checkpoint=checkpoint_path,
            expert_dir=temp_expert_dir,
            n_experts=2,
            n_layers=1,
            d_model=768,
            expert_hidden=3072,
            random_state=42,
        )
        
        # Should still have only 2 experts
        for e in range(2):
            path = os.path.join(temp_expert_dir, f"layer0_expert{e}.safetensors")
            assert os.path.exists(path)
    
    def test_bias_handling(self, mock_checkpoint):
        """Test bias extraction and averaging."""
        c_fc_bias = mock_checkpoint['model']['transformer.blocks.0.mlp.c_fc.bias']
        c_proj_bias = mock_checkpoint['model']['transformer.blocks.0.mlp.c_proj.bias']
        
        # Simulate cluster labels
        cluster_labels = np.random.randint(0, 64, size=3072)
        expert_hidden = 3072
        
        for e in range(64):
            mask = cluster_labels == e
            if mask.sum() > 0:
                bias_base = c_fc_bias.numpy()[mask]
                gate_bias = np.mean(bias_base) * np.ones(expert_hidden, dtype=np.float32)
                assert gate_bias.shape == (expert_hidden,)
            else:
                gate_bias = np.zeros(expert_hidden, dtype=np.float32)
                assert gate_bias.shape == (expert_hidden,)
            
            down_bias = c_proj_bias.numpy().astype(np.float32)
            assert down_bias.shape == (768,)
    
    def test_dtype_bfloat16(self):
        """Test that weights are converted to bfloat16."""
        gate_weight = np.random.randn(3072, 768).astype(np.float32)
        gate_tensor = torch.from_numpy(gate_weight).to(torch.bfloat16)
        
        assert gate_tensor.dtype == torch.bfloat16
        assert gate_tensor.shape == (3072, 768)


class TestMoEBlock:
    """Test MoE block forward pass."""
    
    def test_moe_block_forward(self):
        """Test MoEBlock forward pass with NoPager variant."""
        from models.moe_block import MoEBlockNoPager
        from omegaconf import OmegaConf
        
        config = OmegaConf.create({
            'n_embd': 768,
            'n_layer': 12,
            'n_head': 12,
            'dropout': 0.1,
            'bias': True,
        })
        
        moe = MoEBlockNoPager(config, layer_idx=0, n_experts=4, expert_hidden=3072)
        x = torch.randn(2, 128, 768)
        
        out = moe(x)
        
        assert out.shape == (2, 128, 768)
        assert hasattr(moe, 'router_loss')
    
    def test_top2_router(self):
        """Test Top2Router forward pass."""
        from models.moe_block import Top2Router
        
        router = Top2Router(768, 8)
        x = torch.randn(2, 128, 768)
        
        weights, indices, logits = router(x)
        
        assert weights.shape == (2, 128, 2)
        assert indices.shape == (2, 128, 2)
        assert logits.shape == (2, 128, 8)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 128), atol=1e-5)


class TestEngramModule:
    """Test Engram module."""
    
    def test_engram_indices(self):
        """Test n-gram index generation."""
        from models.engram import engram_indices, engram_geometry
        
        tokens = torch.randint(0, 8192, (2, 128))
        orders = [2, 3]
        num_heads = 4
        slots = 8192
        
        idx = engram_indices(tokens, orders, num_heads, slots)
        
        num_tables = len(orders) * num_heads
        assert idx.shape == (2, 128, num_tables)
        assert idx.max() < slots
    
    def test_engram_forward(self):
        """Test EngramModule forward pass."""
        from models.engram import EngramModule
        
        engram = EngramModule(
            d_model=768,
            orders=[2, 3],
            num_heads=4,
            sub_dim=128,
            num_slots=8192,
            conv_taps=4,
            conv_dilation=3,
        )
        
        x = torch.randn(2, 128, 768)
        tokens = torch.randint(0, 8192, (2, 128))
        
        k, v = engram(tokens)
        
        assert k.shape == (2, 128, 768)
        assert v.shape == (2, 128, 768)


class TestKVCache:
    """Test bounded KV cache."""
    
    def test_kv_cache_basic(self):
        """Test basic KV cache operations."""
        from models.kv_cache import BoundedKVCache
        
        cache = BoundedKVCache(max_window=10, sink_tokens=1, n_kv_heads=3, head_dim=64)
        
        k = torch.randn(1, 3, 5, 64)
        v = torch.randn(1, 3, 5, 64)
        
        k_full, v_full = cache.update(0, k, v)
        
        # First call: sink=first token, window=remaining 4 → total=5
        assert k_full.shape[2] == 5  # 1 sink + 4 window (from 5 input)
        assert v_full.shape[2] == 5
        
        # Add more
        k2 = torch.randn(1, 3, 5, 64)
        v2 = torch.randn(1, 3, 5, 64)
        k_full2, v_full2 = cache.update(0, k2, v2)
        # Second call: sink(1) + window(4+5=9) = 10 (capped at max_window=10)
        assert k_full2.shape[2] == 10  # 1 sink + 10 window
        """Test that old tokens are evicted when window exceeded."""
        from models.kv_cache import BoundedKVCache
        
        cache = BoundedKVCache(max_window=10, sink_tokens=1, n_kv_heads=3, head_dim=64)
        
        # Fill beyond window
        for i in range(5):
            k = torch.randn(1, 3, 5, 64)
            v = torch.randn(1, 3, 5, 64)
            k_full, v_full = cache.update(0, k, v)
        
        # Window should be capped at max_window
        assert cache.k_cache[0].shape[2] <= 10
    
    def test_kv_cache_int8_quantization(self):
        """Test INT8 quantization/dequantization."""
        from models.kv_cache import quantize_int8, dequantize_int8
        
        x = torch.randn(2, 12, 128, 64, dtype=torch.bfloat16)
        x_q, scale = quantize_int8(x)
        x_dq = dequantize_int8(x_q, scale)
        
        assert x_q.dtype == torch.int8
        assert x_dq.dtype == torch.bfloat16
        assert x_dq.shape == x.shape
        
        # Check approximate reconstruction
        rel_error = (x - x_dq).abs().max() / (x.abs().max() + 1e-8)
        assert rel_error < 0.05  # ~5% relative error acceptable for INT8


class TestIntegratedModel:
    """Test integrated model components."""
    
    def test_model_creation(self):
        """Test creating the full model with mock VQ-VAE."""
        from models.moe_engram_generator import MoEEngramNextConceptTransformer, create_moe_engram_model
        from omegaconf import OmegaConf
        from unittest.mock import MagicMock
        import torch
        
        config = OmegaConf.create({
            'model': {
                'n_layer': 2,
                'n_head': 4,
                'n_embd': 256,
                'vocab_size': [4096] * 11,
                'context_length': 129,
                'prefix_len': 1,
                'tie_heads_to_codebook': True,
                'use_rope': True,
                'rope_base': 100000,
                'use_qk_norm': True,
                'use_level_encoding': True,
                'use_positional_encoding': True,
                'attn_pattern': 'block_diagonal',
                'dropout': 0.1,
                'bias': True,
            }
        })
        
        mock_vqvae = MagicMock()
        # Codebooks must be 384 dim (hardcoded in model)
        mock_vqvae.get_codebooks.return_value = [torch.randn(4096, 384, dtype=torch.bfloat16) for _ in range(11)]
        mock_vqvae.get_unigram.return_value = torch.ones(4096, dtype=torch.bfloat16) / 4096
        
        # Use NoPager MoE for testing
        model = MoEEngramNextConceptTransformer(vqvae_local=mock_vqvae, config=config.model, expert_dir='experts')
        
        for i, block in enumerate(model.blocks):
            from models.moe_block import MoEBlockNoPager
            block.mlp = MoEBlockNoPager(config.model, layer_idx=i, n_experts=4, expert_hidden=1024)
        # Test forward
        x = torch.randint(0, 4096, (2, 128))
        with torch.no_grad():
            logits, loss = model(x, level=0, targets=x)
        
        assert logits.shape == (2, 128, 4096)
        assert loss is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])