"""
Test cases for training/train_moe_engram.py to find runtime errors.
Run with: PYTHONPATH=. .venv/Scripts/python.exe -m pytest training/test_train.py -v
"""

import os
import tempfile
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import pytest

from omegaconf import OmegaConf


class TestTinyStoriesDataset:
    """Test dataset loading and chunking."""
    
    @pytest.fixture
    def dummy_pkl_data(self):
        """Create dummy tokenized data."""
        # Create sequences of tokens
        seqs = [
            list(np.random.randint(0, 4096, 500)),
            list(np.random.randint(0, 4096, 600)),
            list(np.random.randint(0, 4096, 450)),
        ]
        return seqs
    
    @pytest.fixture
    def temp_pkl(self, dummy_pkl_data):
        """Create temporary .pkl file."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            import pickle
            pickle.dump(dummy_pkl_data, f)
            yield f.name
        os.unlink(f.name)
    
    def test_dataset_creation(self, temp_pkl):
        """Test dataset loads and chunks correctly."""
        from training.train_moe_engram import TinyStoriesDataset
        
        dataset = TinyStoriesDataset(temp_pkl, context_length=128)
        
        assert len(dataset) > 0
        assert dataset.context_length == 128
        
        # Check first item
        item = dataset[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (128,)
        assert item.dtype == torch.long
    
    def test_dataset_chunking(self, temp_pkl):
        """Test that sequences are chunked with overlap."""
        from training.train_moe_engram import TinyStoriesDataset
        
        dataset = TinyStoriesDataset(temp_pkl, context_length=128)
        
        # With 500 tokens and stride 64, expect ~6 chunks per sequence
        # Total chunks should be > 3 (one per sequence) and < 20
        assert 3 < len(dataset) < 20


class TestPrepareDataset:
    """Test dataset preparation."""
    
    @patch('training.train_moe_engram.snapshot_download')
    def test_prepare_dataset(self, mock_snapshot):
        """Test prepare_dataset downloads to correct path."""
        from training.train_moe_engram import prepare_dataset
        
        path = prepare_dataset()
        
        assert path == "data/tinystories/tinystories/train.pkl"
        mock_snapshot.assert_called_once()


class TestOptimizer:
    """Test two-timescale optimizer creation."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model with named parameters matching the three groups."""
        model = MagicMock()
        
        # Create mock parameters for three groups
        trunk_params = [torch.nn.Parameter(torch.randn(100, 100)) for _ in range(3)]
        engram_params = [torch.nn.Parameter(torch.randn(50, 50)) for _ in range(2)]
        expert_params = [torch.nn.Parameter(torch.randn(80, 80)) for _ in range(4)]
        
        # Named parameters
        named_params = []
        for i, p in enumerate(trunk_params):
            named_params.append((f'transformer.blocks.{i}.attn.q_proj.weight', p))
        for i, p in enumerate(engram_params):
            named_params.append((f'engram.embeddings.{i}', p))
        for i, p in enumerate(expert_params):
            named_params.append((f'transformer.blocks.{i}.mlp.gate.weight', p))
            named_params.append((f'transformer.blocks.{i}.mlp.up.weight', p))
            named_params.append((f'transformer.blocks.{i}.mlp.down.weight', p))
        
        model.named_parameters.return_value = named_params
        model.parameters.return_value = trunk_params + engram_params + expert_params
        
        return model
    
    def test_optimizer_groups(self, mock_model):
        """Test optimizer creates three parameter groups with correct LRs."""
        from training.train_moe_engram import get_optimizer
        
        optimizer = get_optimizer(mock_model, lr_trunk=3e-4, lr_engram=1e-4, lr_expert=3e-3)
        
        assert len(optimizer.param_groups) == 3
        
        # Check LRs
        lrs = [g['lr'] for g in optimizer.param_groups]
        assert 3e-4 in lrs
        assert 1e-4 in lrs
        assert 3e-3 in lrs
        
        # Check parameter counts
        total_params = sum(len(g['params']) for g in optimizer.param_groups)
        assert total_params > 0


class TestScheduler:
    """Test learning rate scheduler."""
    
    def test_scheduler_warmup(self):
        """Test scheduler warmup phase."""
        from training.train_moe_engram import get_scheduler
        
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10, 10))], lr=3e-4)
        scheduler = get_scheduler(optimizer, warmup_steps=1000, max_steps=50000)
        
        # Step 0: LR should be 0
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        assert lr < 3e-4 * 0.01  # Very small at step 1
        
        # Step 500: LR should be halfway
        for _ in range(499):
            scheduler.step()
        lr = scheduler.get_last_lr()[0]
        assert abs(lr - 3e-4 * 0.5) < 3e-4 * 0.1
        
        # Step 1000: LR should be at base
        for _ in range(500):
            scheduler.step()
        lr = scheduler.get_last_lr()[0]
        assert abs(lr - 3e-4) < 3e-4 * 0.01
    
    def test_scheduler_cosine_decay(self):
        """Test scheduler cosine decay after warmup."""
        from training.train_moe_engram import get_scheduler
        
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10, 10))], lr=3e-4)
        scheduler = get_scheduler(optimizer, warmup_steps=10, max_steps=100)
        
        # Warmup
        for _ in range(10):
            scheduler.step()
        lr_warmup_end = scheduler.get_last_lr()[0]
        
        # Decay to near zero
        for _ in range(90):
            scheduler.step()
        lr_end = scheduler.get_last_lr()[0]
        
        assert lr_end < lr_warmup_end * 0.1


class TestTrainStep:
    """Test single training step."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock model, tokenizer, optimizer, scaler."""
        from models.moe_engram_generator import MoEEngramNextConceptTransformer
        from models.moe_block import MoEBlockNoPager
        
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
                'attn_pattern': 'block_diagonal',
                'dropout': 0.1,
                'bias': True,
            }
        })
        
        mock_vqvae = MagicMock()
        mock_vqvae.get_codebooks.return_value = [torch.randn(4096, 256, dtype=torch.bfloat16) for _ in range(11)]
        mock_vqvae.get_unigram.return_value = torch.ones(4096, dtype=torch.bfloat16) / 4096
        mock_vqvae.get_ncm_inputs.return_value = (
            torch.randint(0, 4096, (2, 128, 11)),
            torch.randint(0, 4096, (2, 128))
        )
        
        model = MoEEngramNextConceptTransformer(vqvae_local=mock_vqvae, config=config.model, expert_dir='experts')
        for i, block in enumerate(model.blocks):
            block.mlp = MoEBlockNoPager(config.model, layer_idx=i, n_experts=4, expert_hidden=1024)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scaler = torch.amp.GradScaler('cpu')
        
        return model, mock_vqvae, optimizer, scaler
    
    def test_train_step_cpu(self, mock_components):
        """Test training step runs without error on CPU."""
        from training.train_moe_engram import train_step
        
        model, tokenizer, optimizer, scaler = mock_components
        batch = torch.randint(0, 4096, (2, 128))
        
        loss, router_loss = train_step(model, batch, tokenizer, optimizer, scaler, accum_steps=2, step=0)
        
        assert isinstance(loss, float)
        assert isinstance(router_loss, float)
        assert loss >= 0
    
    def test_train_step_accumulation(self, mock_components):
        """Test gradient accumulation works."""
        from training.train_moe_engram import train_step
        
        model, tokenizer, optimizer, scaler = mock_components
        batch = torch.randint(0, 4096, (2, 128))
        
        # Step 0 (not accumulation boundary)
        loss1, _ = train_step(model, batch, tokenizer, optimizer, scaler, accum_steps=4, step=0)
        
        # Step 1
        loss2, _ = train_step(model, batch, tokenizer, optimizer, scaler, accum_steps=4, step=1)
        
        # Step 2
        loss3, _ = train_step(model, batch, tokenizer, optimizer, scaler, accum_steps=4, step=2)
        
        # Step 3 (accumulation boundary - optimizer should step)
        loss4, _ = train_step(model, batch, tokenizer, optimizer, scaler, accum_steps=4, step=3)
        
        # All should return valid losses
        for loss in [loss1, loss2, loss3, loss4]:
            assert isinstance(loss, float)
            assert loss >= 0


class TestSaveCheckpoint:
    """Test checkpoint saving."""
    
    @pytest.fixture
    def mock_model_with_pager(self):
        """Create mock model with MoE pagers."""
        from models.moe_engram_generator import MoEEngramNextConceptTransformer
        from models.moe_block import MoEBlockNoPager
        from utils.expert_io import MoEPager
        
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
                'attn_pattern': 'block_diagonal',
                'dropout': 0.1,
                'bias': True,
            }
        })
        
        mock_vqvae = MagicMock()
        mock_vqvae.get_codebooks.return_value = [torch.randn(4096, 256, dtype=torch.bfloat16) for _ in range(11)]
        mock_vqvae.get_unigram.return_value = torch.ones(4096, dtype=torch.bfloat16) / 4096
        mock_vqvae.get_ncm_inputs.return_value = (
            torch.randint(0, 4096, (2, 128, 11)),
            torch.randint(0, 4096, (2, 128))
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MoEEngramNextConceptTransformer(vqvae_local=mock_vqvae, config=config.model, expert_dir=tmpdir)
            for i, block in enumerate(model.blocks):
                block.mlp = MoEBlockNoPager(config.model, layer_idx=i, n_experts=4, expert_hidden=1024)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
            
            yield model, optimizer, scheduler, tmpdir
    
    def test_save_checkpoint(self, mock_model_with_pager):
        """Test checkpoint saves model, optimizer, scheduler, step."""
        from training.train_moe_engram import save_checkpoint
        
        model, optimizer, scheduler, tmpdir = mock_model_with_pager
        checkpoint_path = os.path.join(tmpdir, 'checkpoints', 'step_1000.pt')
        
        save_checkpoint(model, optimizer, scheduler, 1000, checkpoint_path)
        
        assert os.path.exists(checkpoint_path)
        
        # Verify loadable
        loaded = torch.load(checkpoint_path, map_location='cpu')
        assert 'model' in loaded
        assert 'optimizer' in loaded
        assert 'scheduler' in loaded
        assert loaded['step'] == 1000
    
    def test_save_checkpoint_creates_dir(self, mock_model_with_pager):
        """Test checkpoint creates parent directory."""
        from training.train_moe_engram import save_checkpoint
        
        model, optimizer, scheduler, tmpdir = mock_model_with_pager
        checkpoint_path = os.path.join(tmpdir, 'new_subdir', 'step_1000.pt')
        
        save_checkpoint(model, optimizer, scheduler, 1000, checkpoint_path)
        
        assert os.path.exists(checkpoint_path)


class TestMainFunction:
    """Test main training function components."""
    
    def test_model_creation_in_main(self):
        """Test create_moe_engram_model works."""
        from models.moe_engram_generator import create_moe_engram_model
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
                'attn_pattern': 'block_diagonal',
                'dropout': 0.1,
                'bias': True,
            },
            'tokenizer': {}
        })
        
        mock_vqvae = MagicMock()
        mock_vqvae.get_codebooks.return_value = [torch.randn(4096, 256, dtype=torch.bfloat16) for _ in range(11)]
        mock_vqvae.get_unigram.return_value = torch.ones(4096, dtype=torch.bfloat16) / 4096
        
        # This will fail without expert dir, but we can test the mock path
        with patch('models.moe_engram_generator.VQVAETokenizer.from_config', return_value=mock_vqvae):
            with patch('models.moe_engram_generator.MoEEngramNextConceptTransformer') as mock_cls:
                mock_instance = MagicMock()
                mock_cls.return_value = mock_instance
                
                # Can't fully test without experts, but we verify no import errors
                pass
    
    def test_create_dummy_dataset(self):
        """Test create_dummy_dataset.py creates valid pkl."""
        from training.create_dummy_dataset import create_dummy_dataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'train.pkl')
            
            # Create dummy dataset
            create_dummy_dataset(output_path, n_sequences=10, seq_length=200)
            
            assert os.path.exists(output_path)
            
            # Verify loadable
            import pickle
            with open(output_path, 'rb') as f:
                data = pickle.load(f)
            
            assert isinstance(data, list)
            assert len(data) == 10
            for seq in data:
                assert isinstance(seq, list)
                assert len(seq) == 200
                assert all(isinstance(t, int) for t in seq)


class TestIntegration:
    """Integration tests for training pipeline."""
    
    @patch('training.train_moe_engram.prepare_dataset')
    def test_training_loop_structure(self, mock_prepare):
        """Test that training loop components wire together."""
        from training.train_moe_engram import (
            TinyStoriesDataset, get_optimizer, get_scheduler, 
            train_step, save_checkpoint
        )
        from models.moe_engram_generator import MoEEngramNextConceptTransformer
        from models.moe_block import MoEBlockNoPager
        from unittest.mock import MagicMock
        import torch
        
        mock_prepare.return_value = "dummy.pkl"
        
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
                'attn_pattern': 'block_diagonal',
                'dropout': 0.1,
                'bias': True,
            }
        })
        
        mock_vqvae = MagicMock()
        mock_vqvae.get_codebooks.return_value = [torch.randn(4096, 256, dtype=torch.bfloat16) for _ in range(11)]
        mock_vqvae.get_unigram.return_value = torch.ones(4096, dtype=torch.bfloat16) / 4096
        mock_vqvae.get_ncm_inputs.return_value = (
            torch.randint(0, 4096, (2, 128, 11)),
            torch.randint(0, 4096, (2, 128))
        )
        
        model = MoEEngramNextConceptTransformer(vqvae_local=mock_vqvae, config=config.model, expert_dir='experts')
        for i, block in enumerate(model.blocks):
            block.mlp = MoEBlockNoPager(config.model, layer_idx=i, n_experts=4, expert_hidden=1024)
        
        optimizer = get_optimizer(model)
        scheduler = get_scheduler(optimizer, warmup_steps=10, max_steps=100)
        scaler = torch.amp.GradScaler('cpu')
        
        # Verify all components created
        assert model is not None
        assert optimizer is not None
        assert scheduler is not None
        assert scaler is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])