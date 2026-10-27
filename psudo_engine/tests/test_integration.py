"""
Integration Tests for psudo-Engine.

End-to-end tests covering the full pipeline from generation to training.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from psudo_engine.context.context_builder import ContextBuilder, ContextConfig
from psudo_engine.context.retriever import HybridRetriever
from psudo_engine.context.syntax_hub import SyntaxHub
from psudo_engine.inference.generator import GenerationConfig, PsudoGenerator
from psudo_engine.inference.tokenizer import PsudoTokenizer
from psudo_engine.model.psudo_transformer import create_model
from psudo_engine.training.curriculum import CurriculumScheduler
from psudo_engine.training.psudo_dataset import create_datasets, generate_synthetic_data
from psudo_engine.training.ternary_trainer import TernaryTrainer, TrainingConfig


class TestEndToEndGeneration:
    """End-to-end generation tests."""

    def test_full_generation_pipeline(self):
        """Test complete generation from prompt to output."""
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
        tokenizer = PsudoTokenizer(vocab_size=1000)
        generator = PsudoGenerator(model=model, tokenizer=tokenizer)

        prompt = "fn hello() -> string {"
        gen_config = GenerationConfig(
            max_new_tokens=20,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
        )

        result = generator.generate(prompt, gen_config)

        assert isinstance(result, str)
        assert len(result) > 0
        # Just check that generation produces valid output of reasonable length
        # (small model may not produce coherent text)
        assert len(result) >= len(prompt)

    def test_chat_mode(self):
        """Test chat-style conversation."""
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
        tokenizer = PsudoTokenizer(vocab_size=1000)
        generator = PsudoGenerator(model=model, tokenizer=tokenizer)

        messages = [
            {"role": "user", "content": "Write a function to add two numbers"},
        ]

        gen_config = GenerationConfig(max_new_tokens=30, temperature=0.8)
        response = generator.chat(messages, gen_config)

        assert isinstance(response, str)
        assert len(response) > 0


class TestTrainingPipeline:
    """Tests for training pipeline."""

    def test_train_step(self):
        """Test single training step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate synthetic data
            data_path = Path(tmpdir) / "train.jsonl"
            generate_synthetic_data(str(data_path), num_samples=20)
            
            config = TrainingConfig(
                vocab_size=1000,
                d_model=256,
                n_layers=2,
                n_heads=4,
                head_dim=32,
                d_ff=1024,
                n_experts=4,
                max_steps=10,
                batch_size=2,
                grad_accum=2,
                output_dir="test_checkpoints",
                train_data_path=str(data_path),
                val_data_path=str(data_path),
                max_length=64,
                engram_layers=[1],
                engram_slots=1024,
                engram_heads=2,
                engram_orders=[2, 3],
                max_window=128,
                sink_tokens=1,
                use_ternary=True,
                tie_embeddings=True,
                max_seq_len=256,
            )
            
            trainer = TernaryTrainer(config)
            
            # Create dummy batch
            batch = {
                'input_ids': torch.randint(0, 1000, (2, 64)),
                'labels': torch.randint(0, 1000, (2, 64)),
            }
            
            metrics = trainer.train_step(batch)
            
            assert 'loss' in metrics
            assert metrics['loss'] >= 0
            assert 'grad_norm' in metrics
            assert 'lr_trunk' in metrics

    def test_evaluation(self):
        """Test evaluation mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate synthetic data
            data_path = Path(tmpdir) / "train.jsonl"
            generate_synthetic_data(str(data_path), num_samples=20)
            
            config = TrainingConfig(
                vocab_size=1000,
                d_model=256,
                n_layers=2,
                n_heads=4,
                head_dim=32,
                d_ff=1024,
                n_experts=4,
                max_steps=10,
                batch_size=2,
                output_dir="test_checkpoints",
                train_data_path=str(data_path),
                val_data_path=str(data_path),
                max_length=64,
                engram_layers=[1],
                engram_slots=1024,
                engram_heads=2,
                engram_orders=[2, 3],
                max_window=128,
                sink_tokens=1,
                use_ternary=True,
                tie_embeddings=True,
                max_seq_len=256,
            )
            
            trainer = TernaryTrainer(config)
            val_metrics = trainer.evaluate()
            
            assert 'val_loss' in val_metrics
            assert val_metrics['val_loss'] >= 0

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                vocab_size=1000,
                d_model=256,
                n_layers=2,
                n_heads=4,
                head_dim=32,
                d_ff=1024,
                n_experts=4,
                max_steps=10,
                batch_size=2,
                output_dir=tmpdir,
            )

            trainer = TernaryTrainer(config)

            # Save checkpoint
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            trainer.save_checkpoint(str(checkpoint_path))

            assert checkpoint_path.exists()

            # Load checkpoint
            trainer2 = TernaryTrainer(config)
            trainer2._load_checkpoint(str(checkpoint_path))

            assert trainer2.step == trainer.step


class TestCurriculumLearning:
    """Tests for curriculum learning."""

    def test_curriculum_progression(self):
        """Test curriculum advances through stages."""
        scheduler = CurriculumScheduler()

        # Simulate training with decreasing validation loss
        for i in range(15):
            val_loss = 5.0 - i * 0.25
            changed = scheduler.step(val_loss)

            if changed:
                progress = scheduler.get_progress()
                assert progress['current_stage'] in [
                    "Basic Syntax", "Control Flow", "Data Structures & Functions", "Full Programs"
                ]

    def test_synthetic_data_generation(self):
        """Test synthetic data generation for curriculum stages."""
        from psudo_engine.training.curriculum import SyntheticDataGenerator

        generator = SyntheticDataGenerator()
        scheduler = CurriculumScheduler()

        samples = generator.generate_progressive(scheduler.stages, 2)

        assert len(samples) == 8  # 4 stages * 2 samples
        assert all('code' in s for s in samples)
        assert all('stage' in s for s in samples)


class TestContextRetrieval:
    """Tests for context retrieval pipeline."""

    def test_syntax_hub_retrieval(self):
        """Test syntax hub provides relevant rules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hub = SyntaxHub(tmpdir)
            hub.initialize(create_defaults=True)

            # Search for file I/O related rules
            results = hub.search_rules("read file")
            assert len(results) > 0

            # Get context for task
            context = hub.get_context_for_task(
                "read a file and print its contents",
                language='psudo'
            )

            assert "read_file" in context or "write_file" in context

    def test_hybrid_retriever(self):
        """Test hybrid BM25 + embedding retrieval."""
        retriever = HybridRetriever()

        docs = [
            ("func1", "fn read_file(path: string) -> string { read_file(path) }",
             {"language": "psudo", "category": "builtin"}),
            ("func2", "fn write_file(path: string, content: string) -> () { write_file(path, content) }",
             {"language": "psudo", "category": "builtin"}),
            ("func3", "for i in range(10) { print(i) }",
             {"language": "psudo", "category": "control_flow"}),
        ]

        retriever.add_documents(docs)
        retriever.build()

        results = retriever.search("read file", top_k=2)
        assert len(results) == 2

        # Should find read_file related docs
        found_read = any("read_file" in doc.content for _, doc in results)
        assert found_read

    def test_context_builder(self):
        """Test full context building pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            syntax_hub = SyntaxHub(tmpdir)
            syntax_hub.initialize(create_defaults=True)

            retriever = HybridRetriever()
            retriever.add_document(
                "read_file", "fn read_file(path: string) -> string { read_file(path) }",
                {"language": "psudo", "category": "builtin"}
            )
            retriever.build()

            builder = ContextBuilder(
                config=ContextConfig(max_tokens=500),
                syntax_hub=syntax_hub,
                retriever=retriever
            )

            context = builder.build_context(
                task="Read a configuration file",
                language='psudo'
            )

            assert "Task" in context
            assert len(context) > 0


class TestDataPipeline:
    """Tests for data loading and processing."""

    def test_synthetic_data_generation(self):
        """Test synthetic psudo data generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "synthetic.jsonl"
            generate_synthetic_data(str(data_path), num_samples=100)

            assert data_path.exists()

            # Load and verify
            with open(data_path) as f:
                lines = f.readlines()

            assert len(lines) == 100
            for line in lines:
                import json
                sample = json.loads(line)
                assert 'code' in sample

    def test_dataset_creation(self):
        """Test dataset creation from JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "train.jsonl"
            generate_synthetic_data(str(data_path), num_samples=50)

            train_dataset, val_dataset, feedback_dataset, tokenizer = create_datasets({
                'vocab_size': 1000,
                'train_data_path': str(data_path),
                'val_data_path': str(data_path),
                'max_length': 128,
                'feedback_enabled': False,
            })

            assert len(train_dataset) > 0
            assert len(val_dataset) > 0
            assert feedback_dataset is None

            # Test sample
            sample = train_dataset[0]
            assert 'input_ids' in sample
            assert 'labels' in sample
            assert sample['input_ids'].shape == sample['labels'].shape

    def test_feedback_dataset(self):
        """Test feedback dataset merging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create base data
            base_path = Path(tmpdir) / "base.jsonl"
            generate_synthetic_data(str(base_path), num_samples=20)

            # Create feedback data
            feedback_dir = Path(tmpdir) / "feedback"
            feedback_dir.mkdir()
            feedback_file = feedback_dir / "corrections_001.jsonl"

            feedback_samples = [
                {
                    "prompt": "Read a file",
                    "faulty_code": "fn read(path) { open(path) }",
                    "error_message": "unknown function open",
                    "corrected_code": "fn read_file(path: string) -> string { read_file(path) }"
                },
                {
                    "prompt": "Write to file",
                    "faulty_code": "fn write(path, content) { save(path, content) }",
                    "error_message": "unknown function save",
                    "corrected_code": "fn write_file(path: string, content: string) -> () { write_file(path, content) }"
                }
            ]

            import json
            with open(feedback_file, 'w') as f:
                for sample in feedback_samples:
                    f.write(json.dumps(sample) + '\n')

            # Create datasets
            train_dataset, val_dataset, feedback_dataset, tokenizer = create_datasets({
                'vocab_size': 1000,
                'train_data_path': str(base_path),
                'val_data_path': str(base_path),
                'max_length': 128,
                'feedback_enabled': True,
                'feedback_path': str(feedback_dir),
            })

            assert feedback_dataset is not None
            assert len(feedback_dataset) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
