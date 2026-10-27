"""
Main Training Entry Point for psudo-Engine.

Orchestrates the complete training pipeline with configuration management,
schema version checking, checkpointing, and feedback integration.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from .psudo_dataset import generate_synthetic_data
from .ternary_trainer import create_trainer

# Schema version for psudo-spec compatibility
PSUDO_SPEC_VERSION = "1.0.0"


def validate_schema_version(config: DictConfig) -> bool:
    """
    Validate schema version in config matches expected version.

    Args:
        config: Training configuration

    Returns:
        True if version matches
    """
    schema_version = config.get('schema_version', '0.0.0')

    return schema_version == PSUDO_SPEC_VERSION


def check_dependencies() -> bool:
    """Check required dependencies are available."""
    try:
        required = ['numpy', 'omegaconf', 'sentence_transformers']
        for pkg in required:
            if importlib.util.find_spec(pkg) is None:
                return False

        if torch.cuda.is_available():
            pass
        return True
    except ImportError:
        return False


def setup_environment(config: DictConfig):
    """Setup training environment."""
    # Create output directories
    output_dir = Path(config.get('output_dir', 'checkpoints'))
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_dir = Path(config.get('weight_store_dir', 'weights'))
    weight_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(config.get('data_dir', 'data'))
    data_dir.mkdir(parents=True, exist_ok=True)

    feedback_dir = Path(config.get('feedback_path', 'data/feedback_loop/'))
    feedback_dir.mkdir(parents=True, exist_ok=True)

    # Set random seeds for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Enable TF32 for Ampere+ GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True



def prepare_data(config: DictConfig, force_regenerate: bool = False):
    """Prepare training data if not present."""
    train_path = Path(config.get('train_data_path', 'data/base_grammar.jsonl'))
    val_path = Path(config.get('val_data_path', 'data/val_grammar.jsonl'))

    if not train_path.exists() or force_regenerate:
        num_train = config.get('num_train_samples', 10000)
        generate_synthetic_data(str(train_path), num_samples=num_train)

    if not val_path.exists() or force_regenerate:
        num_val = config.get('num_val_samples', 1000)
        generate_synthetic_data(str(val_path), num_samples=num_val)


def save_config(config: DictConfig, path: str):
    """Save configuration to file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='psudo-Engine Training')

    # Config
    parser.add_argument('--config', type=str, default='psudo_engine/config/training.yaml',
                        help='Path to training config')
    parser.add_argument('--model-config', type=str, default='psudo_engine/config/model.yaml',
                        help='Path to model config')

    # Training control
    parser.add_argument('--max-steps', type=int, help='Override max training steps')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--grad-accum', type=int, help='Override gradient accumulation')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--eval-only', action='store_true', help='Run evaluation only')

    # Data
    parser.add_argument('--prepare-data', action='store_true', help='Prepare/regenerate data')
    parser.add_argument('--force-regenerate', action='store_true', help='Force data regeneration')

    # Output
    parser.add_argument('--output-dir', type=str, help='Override output directory')

    # Schema
    parser.add_argument('--skip-schema-check', action='store_true', help='Skip schema version check')

    args = parser.parse_args()

    # Load configs
    model_config = OmegaConf.load(args.model_config)
    train_config = OmegaConf.load(args.config)

    # Merge configs
    config = OmegaConf.merge(train_config, model_config)

    # Apply CLI overrides
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.grad_accum:
        config.grad_accum = args.grad_accum
    if args.resume:
        config.resume_from = args.resume
    if args.output_dir:
        config.output_dir = args.output_dir

    # Add schema version
    config.schema_version = PSUDO_SPEC_VERSION


    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Validate schema version
    if not args.skip_schema_check and not validate_schema_version(config):
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Setup environment
    setup_environment(config)

    # Prepare data if requested
    if args.prepare_data:
        prepare_data(config, force_regenerate=args.force_regenerate)

    # Save merged config
    save_config(config, Path(config.output_dir) / 'training_config.yaml')

    # Run training or evaluation
    if args.eval_only:
        # TODO: Implement evaluation mode
        pass
    else:
        trainer = create_trainer(config)
        trainer.train()


def run_training(config_path: str, model_config_path: str = None, **overrides):
    """
    Programmatic training entry point.

    Args:
        config_path: Path to training config
        model_config_path: Path to model config (optional)
        **overrides: Config overrides
    """
    model_config_path = model_config_path or 'psudo_engine/config/model.yaml'

    model_config = OmegaConf.load(model_config_path)
    train_config = OmegaConf.load(config_path)

    config = OmegaConf.merge(train_config, model_config)

    # Apply overrides
    for key, value in overrides.items():
        OmegaConf.update(config, key, value)

    config.schema_version = PSUDO_SPEC_VERSION

    setup_environment(config)
    save_config(config, Path(config.output_dir) / 'training_config.yaml')

    trainer = create_trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
