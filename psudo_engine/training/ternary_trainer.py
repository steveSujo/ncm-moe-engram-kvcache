"""
Ternary Trainer for psudo-Engine.

Implements Straight-Through Estimator (STE) training loop for ternary quantized models
with gradient accumulation, mixed precision, and two-timescale optimizer.
"""

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig, OmegaConf

from ..model.psudo_transformer import PsudoTransformer, create_model
from .psudo_dataset import PsudoTokenizer, create_datasets


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    vocab_size: int = 8192
    d_model: int = 512
    n_layers: int = 24
    n_heads: int = 8
    head_dim: int = 64
    d_ff: int = 2048
    n_experts: int = 64
    top_k: int = 2
    engram_layers: list[int] = None
    engram_slots: int = 8192
    engram_heads: int = 4
    engram_orders: list[int] = None
    max_window: int = 256
    sink_tokens: int = 1
    use_ternary: bool = True
    tie_embeddings: bool = True
    max_seq_len: int = 2048
    
    # Training
    batch_size: int = 4
    grad_accum: int = 8
    max_steps: int = 50000
    checkpoint_every: int = 1000
    eval_every: int = 500
    log_every: int = 10
    
    # Optimizer (two-timescale)
    trunk_lr: float = 3e-4
    engram_lr: float = 1e-4
    expert_lr: float = 3e-3
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    
    # Gradient clipping
    grad_clip: float = 1.0
    
    # Mixed precision
    amp: bool = True
    dtype: str = "bfloat16"
    
    # Data
    train_data_path: str = "data/base_grammar.jsonl"
    val_data_path: str = "data/val_grammar.jsonl"
    max_length: int = 512
    
    # Feedback
    feedback_enabled: bool = True
    feedback_path: str = "data/feedback_loop/"
    feedback_ratio: float = 0.3
    
    # Curriculum
    curriculum_enabled: bool = True
    curriculum_warmup: int = 1000
    
    # Paths
    output_dir: str = "checkpoints"
    weight_store_dir: str = "weights"
    
    # Resume
    resume_from: Optional[str] = None
    
    def __post_init__(self):
        if self.engram_layers is None:
            self.engram_layers = [4, 12, 20]
        if self.engram_orders is None:
            self.engram_orders = [2, 3]


class TernaryTrainer:
    """
    Trainer for ternary quantized PsudoTransformer.
    
    Features:
    - Two-timescale optimizer (trunk, engram, experts)
    - STE gradient flow for ternary weights
    - Gradient accumulation
    - Mixed precision (AMP)
    - Gradient clipping
    - Checkpointing
    - Evaluation
    - Curriculum learning
    - Feedback dataset integration
    """
    
    def __init__(self, config: TrainingConfig | DictConfig):
        if isinstance(config, DictConfig):
            config = OmegaConf.to_object(config)
        if not isinstance(config, TrainingConfig):
            config = TrainingConfig(**config)
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = getattr(torch, config.dtype)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Create tokenizer
        self.tokenizer = PsudoTokenizer(vocab_size=config.vocab_size)
        
        # Create model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # Create datasets
        self.train_loader, self.val_loader, self.feedback_loader = self._create_dataloaders()
        
        # AMP scaler
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.amp and self.device.type == 'cuda')
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup training logger."""
        logger = logging.getLogger('ternary_trainer')
        logger.setLevel(logging.INFO)
        
        # File handler
        log_dir = Path(self.config.output_dir) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f'training_{time.strftime("%Y%m%d_%H%M%S")}.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _create_model(self) -> PsudoTransformer:
        """Create model from config."""
        model_config = OmegaConf.create({
            'model': {
                'vocab_size': self.config.vocab_size,
                'd_model': self.config.d_model,
                'n_layers': self.config.n_layers,
                'n_heads': self.config.n_heads,
                'head_dim': self.config.head_dim,
                'd_ff': self.config.d_ff,
                'n_experts': self.config.n_experts,
                'top_k': self.config.top_k,
                'resident_slots': self.config.n_experts // 2,
                'engram_layers': self.config.engram_layers,
                'engram_slots': self.config.engram_slots,
                'engram_heads': self.config.engram_heads,
                'engram_orders': self.config.engram_orders,
                'max_window': self.config.max_window,
                'sink_tokens': self.config.sink_tokens,
                'weight_bits': 1.58 if self.config.use_ternary else 16,
                'tie_embeddings': self.config.tie_embeddings,
                'dropout': 0.1,
                'max_seq_len': self.config.max_seq_len,
            }
        })
        
        model = create_model(model_config)
        
        # Load checkpoint if resuming
        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)
        
        return model
    
    def _create_optimizer(self) -> Optimizer:
        """Create two-timescale optimizer."""
        # Group parameters by type
        trunk_params = []
        engram_params = []
        expert_params = []
        
        for name, param in self.model.named_parameters():
            if 'engram' in name.lower():
                engram_params.append(param)
            elif 'expert' in name.lower() or 'moe' in name.lower() or 'router' in name.lower():
                expert_params.append(param)
            else:
                trunk_params.append(param)
        
        param_groups = [
            {'params': trunk_params, 'lr': self.config.trunk_lr, 'name': 'trunk'},
            {'params': engram_params, 'lr': self.config.engram_lr, 'name': 'engram'},
            {'params': expert_params, 'lr': self.config.expert_lr, 'name': 'experts'},
        ]
        
        # Filter empty groups
        param_groups = [g for g in param_groups if g['params']]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
            eps=self.config.eps,
        )
        
        self.logger.info(f"Optimizer groups: {[g['name'] for g in param_groups]}")
        for g in param_groups:
            self.logger.info(f"  {g['name']}: {len(g['params'])} params, lr={g['lr']}")
        
        return optimizer
    
    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            warmup = self.config.curriculum_warmup
            if step < warmup:
                return step / warmup
            # Cosine decay after warmup
            progress = (step - warmup) / max(1, self.config.max_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _create_dataloaders(self) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """Create train, validation, and feedback dataloaders."""
        train_dataset, val_dataset, feedback_dataset, _ = create_datasets({
            'vocab_size': self.config.vocab_size,
            'train_data_path': self.config.train_data_path,
            'val_data_path': self.config.val_data_path,
            'max_length': self.config.max_length,
            'feedback_enabled': self.config.feedback_enabled,
            'feedback_path': self.config.feedback_path,
            'feedback_ratio': self.config.feedback_ratio,
        })
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=train_dataset.get_collate_fn(),
            num_workers=0,
            pin_memory=self.device.type == 'cuda',
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=val_dataset.get_collate_fn(),
            num_workers=0,
            pin_memory=self.device.type == 'cuda',
        )
        
        feedback_loader = None
        if feedback_dataset:
            feedback_loader = DataLoader(
                feedback_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=feedback_dataset.get_collate_fn(),
                num_workers=0,
                pin_memory=self.device.type == 'cuda',
            )
        
        self.logger.info(f"Train dataset: {len(train_dataset)} samples")
        self.logger.info(f"Val dataset: {len(val_dataset)} samples")
        if feedback_dataset:
            self.logger.info(f"Feedback dataset: {len(feedback_dataset)} samples")
        
        return train_loader, val_loader, feedback_loader
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step with STE gradient flow.
        
        Args:
            batch: Dict with 'input_ids' and 'labels'
            
        Returns:
            Dict with loss and metrics
        """
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass with AMP
        with torch.amp.autocast('cuda', dtype=self.dtype, enabled=self.config.amp and self.device.type == 'cuda'):
            outputs = self.model(input_ids, targets=labels)
            loss = outputs['loss']
            
            # Add router auxiliary losses if using MoE
            if hasattr(self.model, 'blocks'):
                for block in self.model.blocks:
                    if hasattr(block, 'ffn') and hasattr(block.ffn, 'router'):
                        router_losses = block.ffn.router.get_losses()
                        if 'aux_loss' in router_losses:
                            loss = loss + 0.01 * router_losses['aux_loss']
                        if 'z_loss' in router_losses:
                            loss = loss + 0.001 * router_losses['z_loss']
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.grad_accum
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Gradient clipping (after accumulation)
        if (self.step + 1) % self.config.grad_accum == 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Scheduler step
            self.scheduler.step()
        else:
            grad_norm = 0.0
        
        # Get current learning rates
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        return {
            'loss': loss.item() * self.config.grad_accum,
            'grad_norm': grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            'lr_trunk': lrs[0] if len(lrs) > 0 else 0,
            'lr_engram': lrs[1] if len(lrs) > 1 else 0,
            'lr_experts': lrs[2] if len(lrs) > 2 else 0,
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with torch.amp.autocast('cuda', dtype=self.dtype, enabled=self.config.amp and self.device.type == 'cuda'):
                outputs = self.model(input_ids, targets=labels)
                loss = outputs['loss']
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': OmegaConf.to_object(self.config),
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = Path(path).parent / 'best_model.pt'
            torch.save(checkpoint, best_path)
        
        self.logger.info(f"Checkpoint saved: {path}")
    
    def _load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.step = checkpoint.get('step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f"Resumed from checkpoint: {path} (step {self.step})")
    
    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.config.max_steps} steps")
        self.logger.info(f"Device: {self.device}, dtype: {self.dtype}")
        self.logger.info(f"Batch size: {self.config.batch_size}, Grad accum: {self.config.grad_accum}")
        
        self.model.train()
        step_start_time = time.time()
        
        # Iterate over epochs
        while self.step < self.config.max_steps:
            self.epoch += 1
            
            for batch in self.train_loader:
                if self.step >= self.config.max_steps:
                    break
                
                # Training step
                metrics = self.train_step(batch)
                self.train_losses.append(metrics['loss'])
                
                # Logging
                if self.step % self.config.log_every == 0:
                    step_time = time.time() - step_start_time
                    self.logger.info(
                        f"Step {self.step}/{self.config.max_steps} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Grad norm: {metrics['grad_norm']:.4f} | "
                        f"LRs: [{metrics['lr_trunk']:.2e}, {metrics['lr_engram']:.2e}, {metrics['lr_experts']:.2e}] | "
                        f"Time: {step_time:.2f}s"
                    )
                    step_start_time = time.time()
                
                # Evaluation
                if self.step % self.config.eval_every == 0 and self.step > 0:
                    val_metrics = self.evaluate()
                    self.val_losses.append(val_metrics['val_loss'])
                    
                    self.logger.info(f"Validation loss: {val_metrics['val_loss']:.4f}")
                    
                    # Save best model
                    is_best = val_metrics['val_loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics['val_loss']
                    
                    self.save_checkpoint(
                        Path(self.config.output_dir) / f'checkpoint_step_{self.step}.pt',
                        is_best=is_best
                    )
                
                # Regular checkpoint
                if self.step % self.config.checkpoint_every == 0 and self.step > 0:
                    self.save_checkpoint(
                        Path(self.config.output_dir) / f'checkpoint_step_{self.step}.pt'
                    )
                
                self.step += 1
            
            self.model.train()
        
        # Final save
        self.save_checkpoint(Path(self.config.output_dir) / 'final_model.pt')
        self.logger.info("Training completed!")


def create_trainer(config: DictConfig | dict) -> TernaryTrainer:
    """Create TernaryTrainer from config."""
    return TernaryTrainer(config)


def train_model(config_path: str):
    """Train model from config file."""
    config = OmegaConf.load(config_path)
    trainer = create_trainer(config)
    trainer.train()


if __name__ == "__main__":
    # Quick test with dummy config
    print("Testing TernaryTrainer...")
    
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
    )
    
    trainer = TernaryTrainer(config)
    print(f"Model created: {sum(p.numel() for p in trainer.model.parameters())} params")
    print(f"Optimizer groups: {len(trainer.optimizer.param_groups)}")
    
    # Test single step
    dummy_batch = {
        'input_ids': torch.randint(0, 1000, (2, 64)),
        'labels': torch.randint(0, 1000, (2, 64)),
    }
    metrics = trainer.train_step(dummy_batch)
    print(f"Train step metrics: {metrics}")
    
    print("\nAll tests passed!")