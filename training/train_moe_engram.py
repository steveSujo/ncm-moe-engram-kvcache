"""
Training Loop for NCM + MoE + Engram + KV Cache

Two-timescale optimizer:
- Trunk (attention, embeddings, heads, engram projections): LR 3e-4
- Engram tables: LR 1e-4 (slower)
- MoE experts: LR 3e-3 (10x trunk)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from omegaconf import OmegaConf
import os
import pickle
from tqdm import tqdm
import math

from models.moe_engram_generator import MoEEngramNextConceptTransformer, create_moe_engram_model
from models.common import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer


class TinyStoriesDataset(Dataset):
    """Pre-tokenized TinyStories dataset."""
    
    def __init__(self, data_path: str, context_length: int = 128):
        self.context_length = context_length
        
        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            # Raw text -> tokenize on the fly
            raise NotImplementedError("Only pre-tokenized .pkl supported")
        
        # Flatten into chunks of context_length
        self.chunks = []
        for seq in self.data:
            for i in range(0, len(seq) - context_length, context_length // 2):
                self.chunks.append(seq[i:i + context_length])
        
        print(f"Dataset: {len(self.chunks)} chunks of length {context_length}")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return torch.tensor(self.chunks[idx], dtype=torch.long)


def prepare_dataset():
    """Download and prepare TinyStories dataset."""
    from huggingface_hub import snapshot_download
    
    # Download NCM datasets
    os.makedirs("data/tinystories", exist_ok=True)
    snapshot_download(
        repo_id="hazyresearch/ncm-datasets",
        local_dir="data/tinystories",
        local_dir_use_symlinks=False,
        allow_patterns=["tinystories/*"]
    )
    
    # The dataset should have train.pkl
    return "data/tinystories/tinystories/train.pkl"


def get_optimizer(model, lr_trunk=3e-4, lr_engram=1e-4, lr_expert=3e-3, weight_decay=0.1):
    """
    Two-timescale optimizer.
    
    Groups:
    - trunk: attention, embeddings, heads, engram projections, routers, norms
    - engram_tables: Engram embedding tables (can be slower)
    - experts: MoE expert weights (dynamic, higher LR)
    """
    
    trunk_params = []
    engram_table_params = []
    expert_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'expert' in name or 'mlp.gate' in name or 'mlp.up' in name or 'mlp.down' in name:
            expert_params.append(param)
        elif 'engram.embeddings' in name or 'engram.embedding' in name:
            engram_table_params.append(param)
        else:
            trunk_params.append(param)
    
    print(f"Parameter groups:")
    print(f"  Trunk: {sum(p.numel() for p in trunk_params)/1e6:.1f}M params")
    print(f"  Engram tables: {sum(p.numel() for p in engram_table_params)/1e6:.1f}M params")
    print(f"  Experts: {sum(p.numel() for p in expert_params)/1e6:.1f}M params")
    
    optimizer = torch.optim.AdamW([
        {'params': trunk_params, 'lr': lr_trunk, 'weight_decay': weight_decay},
        {'params': engram_table_params, 'lr': lr_engram, 'weight_decay': 0.01},
        {'params': expert_params, 'lr': lr_expert, 'weight_decay': weight_decay},
    ], betas=(0.9, 0.95), fused=True)
    
    return optimizer


def get_scheduler(optimizer, warmup_steps=1000, max_steps=50000):
    """Cosine decay with warmup."""
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(model, batch, tokenizer, optimizer, scaler, accum_steps, step):
    """Single training step with gradient accumulation."""
    
    model.train()
    
    # Move to device
    device = next(model.parameters()).device
    batch = batch.to(device)
    
    # Get NCM inputs from tokenizer (VQ-VAE encoding)
    with torch.no_grad():
        x, targets = tokenizer.vqvae.get_ncm_inputs(batch)
        # x: [B, L, D] multi-level tokens
        # targets: [B, 128] flattened targets for all levels
    
    # Forward with autocast
    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
        # Multi-level forward
        total_loss = 0
        router_losses = []
        
        # For each level, compute logits and loss
        # This is simplified - actual NCM generates level by level
        for level in range(len(model.vocab_size)):
            logits, loss = model(x, level=level, targets=targets)
            total_loss += loss
            
            # Collect router losses from MoE blocks
            for block in model.blocks:
                if hasattr(block.mlp, 'router_loss'):
                    router_losses.append(block.mlp.router_loss)
        
        # Average router loss
        router_loss = torch.stack(router_losses).mean() if router_losses else torch.tensor(0.0, device=device)
        total_loss = total_loss + 0.01 * router_loss
    
    # Backward
    scaler.scale(total_loss / accum_steps).backward()
    
    # Gradient clipping at accumulation boundary
    if (step + 1) % accum_steps == 0:
        scaler.unscale_(optimizer)
        
        # Clip gradients (only trunk + engram + currently routed experts)
        all_params = []
        for group in optimizer.param_groups:
            all_params.extend(group['params'])
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    
    return total_loss.item(), router_loss.item()


def save_checkpoint(model, optimizer, scheduler, step, path):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
    }, path)
    
    # Also save resident experts
    for block in model.blocks:
        if hasattr(block.mlp, 'pager'):
            block.mlp.pager.save_all_resident()


def main():
    # Config
    config = OmegaConf.load('artifacts/tinystories/config.yaml')
    
    # Model
    print("Creating model...")
    model = create_moe_engram_model('artifacts/tinystories/config.yaml', expert_dir='experts')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model on {device}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    
    # Dataset
    print("Preparing dataset...")
    data_path = prepare_dataset()
    dataset = TinyStoriesDataset(data_path, context_length=config.model.context_length)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    
    # Optimizer
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, warmup_steps=1000, max_steps=50000)
    scaler = GradScaler()
    
    # Training loop
    max_steps = 50000
    accum_steps = 8
    step = 0
    
    print(f"Starting training: {max_steps} steps, batch=4, accum={accum_steps}")
    
    model.train()
    
    for epoch in range(100):  # enough epochs
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            loss, router_loss = train_step(model, batch, model.vqvae_local, optimizer, scaler, accum_steps, step)
            
            if (step + 1) % accum_steps == 0:
                scheduler.step()
                
                if step % 500 == 0:
                    print(f"Step {step}: loss={loss:.4f}, router_loss={router_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
                
                # Periodic checkpoint
                if step % 1000 == 0:
                    save_checkpoint(model, optimizer, scheduler, step, f'checkpoints/step_{step}.pt')
                    print(f"Checkpoint saved at step {step}")
                
                if step >= max_steps:
                    print("Training complete!")
                    save_checkpoint(model, optimizer, scheduler, step, f'checkpoints/step_{step}.pt')
                    return
                
                step += 1


if __name__ == "__main__":
    main()