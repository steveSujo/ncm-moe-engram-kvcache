"""
2-bit Training Loop for NCM + MoE + Engram + KV Cache

Native 2-bit training using Cactus Quants (CQ) from Needle:
- 2-bit CQ for weights, BF16 for activations/gradients
- STE (Straight-Through Estimator) for gradient flow
- Two-timescale optimizer: trunk 3e-4, engram 1e-4, experts 3e-3
- Gradient accumulation (accum=16 for effective batch 64)
- Gradient checkpointing on all blocks
"""

import os
# Memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf
import logging
from tqdm import tqdm
import math
import pickle
import argparse

from models.moe_engram_generator_2bit import MoEEngramNextConceptTransformer2Bit, create_moe_engram_model_2bit
from models.block_2bit import Block2Bit
from tokenization.vqvae_tokenizer import VQVAETokenizer
from utils.expert_io_2bit import CQQuantizedMoEPager


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler('training_2bit.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TinyStoriesDataset(Dataset):
    """Pre-tokenized TinyStories dataset."""
    
    def __init__(self, data_path: str, context_length: int = 128):
        self.context_length = context_length
        
        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            raise NotImplementedError("Only pre-tokenized .pkl supported")
        
        # Flatten into chunks of context_length
        self.chunks = []
        for seq in self.data:
            if len(seq) >= context_length:
                for i in range(0, len(seq) - context_length + 1, context_length // 2):
                    self.chunks.append(seq[i:i + context_length])
            else:
                padded = seq + [0] * (context_length - len(seq))
                self.chunks.append(padded[:context_length])
        logger.info(f"Dataset: {len(self.chunks)} chunks of length {context_length}")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return torch.tensor(self.chunks[idx], dtype=torch.long)


def prepare_dataset():
    """Download and prepare TinyStories dataset."""
    from huggingface_hub import snapshot_download
    
    os.makedirs("data/tinystories", exist_ok=True)
    snapshot_download(
        repo_id="hazyresearch/ncm-datasets",
        local_dir="data/tinystories",
        local_dir_use_symlinks=False,
        allow_patterns=["tinystories/*"]
    )
    
    path1 = "data/tinystories/tinystories/train.pkl"
    path2 = "data/tinystories/train.pkl"
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    else:
        raise FileNotFoundError(f"Dataset not found at {path1} or {path2}")


def get_optimizer_2bit(model, lr_trunk=3e-4, lr_engram=1e-4, lr_expert=3e-3, weight_decay=0.1):
    """
    Two-timescale optimizer for 2-bit model.
    
    Groups:
    - trunk: attention, embeddings, heads, engram projections, routers, norms (FP32 params)
    - engram: engram embedding tables (2-bit quantized, but FP32 master)
    - experts: MoE expert weights (2-bit quantized, but FP32 master)
    """
    trunk_params = []
    engram_params = []
    expert_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'engram' in name and ('embedding' in name or 'table' in name):
            engram_params.append(param)
        elif 'mlp' in name and ('expert' in name or 'gate' in name or 'up' in name or 'down' in name):
            expert_params.append(param)
        else:
            trunk_params.append(param)
    
    logger.info(f"Optimizer groups: trunk={sum(p.numel() for p in trunk_params)/1e6:.1f}M, "
                f"engram={sum(p.numel() for p in engram_params)/1e6:.1f}M, "
                f"experts={sum(p.numel() for p in expert_params)/1e6:.1f}M")
    
    optimizer = torch.optim.AdamW([
        {'params': trunk_params, 'lr': lr_trunk, 'weight_decay': weight_decay},
        {'params': engram_params, 'lr': lr_engram, 'weight_decay': 0.01},
        {'params': expert_params, 'lr': lr_expert, 'weight_decay': weight_decay},
    ])
    
    return optimizer


def get_scheduler(optimizer, warmup_steps=1000, max_steps=50000):
    """Cosine decay with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step_2bit(model, batch, tokenizer, optimizer, scaler, accum_steps, step, use_checkpointing=True):
    """Single 2-bit training step with gradient accumulation."""
    
    model.train()
    
    device = next(model.parameters()).device
    batch = batch.to(device)
    
    # Get NCM inputs from tokenizer (VQ-VAE encoding)
    with torch.no_grad():
        x, targets = tokenizer.vqvae.get_ncm_inputs(batch)
    
    # Forward with autocast (BF16 for activations)
    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
        # Use gradient checkpointing for memory efficiency
        def forward_fn(x):
            return model(x, targets=targets)
        
        if use_checkpointing and model.training:
            logits, loss = checkpoint(forward_fn, x, use_reentrant=False)
        else:
            logits, loss = forward_fn(x)
        
        # Collect router losses from MoE blocks
        router_losses = []
        for block in model.blocks:
            if hasattr(block.mlp, 'router_loss') and block.mlp.router_loss != 0:
                router_losses.append(block.mlp.router_loss)
        
        router_loss = torch.stack(router_losses).mean() if router_losses else torch.tensor(0.0, device=device)
        total_loss = loss + 0.01 * router_loss
    
    # Backward with gradient accumulation
    scaler.scale(total_loss / accum_steps).backward()
    
    # Gradient clipping at accumulation boundary
    if (step + 1) % accum_steps == 0:
        scaler.unscale_(optimizer)
        
        # Clip gradients for all params
        all_params = []
        for group in optimizer.param_groups:
            all_params.extend(group['params'])
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_none=True)
    
    return total_loss.item(), router_loss.item()


def save_checkpoint_2bit(model, optimizer, scheduler, step, path):
    """Save 2-bit training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
    }, path)
    
    # Also save resident experts in 2-bit format
    for block in model.blocks:
        if hasattr(block.mlp, 'pager'):
            block.mlp.pager.save_all_resident()


def main():
    parser = argparse.ArgumentParser(description='2-bit NCM Training')
    parser.add_argument('--config', type=str, default='artifacts/tinystories/config.yaml')
    parser.add_argument('--expert_dir', type=str, default='experts_2bit')
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accum_steps', type=int, default=16)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--lr_trunk', type=float, default=3e-4)
    parser.add_argument('--lr_engram', type=float, default=1e-4)
    parser.add_argument('--lr_expert', type=float, default=3e-3)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_2bit')
    parser.add_argument('--no_checkpointing', action='store_true', help='Disable gradient checkpointing')
    args = parser.parse_args()
    
    # Config
    config = OmegaConf.load(args.config)
    
    # Model
    logger.info("Creating 2-bit model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_moe_engram_model_2bit(args.config, expert_dir=args.expert_dir)
    model = model.to(device)
    
    logger.info(f"Model on {device}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    
    # Load tokenizer for encoding
    from tokenization.vqvae_tokenizer import VQVAETokenizer
    tokenizer = VQVAETokenizer.from_config(config.tokenizer, base_dir='artifacts/tinystories')
    tokenizer.vqvae.eval()
    for p in tokenizer.vqvae.parameters():
        p.requires_grad = False
    tokenizer.vqvae.to(device)
    tokenizer.vqvae = tokenizer.vqvae.to(torch.bfloat16)
    
    logger.info("Preparing dataset...")
    data_path = prepare_dataset()
    dataset = TinyStoriesDataset(data_path, context_length=config.model.context_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    # Optimizer
    optimizer = get_optimizer_2bit(
        model,
        lr_trunk=args.lr_trunk,
        lr_engram=args.lr_engram,
        lr_expert=args.lr_expert,
    )
    scheduler = get_scheduler(optimizer, warmup_steps=args.warmup_steps, max_steps=args.max_steps)
    scaler = GradScaler()
    
    # Training loop
    max_steps = args.max_steps
    accum_steps = args.accum_steps
    step = 0
    use_checkpointing = not args.no_checkpointing
    
    logger.info(f"Starting 2-bit training: {max_steps} steps, batch={args.batch_size}, accum={accum_steps} (effective batch={args.batch_size * accum_steps})")
    logger.info(f"Gradient checkpointing: {use_checkpointing}")
    
    model.train()
    
    for epoch in range(100):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            loss, router_loss = train_step_2bit(
                model, batch, tokenizer, optimizer, scaler, accum_steps, step, use_checkpointing
            )
            
            if (step + 1) % accum_steps == 0:
                scheduler.step()
                
                if step % 500 == 0:
                    logger.info(f"Step {step}: loss={loss:.4f}, router_loss={router_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
                
                # Periodic checkpoint
                if step % 1000 == 0:
                    save_checkpoint_2bit(model, optimizer, scheduler, step, f'{args.checkpoint_dir}/step_{step}.pt')
                    logger.info(f"Checkpoint saved at step {step}")
                
                # Periodic cache clear
                if step % 100 == 0:
                    torch.cuda.empty_cache()
                
                if step >= max_steps:
                    logger.info("Training complete!")
                    save_checkpoint_2bit(model, optimizer, scheduler, step, f'{args.checkpoint_dir}/step_{step}.pt')
                    return
                
                step += 1


if __name__ == "__main__":
    main()