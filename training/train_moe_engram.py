"""
Training Loop for NCM + MoE + Engram + KV Cache

Two-timescale optimizer:
- Trunk (attention, embeddings, heads, engram projections): LR 3e-4
- Engram tables: LR 1e-4 (slower)
- MoE experts: LR 3e-3 (10x trunk)
"""

import os
# Memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from omegaconf import OmegaConf
import logging
from tqdm import tqdm
import math
import pickle

from models.moe_engram_generator import MoEEngramNextConceptTransformer, create_moe_engram_model
from models.common import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler('training.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TinyStoriesDataset(Dataset):
    """Pre-tokenized TinyStories dataset from hazyresearch/ncm-datasets format."""
    
    def __init__(self, data_dir: str, context_length: int = 128):
        self.context_length = context_length
        self.data_dir = data_dir
        
        # Load the hazyresearch/ncm-datasets format
        # train_bpe_8192.bin + train_bpe_8192_offsets.npy + train_bpe_8192.pickle
        bin_path = os.path.join(data_dir, 'train_bpe_8192.bin')
        offsets_path = os.path.join(data_dir, 'train_bpe_8192_offsets.npy')
        pickle_path = os.path.join(data_dir, 'train_bpe_8192.pickle')
        
        # Memory-map the binary data
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.offsets = np.load(offsets_path)
        
        # Also load validation data
        val_bin_path = os.path.join(data_dir, 'valid_bpe_8192.bin')
        val_pickle_path = os.path.join(data_dir, 'valid_bpe_8192.pickle')
        if os.path.exists(val_bin_path):
            self.val_tokens = np.memmap(val_bin_path, dtype=np.uint16, mode='r')
            self.val_data = pickle.load(open(val_pickle_path, 'rb'))
        else:
            self.val_tokens = None
            self.val_data = None
        
        # Build chunks from training data
        self.chunks = []
        self._build_chunks()
        
        logger.info(f"Dataset: {len(self.chunks)} chunks of length {context_length}")
    
    def _build_chunks(self):
        """Build context-length chunks from the token sequences using offsets."""
        context_length = self.context_length
        
        # Iterate through each story using offsets
        for i in range(len(self.offsets) - 1):
            start = self.offsets[i]
            end = self.offsets[i + 1]
            seq = self.tokens[start:end].tolist()
            
            if len(seq) >= context_length:
                # Chunk into overlapping windows
                for j in range(0, len(seq) - context_length + 1, context_length // 2):
                    self.chunks.append(seq[j:j + context_length])
            else:
                # Pad short sequences
                padded = seq + [0] * (context_length - len(seq))
                self.chunks.append(padded[:context_length])
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return torch.tensor(self.chunks[idx], dtype=torch.long)


def prepare_dataset():
    """Verify dataset exists (download handled externally)."""
    data_dir = "data/tinystories/tinystories"
    
    # Check for the hazyresearch/ncm-datasets format files
    bin_path = os.path.join("data/tinystories/tinystories", 'train_bpe_8192.bin')
    offsets_path = os.path.join("data/tinystories/tinystories", 'train_bpe_8192_offsets.npy')
    pickle_path = os.path.join("data/tinystories/tinystories", 'train_bpe_8192.pickle')
    
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Dataset binary not found at {bin_path}")
    if not os.path.exists(offsets_path):
        raise FileNotFoundError(f"Dataset offsets not found at {offsets_path}")
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Dataset pickle not found at {pickle_path}")
    
    # Return the data directory for the dataset class
    return "data/tinystories/tinystories"
def get_optimizer(model, lr_trunk=3e-4, lr_engram=1e-4, lr_expert=3e-3, weight_decay=0.1):
    """
    Two-timescale optimizer.
    
    Groups:
    - trunk: attention, embeddings, heads, engram projections, routers, norms
    - engram: engram embedding tables
    - experts: MoE expert weights (only resident experts)
    """
    trunk_params = []
    engram_params = []
    expert_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'engrams' in name and 'embedding' in name:
            engram_params.append(param)
        elif 'blocks' in name and 'mlp' in name and 'expert' in name:
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


def train_step(model, batch, tokenizer, optimizer, scaler, accum_steps, step):
    """Single training step with gradient accumulation."""
    
    model.train()
    
    # Move to device
    device = next(model.parameters()).device
    batch = batch.to(device)
    
    # Get NCM inputs from tokenizer (VQ-VAE encoding)
    with torch.no_grad():
        x, targets = tokenizer.vqvae.get_ncm_inputs(batch)
        # x: [B, sum_levels, embed_dim] concatenated multi-level codes
        # targets: [B, sum_levels] flattened target indices for all levels
    
    # Forward with autocast
    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
        # NCM processes all levels simultaneously with block-diagonal attention
        # Forward returns logits for all levels concatenated and loss
        logits, loss = model(x, targets=targets)
        
        # Collect router losses from MoE blocks
        router_losses = []
        for block in model.blocks:
            if hasattr(block.mlp, 'router_loss'):
                router_losses.append(block.mlp.router_loss)
        
        # Average router loss
        router_loss = torch.stack(router_losses).mean() if router_losses else torch.tensor(0.0, device=device)
        total_loss = loss + 0.01 * router_loss
    
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
        optimizer.zero_grad(set_none=True)
    
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
    logger.info("Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_moe_engram_model('artifacts/tinystories/config.yaml', expert_dir='experts')
    model = model.to(device)
    
    logger.info(f"Model on {device}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    
    # Also load tokenizer for encoding (create_moe_engram_model already creates one, but we need it for get_ncm_inputs)
    from tokenization.vqvae_tokenizer import VQVAETokenizer
    tokenizer = VQVAETokenizer.from_config(config.tokenizer, base_dir='artifacts/tinystories')
    tokenizer.vqvae.eval()
    for p in tokenizer.vqvae.parameters():
        p.requires_grad = False
    tokenizer.vqvae.to(device)
    # Convert tokenizer to bfloat16 to match model
    tokenizer.vqvae = tokenizer.vqvae.to(torch.bfloat16)
    logger.info("Preparing dataset...")
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
    
    logger.info(f"Starting training: {max_steps} steps, batch=4, accum={accum_steps}")
    
    model.train()
    
    for epoch in range(100):  # enough epochs
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            loss, router_loss = train_step(model, batch, tokenizer, optimizer, scaler, accum_steps, step)
            
            if (step + 1) % accum_steps == 0:
                scheduler.step()
                
                if step % 500 == 0:
                    logger.info(f"Step {step}: loss={loss:.4f}, router_loss={router_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
                
                # Periodic checkpoint
                if step % 1000 == 0:
                    save_checkpoint(model, optimizer, scheduler, step, f'checkpoints/step_{step}.pt')
                    logger.info(f"Checkpoint saved at step {step}")
                
                # Periodic cache clear to reduce fragmentation
                if step % 100 == 0:
                    torch.cuda.empty_cache()
                
                if step >= max_steps:
                    logger.info("Training complete!")
                    save_checkpoint(model, optimizer, scheduler, step, f'checkpoints/step_{step}.pt')
                    return
                
                step += 1


if __name__ == "__main__":
    main()