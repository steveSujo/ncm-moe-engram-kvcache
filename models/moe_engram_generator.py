"""
MoE + Engram + KV Cache Enhanced NCM Generator

Integrates:
- NCM parallel coarse-to-fine transformer
- MoE FFN replacement (mini-AGI paging)
- Engram memory at layers 2, 15
- Bounded KV cache with attention sinks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from omegaconf import OmegaConf
import os

from models.common import Block, SelfAttention, MLP, RMSNorm
from models.engram import EngramModule, needle2_engram_config, engram_geometry
from models.kv_cache import BoundedKVCache, MultiLayerKVCache
from models.moe_block import MoEBlock
from utils.expert_io import MoEPager


class MoEEngramNextConceptTransformer(nn.Module):
    """
    Enhanced NCM Generator with:
    - MoE FFN layers (paged from SSD)
    - Engram memory at layers 2, 15
    - Bounded KV cache (256 window + 1 sink + INT8)
    """
    
    def __init__(
        self,
        vqvae_local,
        config,
        expert_dir: str = "experts",
        resident_slots: int = 32,
        engram_config: Optional[dict] = None,
        kv_cache_config: Optional[dict] = None,
        pager: Optional["MoEPager"] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.vqvae_local = vqvae_local
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.expert_dir = expert_dir
        
        # Extract model params
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        self.context_length = config.context_length
        self.prefix_len = config.prefix_len
        self.tie_heads_to_codebook = config.tie_heads_to_codebook
        
        # Embeddings
        vocab_size_0 = int(config.vocab_size[0]) if hasattr(config.vocab_size, '__getitem__') else int(config.vocab_size)
        self.wte = nn.Embedding(vocab_size_0, config.n_embd)
        self.wle = nn.Embedding(len(config.vocab_size), config.n_embd) if config.use_level_encoding else None
        # Use expanded length from VQ-VAE (640) for positional encoding
        self.wpe = nn.Embedding(640, config.n_embd) if config.use_positional_encoding else None
        self.drop = nn.Dropout(config.dropout)
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config.n_layer)
        ])
        
        # Replace MLP with MoEBlock
        if pager is None:
            from utils.expert_io import MoEPager
            pager = MoEPager(
                expert_dir=expert_dir,
                resident_slots=resident_slots,
                n_layers=config.n_layer,
                n_experts=64,
                d_model=config.n_embd,
                expert_hidden=3072,
                device=self.device,
            )
        
        for i, block in enumerate(self.blocks):
            block.mlp = MoEBlock(config, layer_idx=i, pager=pager)
        
        self.pager = pager
        self.ln_f = RMSNorm(config.n_embd)
        
        # Output heads (tied to codebook - 384 dim from VQ-VAE)
        codebook_dim = 384
        self.lm_heads = nn.ModuleList([
            nn.Linear(codebook_dim, vs, bias=False) for vs in config.vocab_size
        ])
        
        # Projection from codebook dim to model dim (matches NCM transformer.wte: [768, 384])
        self.proj_from_codebook = nn.Linear(codebook_dim, config.n_embd, bias=False)
        
        # Projection from model dim to codebook dim
        self.proj_to_codebook = nn.Linear(config.n_embd, codebook_dim, bias=False)
        engram_cfg = engram_config or needle2_engram_config()
        orders, num_heads, sub_dim = engram_geometry(engram_cfg)
        self.engram_layers = [2, 15] if config.n_layer > 15 else [2, min(11, config.n_layer - 1)]
        self.engrams = nn.ModuleList([
            EngramModule(
                config.n_embd, orders, num_heads, sub_dim,
                engram_cfg["slots"], engram_cfg["conv_taps"],
                engram_cfg["conv_dilation"]
            )
        ])
        kv_cfg = kv_cache_config or {}
        self.kv_cache = MultiLayerKVCache(
            n_layers=config.n_layer,
            max_window=kv_cfg.get("max_window", 256),
            sink_tokens=kv_cfg.get("sink_tokens", config.prefix_len),
            n_kv_heads=kv_cfg.get("n_kv_heads", config.n_head),  # MHA for compatibility
            head_dim=kv_cfg.get("head_dim", config.n_embd // config.n_head),
        )
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights if needed
        if self.tie_heads_to_codebook:
            self._tie_heads_to_codebook()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def _tie_heads_to_codebook(self):
        """Tie output heads to VQ codebook embeddings."""
        codebooks = self.vqvae_local.get_codebooks()  # [n_levels, vocab_size, embed_dim]
        for level, head in enumerate(self.lm_heads):
            if level < len(codebooks):
                head.weight.data = codebooks[level].to(head.weight.dtype).clone()
    
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[MultiLayerKVCache] = None,
    ):
        """Main forward - dispatches to multilevel if input is 3D (concatenated), else single level."""
        if x.dim() == 3:
            return self.forward_multilevel(x, targets)
        else:
            # Single level forward (for generation) - x is [B, T] token indices
            # Need to know which level - default to level 0
            return self.forward_single_level(x, level=0, targets=targets, kv_cache=kv_cache)
    def forward_multilevel(
        self,
        x: torch.Tensor,  # [B, sum_levels, embed_dim] concatenated multi-level codes
        targets: Optional[torch.Tensor] = None,  # [B, sum_levels] flattened
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for all levels simultaneously (NCM style).
        
        Args:
            x: [B, sum_levels, embed_dim] concatenated multi-level codes from VQ-VAE
            targets: [B, sum_levels] flattened target indices for all levels
        Returns:
            logits: [B, sum_levels, vocab_size] concatenated logits for all levels
            loss: scalar if targets provided
        """
        B, sum_T, embed_dim = x.shape
        device = x.device
        dtype = next(self.parameters()).dtype
        
        # x is already flattened: [B, sum_levels, embed_dim]
        x_flat = x
        
        # Project from VQ-VAE codebook dim (384) to model dim (768)
        # This matches NCM's transformer.wte: [768, 384]
        x_flat = self.proj_from_codebook(x_flat)
        
        # Add level embeddings using lvl_1L (like NCM)
        if self.wle is not None and hasattr(self, 'lvl_1L'):
            lvl_emb = self.wle(self.lvl_1L).squeeze(0)  # [sum_levels, n_embd]
            x_flat = x_flat + lvl_emb.unsqueeze(0)
        elif self.wle is not None:
            # Fallback: create level indices
            level_indices = torch.arange(len(self.vocab_size), device=device).repeat_interleave(
                torch.tensor([min(128, vs) for vs in self.vocab_size], device=device)
            )
            lvl_emb = self.wle(level_indices)  # [sum_levels, n_embd]
            x_flat = x_flat + lvl_emb.unsqueeze(0)
        # Add positional embeddings
        if self.wpe is not None:
            pos = torch.arange(sum_T, device=device).unsqueeze(0)
            pos_emb = self.wpe(pos)
            x_flat = x_flat + pos_emb
        
        x_flat = self.drop(x_flat)
        x_flat = x_flat.to(dtype)
        
        # Forward through blocks
        for i, block in enumerate(self.blocks):
            # Engram injection at specified layers
            if i in self.engram_layers:
                engram_idx = self.engram_layers.index(i)
                # For Engram, we need token indices - use level info
                # Create dummy token indices for hashing with correct batch size
                dummy_idx = torch.arange(sum_T, device=device).unsqueeze(0).expand(B, -1)  # [B, sum_T]
                engram_out = self.engrams[engram_idx].apply_engam(x_flat, dummy_idx)
                x_flat = engram_out
            
            # Block forward (without KV cache for training)
            x_flat = block(x_flat, kv_cache=None, layer_idx=i)
        
        x_flat = self.ln_f(x_flat)
        
        # Project to codebook dimension
        x_flat = self.proj_to_codebook(x_flat)
        
        # Apply level-specific heads
        # Need to split by level boundaries
        logits_list = []
        start = 0
        for level in range(len(self.vocab_size)):
            level_T = self.vocab_size[level]
            level_T = min(level_T, 128)  # cap at context length
            end = start + level_T
            level_x = x_flat[:, start:end, :]
            level_logits = self.lm_heads[level](level_x)  # [B, level_T, vocab_size]
            logits_list.append(level_logits)
            start = end
        
        logits = torch.cat(logits_list, dim=1)  # [B, sum_levels, vocab_size]
        
        loss = None
        if targets is not None:
            # targets is [B, sum_levels] flattened
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def forward_single_level(
        self,
        idx: torch.Tensor,  # [B, T] token indices
        level: int = 0,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[MultiLayerKVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        dtype = next(self.parameters()).dtype
        
        # Token embeddings
        tok_emb = self.wte(idx)  # [B, T, n_embd]
        
        # Level embedding
        if self.wle is not None:
            lvl_emb = self.wle(torch.full((B, T), level, dtype=torch.long, device=device))
            tok_emb = tok_emb + lvl_emb
        
        # Positional embeddings
        if self.wpe is not None:
            pos = torch.arange(T, device=device).unsqueeze(0)
            pos_emb = self.wpe(pos)
            tok_emb = tok_emb + pos_emb
        
        x = self.drop(tok_emb)
        x = x.to(dtype)  # Ensure model dtype (BF16)
        
        # Use provided KV cache or internal
        cache = kv_cache or self.kv_cache
        
        # Forward through blocks
        for i, block in enumerate(self.blocks):
            # Engram injection at specified layers
            if i in self.engram_layers:
                engram_idx = self.engram_layers.index(i)
                # Engram applies to residual stream before attention
                # Need tokens for hashing - use input idx
                engram_out = self.engrams[engram_idx].apply_engam(x, idx)
                x = engram_out
            
            # Block forward with KV cache
            x = block(x, kv_cache=cache, layer_idx=i)
        
        x = self.ln_f(x)
        
        # Project to codebook dimension
        x = self.proj_to_codebook(x)
        
        # Level-specific head
        logits = self.lm_heads[level](x)  # [B, T, vocab_size]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def generate(
        self,
        idx: torch.Tensor,
        l0: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        cfg_scale: float = 0.0,
        max_levels: int = 11,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation across all levels.
        
        NCM generates level by level (coarse to fine).
        """
        self.eval()
        
        # Reset KV cache for generation
        if use_kv_cache:
            self.kv_cache.reset()
        
        # Level 0: unigram or provided
        if l0 is None:
            # Sample from unigram
            unigram = self.vqvae_local.get_unigram()  # [vocab_size]
            l0 = torch.multinomial(unigram, num_samples=idx.size(0), replacement=True)
            l0 = l0.to(idx.device).unsqueeze(-1)  # [B, 1]
        else:
            l0 = l0.unsqueeze(-1) if l0.dim() == 1 else l0
        
        generated = [l0]
        
        for level in range(1, max_levels):
            # Build input for this level: prefix + all previous levels
            # NCM uses hierarchical conditioning
            # For simplicity, concatenate all previous level tokens
            ctx = torch.cat(generated, dim=-1)  # [B, sum_T]
            
            # Forward with KV cache
            logits, _ = self.forward_single_level(ctx, level=level, kv_cache=self.kv_cache if use_kv_cache else None)
            
            # Sample next token
            logits = logits[:, -1, :] / temperature  # [B, vocab]
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumprobs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token)
        
        return torch.cat(generated, dim=-1)
    
    def load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained dense weights and convert to MoE.
        
        Note: NCM architecture differs from our MoE+Engram architecture:
        - NCM transformer.wte: [768, 384] projects codebook_dim -> n_embd (NOT token embedding)
        - NCM lm_heads: [384, 768] projects n_embd -> codebook_dim (tied to codebook)
        - Our wte: [vocab_size_0, n_embd] token embedding for level 0
        - Our lm_heads: [n_embd, vocab_size] output heads
        
        So we only load compatible parts: attention, layernorms, wle, wpe.
        Experts are initialized from dense MLP via k-means.
        """
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['model']
        
        # Load level embedding (compatible: [11, 768])
        if self.wle is not None and 'transformer.wle.weight' in state_dict:
            self.wle.load_state_dict({'weight': state_dict['transformer.wle.weight']})
        
        # Load positional embedding (compatible: [129, 768])
        if self.wpe is not None and 'transformer.wpe.weight' in state_dict:
            self.wpe.load_state_dict({'weight': state_dict['transformer.wpe.weight']})
        
        # Load transformer blocks (attention + layernorms)
        for i, block in enumerate(self.blocks):
            # Attention
            block.attn.c_attn.weight.data = state_dict[f'transformer.blocks.{i}.attn.c_attn.weight']
            block.attn.c_attn.bias.data = state_dict[f'transformer.blocks.{i}.attn.c_attn.bias']
            block.attn.c_proj.weight.data = state_dict[f'transformer.blocks.{i}.attn.c_proj.weight']
            block.attn.c_proj.bias.data = state_dict[f'transformer.blocks.{i}.attn.c_proj.bias']
            block.attn.q_norm.weight.data = state_dict[f'transformer.blocks.{i}.attn.q_norm.weight']
            block.attn.k_norm.weight.data = state_dict[f'transformer.blocks.{i}.attn.k_norm.weight']
            
            # Layer norms - NCM uses LayerNorm with bias, we use RMSNorm (no bias)
            block.ln_1.load_state_dict({
                'weight': state_dict[f'transformer.blocks.{i}.ln_1.weight'],
            }, strict=False)
            block.ln_2.load_state_dict({
                'weight': state_dict[f'transformer.blocks.{i}.ln_2.weight'],
            }, strict=False)
            
            # Initialize MoE experts from dense MLP (k-means)
            dense_fc = state_dict[f'transformer.blocks.{i}.mlp.c_fc.weight']  # [3072, 768]
            dense_proj = state_dict[f'transformer.blocks.{i}.mlp.c_proj.weight']  # [768, 3072]
            self._init_experts_from_dense(i, dense_fc, dense_proj)
        
        # Final norm - NCM uses LayerNorm with bias, we use RMSNorm (no bias)
        self.ln_f.load_state_dict({
            'weight': state_dict['transformer.ln_f.weight'],
        }, strict=False)
        
        # lvl_1L if present
        if 'lvl_1L' in state_dict:
            self.register_buffer('lvl_1L', state_dict['lvl_1L'])
        
        # Level 0 unigram for generation
        if 'inference' in ckpt and 'level0_unigram' in ckpt['inference']:
            self.register_buffer('level0_unigram', ckpt['inference']['level0_unigram'].to(self.device))
        
        # Tie heads to codebook (our architecture)
        self._tie_heads_to_codebook()
        
        print("Loaded pretrained weights (compatible parts only)")
    
    def _init_experts_from_dense(self, layer_idx: int, dense_fc: torch.Tensor, dense_proj: torch.Tensor):
        """
        Initialize 64 experts from dense MLP using k-means clustering.
        
        dense_fc: [3072, 768] - up projection
        dense_proj: [768, 3072] - down projection
        
        This is a fallback if expert checkpoints don't exist.
        """
        # Check if experts already exist on disk
        expert_path = os.path.join(self.expert_dir, f"layer{layer_idx}_expert0.safetensors")
        if os.path.exists(expert_path):
            print(f"Experts for layer {layer_idx} already exist, skipping k-means init")
            return
        
        from sklearn.cluster import KMeans
        import numpy as np
        
        # For SwiGLU, we need gate, up, down
        # Original NCM uses standard MLP: x -> fc -> gelu -> proj
        # We'll split fc into gate and up (both [3072, 768])
        d_model, hidden = dense_fc.shape[1], dense_fc.shape[0]
        
        # Combine gate and up as single matrix for clustering
        # [2 * hidden, d_model]
        combined = torch.cat([dense_fc, dense_fc], dim=0).numpy()  # duplicate for gate/up init
        
        # K-means on rows
        kmeans = KMeans(n_clusters=64, random_state=42, n_init=10)
        kmeans.fit(combined)
        
        centroids = torch.from_numpy(kmeans.cluster_centers_).to(dense_fc.dtype)
        
        # Split centroids into gate and up
        gate_centroids = centroids[:hidden]  # [64, 768]
        up_centroids = centroids[hidden:]    # [64, 768]
        down_centroids = dense_proj.T.numpy()  # [3072, 768]
        kmeans_down = KMeans(n_clusters=64, random_state=42, n_init=10)
        kmeans_down.fit(down_centroids)
        down_centroids = torch.from_numpy(kmeans_down.cluster_centers_).to(dense_proj.dtype)  # [64, 768]
        down_centroids = down_centroids.T  # [768, 64]
        
        # Save each expert
        pager = self.blocks[layer_idx].mlp.pager
        for e in range(64):
            expert_tensors = {
                "gate": gate_centroids[e].unsqueeze(0).repeat(hidden, 1) + torch.randn(hidden, d_model) * 0.01,
                "up": up_centroids[e].unsqueeze(0).repeat(hidden, 1) + torch.randn(hidden, d_model) * 0.01,
                "down": down_centroids[:, e].unsqueeze(1).repeat(1, hidden) + torch.randn(d_model, hidden) * 0.01,
            }
            pager.save_expert(layer_idx, e, expert_tensors)


def create_moe_engram_model(config_path: str, expert_dir: str = "experts") -> MoEEngramNextConceptTransformer:
    """Factory function to create model from config."""
    import os
    config = OmegaConf.load(config_path)
    
    # Load tokenizer (frozen) - need base_dir for checkpoint paths
    base_dir = os.path.dirname(config_path)
    from tokenization.vqvae_tokenizer import VQVAETokenizer
    tokenizer = VQVAETokenizer.from_config(config.tokenizer, base_dir=base_dir)
    tokenizer.vqvae.eval()
    for p in tokenizer.vqvae.parameters():
        p.requires_grad = False
    
    model = MoEEngramNextConceptTransformer(
        vqvae_local=tokenizer.vqvae,
        config=config.model,
        expert_dir=expert_dir,
    )
    
    # Load pretrained weights
    generator_path = os.path.join(base_dir, config.checkpoints.generator)
    model.load_pretrained_weights(generator_path)
    
    return model


if __name__ == "__main__":
    # Test model creation
    import torch
    from omegaconf import OmegaConf
    
    config = OmegaConf.load('artifacts/tinystories/config.yaml')
    model = create_moe_engram_model('artifacts/tinystories/config.yaml', expert_dir='experts')
    print(f"Model created: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    
    # Test forward_multilevel
    B, T = 2, 128
    num_levels = len(config.model.vocab_size)
    x = torch.randn(B, num_levels, T, 384)
    targets = torch.randint(0, 4096, (B, num_levels * T))
    
    model.eval()
    with torch.no_grad():
        logits, loss = model.forward_multilevel(x, targets)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print("Model test passed!")