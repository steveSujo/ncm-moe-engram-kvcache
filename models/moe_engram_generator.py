"""
MoE + Engram + KV Cache Enhanced NCM Generator
===============================================

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
from omegaconf import OmegaConf, ListConfig
import os

from models.common import Block, SelfAttention, MLP, RMSNorm
from models.engram import EngramModule, needle2_engram_config, engram_geometry
from models.kv_cache import BoundedKVCache, MultiLayerKVCache
from models.moe_block import MoEBlock
from models.block_2bit import Block2Bit
from utils.expert_io import MoEPager


class MoEEngramNextConceptTransformer(nn.Module):
    """
    NCM Generator with MoE + Engram + Bounded KV Cache.
    """
    
    def __init__(
        self,
        vqvae_local,
        config,
        expert_dir: str = "experts",
    ):
        super().__init__()
        self.vqvae_local = vqvae_local
        self.config = config
        self.expert_dir = expert_dir
        
        # Model dimensions
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.n_kv_head = getattr(config, "n_kv_head", config.n_head)
        self.head_dim = self.n_embd // self.n_head
        # vocab_size can be ListConfig (OmegaConf) or list/tuple/int
        from omegaconf import ListConfig
        if isinstance(config.vocab_size, (list, tuple, ListConfig)) and hasattr(config.vocab_size, '__len__'):
            self.vocab_size = int(config.vocab_size[0])
        else:
            self.vocab_size = int(config.vocab_size)
        self.max_seq_len = getattr(config, "max_seq_len", config.get("context_length", 128))
        
        # Engram configuration
        self.engram_layers = getattr(config, "engram_layers", [2, 11])
        self.engram_config = needle2_engram_config()
        if hasattr(config, "engram"):
            self.engram_config.update(config.engram)
        
        # Quantization settings
        self.bits = 2.0  # WEIGHT_BITS
        self.group_size = 128  # CQ_GROUP_SIZE
        
        # Token embeddings (keep BF16)
        # vocab_size can be a list (NCM multi-level), use first level
        vocab_size = int(config.vocab_size[0]) if hasattr(config.vocab_size, '__getitem__') else config.vocab_size
        self.wte = nn.Embedding(vocab_size, self.n_embd)
        self.wpe = nn.Embedding(self.max_seq_len, self.n_embd)
        
        # Level embeddings for NCM (keep BF16)
        self.n_levels = getattr(config, "n_levels", 11)
        self.wle = nn.Embedding(self.n_levels, self.n_embd)
        
        # Dropout
        self.drop = nn.Dropout(getattr(config, "embd_dropout", 0.0))
        
        # Expert pager (created before blocks)
        from utils.expert_io import MoEPager
        self.pager = MoEPager(
            expert_dir=expert_dir,
            n_layers=self.n_layer,
            n_experts=64,
            d_model=self.n_embd,
            expert_hidden=3072,
            resident_slots=getattr(config, "moe_resident_slots", 32),
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        
        # Transformer blocks (2-bit)
        self.blocks = nn.ModuleList([
            Block2Bit(
                config,
                layer_idx=i,
                pager=self.pager,  # Pass pager to blocks
                engram_layers=self.engram_layers,
                engram_config=self.engram_config,
            )
            for i in range(self.n_layer)
        ])
        
        # Final norm (FP32)
        self.ln_f = RMSNorm(self.n_embd, bias=config.bias)
        
        # Output heads per level (keep BF16)
        self.lm_heads = nn.ModuleList([
            nn.Linear(self.n_embd, self.vocab_size, bias=False)
            for _ in range(self.n_levels)
        ])
        
        # Tie weights
        for level in range(self.n_levels):
            self.lm_heads[level].weight = self.wte.weight
        
        # KV Cache (2-bit)
        self.kv_cache = None
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _tie_heads_to_codebook(self):
        """Tie output heads to VQ codebook embeddings."""
        codebooks = self.vqvae_local.get_codebooks()
        for level, head in enumerate(self.lm_heads):
            if level < len(codebooks):
                head.weight.data = codebooks[level].to(head.weight.dtype).clone()
    
    def init_kv_cache(self, batch_size: int, device: torch.device):
        """Initialize 2-bit KV cache."""
        self.kv_cache = MultiLayerKVCache2Bit(
            n_layers=self.n_layer,
            max_window=getattr(self.config, "kv_cache_window", 256),
            sink_tokens=getattr(self.config, "kv_cache_sink", 1),
            n_heads=self.n_kv_head,
            head_dim=self.head_dim,
            bits=self.bits,
            group_size=64,  # Smaller group for KV
            dtype=torch.bfloat16,
            device=device,
        )
    
    def reset_kv_cache(self):
        """Reset KV cache."""
        if self.kv_cache is not None:
            self.kv_cache.reset()
    
    def forward(
        self,
        idx: torch.Tensor,
        level: torch.Tensor = None,
        targets: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        rope_cos_sin: Tuple[torch.Tensor, torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            idx: [B, T] token indices
            level: [B, T] level indices (for NCM)
            targets: [B, T] target tokens (for loss computation)
            attn_mask: Attention mask
            rope_cos_sin: RoPE cos/sin tables
            use_cache: Whether to use KV cache
            
        Returns:
            logits: [B, T, V] or [B, T, n_levels, V] if level provided
            loss: Optional loss tensor
        """
        B, T = idx.shape
        device = idx.device
        
        # Initialize KV cache if needed
        if use_cache and self.kv_cache is None:
            self.init_kv_cache(B, device)
        
        # Embeddings
        tok_emb = self.wte(idx)  # [B, T, D]
        pos_emb = self.wpe(torch.arange(T, device=device).unsqueeze(0))  # [1, T, D]
        x = self.drop(tok_emb + pos_emb)
        
        # Add level embeddings if provided
        if level is not None:
            lvl_emb = self.wle(level)
            x = x + lvl_emb
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                attn_mask=attn_mask,
                rope_cos_sin=rope_cos_sin,
                kv_cache=self.kv_cache if use_cache else None,
            )
        
        # Final norm
        x = self.ln_f(x)
        
        # Output heads
        if level is not None:
            # Multi-level output: [B, T, n_levels, V]
            logits = torch.stack([head(x) for head in self.lm_heads], dim=2)
        else:
            # Single level (default level 0)
            logits = self.lm_heads[0](x)
        
        # Loss computation
        loss = None
        if targets is not None:
            if level is not None:
                # Multi-level loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.vocab_size), targets.view(-1))
        
        return logits, loss
    
    def generate(
        self,
        idx: torch.Tensor,
        level: torch.Tensor = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively with KV cache."""
        self.eval()
        
        # Enable KV cache for generation
        self.init_kv_cache(idx.shape[0], idx.device)
        
        for _ in range(max_new_tokens):
            # Forward with cache
            logits, _ = self.forward(
                idx[:, -1:] if idx.shape[1] > 0 else idx,
                level=level[:, -1:] if level is not None and level.shape[1] > 0 else level,
                use_cache=True,
            )
            
            # Get next token logits
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('inf')
            
            # Top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = -float('inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat([idx, next_token], dim=1)
            
            if level is not None:
                level = torch.cat([level, level[:, -1:]], dim=1)
            
            # Check EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return idx
    
    def prepare_for_inference(self):
        """Switch all quantized modules to inference mode."""
        for block in self.blocks:
            block.prepare_for_inference()
    
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
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt

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
            self.register_buffer('level0_unigram', ckpt['inference']['level0_unigram'].to(next(self.parameters()).device))

        # Tie heads to codebook (our architecture)
        self._tie_heads_to_codebook()

        print("Loaded pretrained weights (compatible parts only)")

    def _init_experts_from_dense(self, layer_idx: int, dense_fc: torch.Tensor, dense_proj: torch.Tensor):
        """
        Initialize 64 experts from dense MLP using k-means clustering.

        dense_fc: [3072, 768] - up projection (hidden=3072, d_model=768)
        dense_proj: [768, 3072] - down projection (d_model=768, hidden=3072)

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
        # dense_fc: [hidden=3072, d_model=768] - shared up projection
        # dense_proj: [d_model=768, hidden=3072] - down projection
        hidden, d_model = dense_fc.shape[0], dense_fc.shape[1]
        
        # For SwiGLU: 64 experts = 32 gate centroids + 32 up centroids
        # Cluster the up projection weights into 32 clusters for up, 32 for gate
        from sklearn.cluster import KMeans
        import numpy as np
        
        # Cluster up projection weights into 32 up centroids
        kmeans_up = KMeans(n_clusters=32, random_state=42, n_init=10)
        kmeans_up.fit(dense_fc.numpy())
        up_centroids = torch.from_numpy(kmeans_up.cluster_centers_).to(dense_fc.dtype)  # [32, d_model]
        
        # Use different seed for gate centroids
        kmeans_gate = KMeans(n_clusters=32, random_state=123, n_init=10)
        kmeans_gate.fit(dense_fc.numpy())
        gate_centroids = torch.from_numpy(kmeans_gate.cluster_centers_).to(dense_fc.dtype)  # [32, d_model]
        
        # Cluster down projection weights into 64 clusters (one per expert)
        down_centroids = dense_proj.T.numpy()  # [hidden=3072, d_model=768]
        kmeans_down = KMeans(n_clusters=64, random_state=42, n_init=10)
        kmeans_down.fit(down_centroids)
        down_centroids = torch.from_numpy(kmeans_down.cluster_centers_).to(dense_proj.dtype)  # [64, d_model]
        down_centroids = down_centroids.T  # [d_model=768, 64]

        # Save each expert
        pager = self.blocks[layer_idx].mlp.pager
        for e in range(64):
            # Each expert uses one gate centroid and one up centroid (mod 32)
            gate_idx = e % 32
            up_idx = e % 32
            expert_tensors = {
                "gate": gate_centroids[gate_idx].unsqueeze(0).repeat(hidden, 1) + torch.randn(hidden, d_model) * 0.01,
                "up": up_centroids[up_idx].unsqueeze(0).repeat(hidden, 1) + torch.randn(hidden, d_model) * 0.01,
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

    # Convert model to BF16 for training
    model = model.to(torch.bfloat16)

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
    x = torch.randn(B, num_levels * 128, 384)
    targets = torch.randint(0, 4096, (B, num_levels * 128))

    model.eval()
    with torch.no_grad():
        logits, loss = model(x, targets=targets)

    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print("Model test passed!")