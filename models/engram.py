"""
Engram Key-Value Memory Module (Needle 2 / DeepSeek style)

Hashed n-gram lookup tables (2-gram + 3-gram) with context-aware gating.
Inserted at specific transformer layers to provide fast associative memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


def fnv1a_hash(data: torch.Tensor, num_buckets: int, seed: int = 0) -> torch.Tensor:
    """
    FNV-1a hash for deterministic multi-head hashing.
    data: [B, T, num_heads] or [B, T] - token IDs
    Returns: [B, T, num_heads] indices in [0, num_buckets)
    """
    if data.dim() == 2:
        data = data.unsqueeze(-1)  # [B, T, 1]
    
    B, T, H = data.shape
    device = data.device
    # Use int64 for compatibility
    hash_vals = torch.zeros((B, T, H), dtype=torch.int64, device=device)
    
    fnv_prime = 16777619
    offset_basis = 2166136261
    
    # Process each head separately
    for h in range(H):
        x = data[:, :, h].to(torch.int64)
        h_val = torch.full((B, T), offset_basis + seed * 12345 + h * 67890, dtype=torch.int64, device=device)
        
        # FNV-1a: process 4 bytes (32-bit)
        for i in range(4):
            byte = (x >> (i * 8)) & 0xFF
            h_val = h_val ^ byte
            h_val = (h_val * 16777619) & 0xFFFFFFFF
        
        # Modulo to get bucket index
        hash_vals[:, :, h] = h_val % num_buckets
    
    return hash_vals.to(torch.long)
def engram_indices(
    tokens: torch.Tensor,
    orders: List[int],
    num_heads: int,
    num_slots: int,
    canonicalize: bool = True
) -> torch.Tensor:
    """
    Generate multi-head n-gram hash indices for Engram lookup.
    
    Args:
        tokens: [B, T] token IDs (int)
        orders: list of n-gram orders, e.g., [2, 3]
        num_heads: number of hash heads per order
        num_slots: hash table size (slots)
        canonicalize: if True, apply simple canonicalization
    
    Returns:
        indices: [B, T, num_tables] where num_tables = len(orders) * num_heads
    """
    B, T = tokens.shape
    device = tokens.device
    all_indices = []
    
    for order in orders:
        if order == 1:
            # Unigram: just the token itself
            ngram_tokens = tokens.unsqueeze(-1).expand(-1, -1, num_heads)
        else:
            # n-gram: combine consecutive tokens
            ngram_tokens_list = []
            for t in range(T):
                # Gather n-gram tokens ending at t
                start = max(0, t - order + 1)
                end = t + 1
                if end - start < order:
                    # Pad with 0 for beginning of sequence
                    pad_len = order - (end - start)
                    ngram = F.pad(tokens[:, start:end], (pad_len, 0), value=0)
                else:
                    ngram = tokens[:, start:end]
                
                # Combine via XOR (order-independent for unigrams, order-dependent via position)
                combined = ngram[:, 0]
                for i in range(1, order):
                    combined = combined ^ (ngram[:, i] << (i * 8))  # simple positional mixing
                ngram_tokens_list.append(combined)
            
            ngram_tokens = torch.stack(ngram_tokens_list, dim=1)  # [B, T]
            ngram_tokens = ngram_tokens.unsqueeze(-1).expand(-1, -1, num_heads)
        
        # Hash per head
        for h in range(num_heads):
            idx = fnv1a_hash(ngram_tokens[:, :, h], num_slots, seed=h)
            all_indices.append(idx.squeeze(-1))  # Remove extra head dim
    
    # Stack: [B, T, num_tables]
    return torch.stack(all_indices, dim=-1)


def engram_geometry(config: dict) -> Tuple[List[int], int, int]:
    """
    Compute Engram geometry from config.
    Returns: (orders, num_heads, sub_dim)
    """
    orders = config.get("orders", [2, 3])
    num_slots = config.get("slots", 8192)
    sub_dim = config.get("sub_dim", 128)
    num_heads = config.get("num_heads", 4)
    return orders, num_heads, sub_dim


class EngramModule(nn.Module):
    """
    Engram Key-Value Memory Module.
    
    At each Engram site:
    1. Token sequence → multi-head n-gram hash indices
    2. Lookup embedding tables (one per hash head)
    3. Concat embeddings → project to K, V
    4. Gated attention: gate = sigmoid(RMSNorm(x)^T RMSNorm(k) / sqrt(d))
    5. Output = gate * v
    6. Depthwise causal Conv1D + residual
    7. Add to residual stream
    """
    
    def __init__(
        self,
        d_model: int,
        orders: List[int],
        num_heads: int,
        sub_dim: int,
        num_slots: int,
        conv_taps: int = 4,
        conv_dilation: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.orders = orders
        self.num_heads = num_heads
        self.sub_dim = sub_dim
        self.num_slots = num_slots
        self.num_tables = len(orders) * num_heads
        
        # Embedding tables: [num_tables, num_slots, sub_dim]
        self.embeddings = nn.Parameter(
            torch.randn(self.num_tables, num_slots, sub_dim) * 0.02
        )
        
        # Key/Value projections
        total_emb_dim = self.num_tables * sub_dim
        self.k_proj = nn.Linear(total_emb_dim, d_model, bias=False)
        self.v_proj = nn.Linear(total_emb_dim, d_model, bias=False)
        
        # RMSNorm for gating
        self.rmsnorm_x = nn.RMSNorm(d_model)
        self.rmsnorm_k = nn.RMSNorm(d_model)
        
        # Depthwise causal Conv1D
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_taps,
            dilation=conv_dilation,
            groups=d_model,  # depthwise
            padding=0,  # causal handled manually
            bias=False
        )
        self.conv_act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize conv weights
        nn.init.normal_(self.conv.weight, std=0.02)
    
    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: [B, T] token IDs
        Returns:
            k, v: [B, T, d_model] - Engram key/value for fusion
        """
        B, T = tokens.shape
        device = tokens.device
        
        # Generate hash indices: [B, T, num_tables]
        indices = engram_indices(tokens, self.orders, self.num_heads, self.num_slots)
        
        # Lookup embeddings: [B, T, num_tables, sub_dim] -> [B, T, total_emb_dim]
        embs = []
        for i in range(self.num_tables):
            table = self.embeddings[i]  # [num_slots, sub_dim]
            idx = indices[:, :, i]  # [B, T]
            # Handle out-of-bounds (shouldn't happen with modulo)
            idx = idx.clamp(0, self.num_slots - 1)
            emb = F.embedding(idx, table)  # [B, T, sub_dim]
            embs.append(emb)
        
        concat_emb = torch.cat(embs, dim=-1)  # [B, T, total_emb_dim]
        
        # Project to K, V
        k = self.k_proj(concat_emb)  # [B, T, d_model]
        v = self.v_proj(concat_emb)  # [B, T, d_model]
        
        return k, v
    
    def apply_engam(
        self,
        x: torch.Tensor,  # [B, T, d_model] residual stream
        tokens: torch.Tensor,  # [B, T] token IDs
    ) -> torch.Tensor:
        """
        Apply Engram memory to residual stream.
        
        Returns:
            x + EngramOutput  # [B, T, d_model]
        """
        B, T, D = x.shape
        
        # Get Engram K, V
        k, v = self.forward(tokens)  # [B, T, d_model]
        
        # Gating: sigmoid(RMSNorm(x)^T RMSNorm(k) / sqrt(d))
        x_norm = self.rmsnorm_x(x)
        k_norm = self.rmsnorm_k(k)
        gate = torch.sigmoid(
            torch.sum(x_norm * k_norm, dim=-1, keepdim=True) / (D ** 0.5)
        )  # [B, T, 1]
        
        # Gated value
        gated_v = gate * v  # [B, T, d_model]
        
        # Causal depthwise conv
        # Conv1d expects [B, C, L], we have [B, T, D] -> [B, D, T]
        conv_in = gated_v.transpose(1, 2)  # [B, d_model, T]
        
        # Causal padding: pad left by (kernel_size - 1) * dilation
        pad_left = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0]
        conv_in = F.pad(conv_in, (pad_left, 0))
        
        conv_out = self.conv(conv_in)  # [B, d_model, T]
        conv_out = self.conv_act(conv_out)
        conv_out = conv_out.transpose(1, 2)  # [B, T, d_model]
        conv_out = self.dropout(conv_out)
        
        # Residual add
        return x + conv_out


# Convenience function for Engram config from Needle 2
def needle2_engram_config() -> dict:
    """Default Needle 2 Engram config from config.json"""
    return {
        "orders": [2, 3],
        "slots": 8192,
        "sub_dim": 128,
        "conv_taps": 4,
        "conv_dilation": 3,
        "num_heads": 4,
    }


if __name__ == "__main__":
    # Quick test
    import torch
    
    config = needle2_engram_config()
    orders, heads, sub_dim = engram_geometry(config)
    print(f"Orders: {orders}, Heads: {heads}, Sub-dim: {sub_dim}")
    print(f"Num tables: {len(orders) * heads}")
    
    # Test indices
    tokens = torch.randint(0, 8192, (2, 128))
    idx = engram_indices(tokens, orders, heads, config["slots"])
    print(f"Indices shape: {idx.shape}")  # [B, T, num_tables]
    
    # Test module
    engram = EngramModule(768, orders, heads, sub_dim, config["slots"], 
                          config["conv_taps"], config["conv_dilation"])
    k, v = engram(tokens)
    print(f"K, V shape: {k.shape}, {v.shape}")  # [B, T, 768]
    
    # Test application
    x = torch.randn(2, 128, 768)
    out = engram.apply_engam(x, tokens)
    print(f"Output shape: {out.shape}")  # [B, T, 768]
    print("Engram module test passed!")