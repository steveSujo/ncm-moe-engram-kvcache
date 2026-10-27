"""
Engram Memory Module for psudo-Engine.

Implements hashed n-gram lookup tables (Needle 2 / DeepSeek style)
with 2-gram and 3-gram orders for fast associative memory.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from ..weight_paging.ternary import TernaryLinear


def fnv1a_hash(
    data: torch.Tensor,
    num_buckets: int,
    seed: int = 0
) -> torch.Tensor:
    """
    FNV-1a hash function for n-gram hashing.

    Args:
        data: Input tensor [..., ngram_size] with integer token IDs
        num_buckets: Number of hash buckets
        seed: Hash seed for different heads

    Returns:
        Hash values modulo num_buckets
    """
    if data.dim() < 2:
        data = data.unsqueeze(0)

    # FNV-1a parameters
    fnv_prime = 16777619
    offset_basis = 2166136261 + seed

    hash_vals = torch.full(data.shape[:-1], offset_basis, dtype=torch.int64, device=data.device)

    for i in range(data.shape[-1]):
        hash_vals = (hash_vals ^ data[..., i].to(torch.int64)) * fnv_prime

    return (hash_vals % num_buckets).to(torch.long)


def compute_ngram_indices(
    tokens: torch.Tensor,
    orders: list[int],
    num_buckets: int,
    num_heads: int
) -> torch.Tensor:
    """
    Compute hash indices for n-grams of different orders.

    Args:
        tokens: [batch, seq_len] token IDs
        orders: List of n-gram orders (e.g., [2, 3])
        num_buckets: Number of hash buckets per head
        num_heads: Number of hash heads

    Returns:
        indices: [batch, seq_len, num_heads, num_orders] hash indices
    """
    batch, seq_len = tokens.shape
    device = tokens.device

    all_indices = []

    for order in orders:
        if order > seq_len:
            continue

        # Create n-grams
        ngrams = []
        for i in range(seq_len - order + 1):
            ngrams.append(tokens[:, i:i+order])
        ngrams = torch.stack(ngrams, dim=1)  # [B, seq_len-order+1, order]

        # Pad to full sequence length
        padding = torch.zeros(batch, order - 1, order, dtype=tokens.dtype, device=device)
        ngrams = torch.cat([padding, ngrams], dim=1)  # [B, seq_len, order]

        # Hash with multiple heads
        head_indices = []
        for h in range(num_heads):
            indices = fnv1a_hash(ngrams, num_buckets, seed=h)
            head_indices.append(indices)

        head_indices = torch.stack(head_indices, dim=2)  # [B, seq_len, num_heads]
        all_indices.append(head_indices)

    if all_indices:
        return torch.stack(all_indices, dim=-1)  # [B, seq_len, num_heads, num_orders]
    else:
        return torch.zeros(batch, seq_len, num_heads, len(orders), dtype=torch.long, device=device)


class EngramModule(nn.Module):
    """
    Engram Key-Value Memory Module (Needle 2 style).

    Hashed n-gram lookup tables with context-aware gating.
    Provides fast associative memory at specific transformer layers.
    """

    def __init__(
        self,
        d_model: int,
        num_slots: int = 8192,
        num_heads: int = 4,
        orders: list[int] = None,
        num_buckets: int = 8192,
        sub_dim: int | None = None,
        gating: bool = True,
        use_ternary: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        if orders is None:
            orders = [2, 3]
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.orders = orders
        self.num_buckets = num_buckets
        self.sub_dim = sub_dim or (d_model // num_heads)
        self.gating = gating
        self.use_ternary = use_ternary

        # Key and Value tables for each head and order
        # [num_heads, num_orders, num_buckets, sub_dim]
        if use_ternary:
            self.keys = nn.ParameterList([
                nn.ParameterList([
                    TernaryLinear(
                        self.sub_dim, self.sub_dim, bias=False, device=device, dtype=dtype
                    )
                    for _ in range(len(orders))
                ])
                for _ in range(num_heads)
            ])
            self.values = nn.ParameterList([
                nn.ParameterList([
                    TernaryLinear(
                        self.sub_dim, self.sub_dim, bias=False, device=device, dtype=dtype
                    )
                    for _ in range(len(orders))
                ])
                for _ in range(num_heads)
            ])
        else:
            self.keys = nn.ParameterList([
                nn.ParameterList([
                    nn.Parameter(
                        torch.randn(num_buckets, self.sub_dim, device=device, dtype=dtype) * 0.02
                    )
                    for _ in range(len(orders))
                ])
                for _ in range(num_heads)
            ])
            self.values = nn.ParameterList([
                nn.ParameterList([
                    nn.Parameter(
                        torch.randn(num_buckets, self.sub_dim, device=device, dtype=dtype) * 0.02
                    )
                    for _ in range(len(orders))
                ])
                for _ in range(num_heads)
            ])

        # Output projection
        if use_ternary:
            self.output_proj = TernaryLinear(
                self.sub_dim * num_heads * len(orders), d_model, bias=False, device=device, dtype=dtype
            )
        else:
            self.output_proj = nn.Linear(
                self.sub_dim * num_heads * len(orders), d_model, bias=False, device=device, dtype=dtype
            )

        # Gating network
        if gating:
            if use_ternary:
                self.gate_proj = TernaryLinear(
                    d_model, num_heads * len(orders), bias=False, device=device, dtype=dtype
                )
            else:
                self.gate_proj = nn.Linear(
                    d_model, num_heads * len(orders), bias=False, device=device, dtype=dtype
                )

        # Layer norm for output
        self.ln = nn.LayerNorm(d_model, eps=1e-6, device=device, dtype=dtype)
        self.dropout = nn.Dropout(0.1)

        self.inference_mode = False

    def prepare_for_inference(self):
        """Prepare for inference."""
        self.inference_mode = True
        if self.use_ternary:
            for h in range(self.num_heads):
                for o in range(len(self.orders)):
                    self.keys[h][o].prepare_for_inference()
                    self.values[h][o].prepare_for_inference()
            self.output_proj.prepare_for_inference()
            if self.gating:
                self.gate_proj.prepare_for_inference()

    def forward(
        self,
        x: torch.Tensor,
        tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through Engram memory.

        Args:
            x: [batch, seq_len, d_model] - hidden states
            tokens: [batch, seq_len] - token IDs for n-gram hashing

        Returns:
            output: [batch, seq_len, d_model] - engram-enhanced hidden states
        """
        batch, seq_len, _ = x.shape

        # Compute n-gram hash indices
        # [B, T, num_heads, num_orders]
        indices = compute_ngram_indices(tokens, self.orders, self.num_buckets, self.num_heads)

        # Process each head
        head_outputs = []

        for h in range(self.num_heads):
            head_out = []

            for o, _order in enumerate(self.orders):
                # Get hash indices for this head and order
                idx = indices[..., h, o]  # [B, T]

                # Lookup keys and values
                if self.use_ternary:
                    # Keys/values are TernaryLinear modules
                    # For inference, use pre-quantized; for training, use STE
                    key_table = self.keys[h][o]
                    value_table = self.values[h][o]

                    # Retrieve keys and values
                    # This is a simplified version - in practice would use embedding lookup
                    k = key_table.weight  # Would need proper indexing
                    v = value_table.weight
                else:
                    key_table = self.keys[h][o]  # [num_buckets, sub_dim]
                    value_table = self.values[h][o]

                    # Embedding lookup
                    k = F.embedding(idx, key_table)  # [B, T, sub_dim]
                    v = F.embedding(idx, value_table)  # [B, T, sub_dim]

                # Attention: query from x, key/value from engram
                # Project x to sub_dim for this head
                x_head = x.view(batch * seq_len, self.d_model)
                x_head = x_head[:, h * self.sub_dim:(h + 1) * self.sub_dim]
                x_head = x_head.view(batch, seq_len, self.sub_dim)

                # Compute attention scores
                scores = torch.matmul(x_head, k.transpose(-2, -1)) / (self.sub_dim ** 0.5)
                attn = F.softmax(scores, dim=-1)

                # Weighted sum of values
                out = torch.matmul(attn, v)  # [B, T, sub_dim]
                head_out.append(out)

            # Concatenate orders for this head
            if head_out:
                head_out = torch.cat(head_out, dim=-1)  # [B, T, num_orders * sub_dim]
                head_outputs.append(head_out)

        if head_outputs:
            # Concatenate heads
            combined = torch.cat(head_outputs, dim=-1)  # [B, T, num_heads * num_orders * sub_dim]

            # Apply gating before output projection
            if self.gating:
                gate_logits = self.gate_proj(x)  # [B, T, num_heads * num_orders]
                gate = torch.sigmoid(gate_logits)

                # Expand gate to match combined
                # gate: [B, T, num_heads * num_orders]
                # combined: [B, T, num_heads * num_orders * sub_dim]
                gate = gate.unsqueeze(-1).repeat(1, 1, 1, self.sub_dim)
                gate = gate.view(batch, seq_len, -1)

                combined = combined * gate

            # Project to d_model
            engram_out = self.output_proj(combined)
        else:
            engram_out = torch.zeros_like(x)

        # Residual connection with layer norm
        out = self.ln(x + self.dropout(engram_out))

        return out

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_slots={self.num_slots}, num_heads={self.num_heads}, orders={self.orders}"


class EngramMemory(nn.Module):
    """
    Multi-layer Engram Memory.

    Manages EngramModules at specific layers of the transformer.
    """

    def __init__(
        self,
        config: dict,
        engram_layers: list[int],
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.engram_layers = engram_layers
        self.config = config

        self.engrams = nn.ModuleDict()
        for layer_idx in engram_layers:
            self.engrams[str(layer_idx)] = EngramModule(
                d_model=config.get('d_model', 512),
                num_slots=config.get('engram_slots', 8192),
                num_heads=config.get('engram_heads', 4),
                orders=config.get('engram_orders', [2, 3]),
                num_buckets=config.get('engram_buckets', 8192),
                gating=config.get('engram_gating', True),
                use_ternary=config.get('use_ternary', False),
                device=device,
                dtype=dtype
            )

    def prepare_for_inference(self):
        """Prepare all engram modules for inference."""
        for engram in self.engrams.values():
            engram.prepare_for_inference()

    def forward(
        self,
        x: torch.Tensor,
        tokens: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """Apply engram at specific layer if configured."""
        if layer_idx in self.engram_layers:
            return self.engrams[str(layer_idx)](x, tokens)
        return x

    def get_engram_layers(self) -> list[int]:
        """Get list of layers with engram."""
        return self.engram_layers


def create_engram_module(config: dict, device: torch.device | None = None) -> EngramModule:
    """Create EngramModule from config."""
    return EngramModule(
        d_model=config.get('d_model', 512),
        num_slots=config.get('engram_slots', 8192),
        num_heads=config.get('engram_heads', 4),
        orders=config.get('engram_orders', [2, 3]),
        num_buckets=config.get('engram_buckets', 8192),
        gating=config.get('engram_gating', True),
        use_ternary=config.get('use_ternary', False),
        device=device,
        dtype=config.get('dtype', torch.bfloat16)
    )


if __name__ == "__main__":
    # Quick test

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16

    config = {
        'd_model': 512,
        'engram_slots': 8192,
        'engram_heads': 4,
        'engram_orders': [2, 3],
        'engram_buckets': 8192,
        'engram_gating': True,
        'use_ternary': True,
        'dtype': dtype,
    }

    engram = create_engram_module(config, device=device)
    engram.prepare_for_inference()

    x = torch.randn(2, 128, 512, device=device, dtype=dtype)
    tokens = torch.randint(0, 8192, (2, 128), device=device)

    out = engram(x, tokens)

    # Test multi-layer
    engram_memory = EngramMemory(config, engram_layers=[4, 12, 20], device=device, dtype=dtype)
    engram_memory.prepare_for_inference()

    for layer in [3, 4, 5, 12]:
        out = engram_memory(x, tokens, layer)

