"""
Psudo Transformer - Full Model Assembly for psudo-Engine.

Integrates all components: embeddings, transformer blocks with MoE,
Engram memory, bounded KV cache, and output heads.
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from .blocks import (
    RMSNorm,
    TransformerBlock,
    precompute_rope_cos_sin,
)
from .engram import EngramMemory
from .kv_cache import create_multi_layer_kv_cache


class PsudoEmbeddings(nn.Module):
    """Token and position embeddings."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.pos_emb = nn.Embedding(max_seq_len, d_model, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        token_emb = self.token_emb(input_ids)  # [B, T, D]

        # Position embeddings
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        pos_emb = self.pos_emb(pos_ids)  # [B, T, D]

        # Combine
        x = token_emb + pos_emb
        x = self.dropout(x)

        return x


class PsudoOutputHead(nn.Module):
    """Output head with tied embeddings."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        use_ternary: bool = False,
        tie_embeddings: bool = True,
        token_emb: nn.Embedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tie_embeddings = tie_embeddings

        if tie_embeddings and token_emb is not None:
            self.output_emb = token_emb
            self.proj = None
        else:
            self.output_emb = None
            if use_ternary:
                from ..weight_paging.ternary import TernaryLinear
                self.proj = TernaryLinear(d_model, vocab_size, bias=False, device=device, dtype=dtype)
            else:
                self.proj = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.tie_embeddings and self.output_emb is not None:
            return F.linear(x, self.output_emb.weight)
        else:
            return self.proj(x)


class PsudoTransformer(nn.Module):
    """
    Psudo Transformer - Complete model for psudo code generation.

    Architecture:
    - Token + Position embeddings
    - N transformer blocks with:
      - Multi-head attention with RoPE
      - MoE feed-forward (paged experts)
      - Engram memory at specified layers
    - Bounded KV cache for autoregressive generation
    - Output head (tied to embeddings)
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        # Model dimensions
        self.vocab_size = config.model.vocab_size
        self.d_model = config.model.d_model
        self.n_layers = config.model.n_layers
        self.n_heads = config.model.n_heads
        self.head_dim = config.model.head_dim
        self.d_ff = config.model.d_ff
        self.max_seq_len = config.model.get('max_seq_len', 2048)
        self.dropout = config.model.get('dropout', 0.1)
        self.use_ternary = config.model.get('weight_bits', 1.58) == 1.58

        # MoE config
        self.n_experts = config.model.n_experts
        self.top_k = config.model.top_k
        self.resident_slots = config.model.resident_slots

        # Engram config
        self.engram_layers = config.model.engram_layers
        self.engram_slots = config.model.engram_slots
        self.engram_heads = config.model.engram_heads
        self.engram_orders = config.model.engram_orders

        # KV cache config
        self.max_window = config.model.max_window
        self.sink_tokens = config.model.sink_tokens

        # Device and dtype
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.dtype = getattr(torch, config.model.get('dtype', 'bfloat16'))

        # Embeddings
        self.embeddings = PsudoEmbeddings(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            device=self.device,
            dtype=self.dtype
        )


        # Precompute RoPE
        cos, sin = precompute_rope_cos_sin(
            self.max_seq_len,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        )
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            has_engram = layer_idx in self.engram_layers

            block = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                head_dim=self.head_dim,
                d_ff=self.d_ff,
                dropout=self.dropout,
                use_ternary=self.use_ternary,
                has_engram=has_engram,
                engram_dim=self.d_model,
                device=self.device,
                dtype=self.dtype
            )
            self.blocks.append(block)

        # Engram memory
        if self.engram_layers:
            self.engram_memory = EngramMemory(
                config={
                    'd_model': self.d_model,
                    'engram_slots': self.engram_slots,
                    'engram_heads': self.engram_heads,
                    'engram_orders': self.engram_orders,
                    'engram_gating': True,
                    'use_ternary': self.use_ternary,
                    'dtype': str(self.dtype).split('.')[-1],
                },
                engram_layers=self.engram_layers,
                device=self.device,
                dtype=self.dtype
            )
        else:
            self.engram_memory = None

        # KV Cache
        self.kv_cache = create_multi_layer_kv_cache(
            {
                'n_layers': self.n_layers,
                'max_window': self.max_window,
                'sink_tokens': self.sink_tokens,
                'n_heads': self.n_heads,
                'head_dim': self.head_dim,
                'dtype': str(self.dtype).split('.')[-1],
                'quantize_kv': True,
            },
            device=self.device
        )

        # Output norm
        self.final_norm = RMSNorm(self.d_model, device=self.device, dtype=self.dtype)

        # Output head
        self.output_head = PsudoOutputHead(
            d_model=self.d_model,
            vocab_size=self.vocab_size,
            use_ternary=self.use_ternary,
            tie_embeddings=config.model.get('tie_embeddings', True),
            token_emb=self.embeddings.token_emb if config.model.get('tie_embeddings', True) else None,
            device=self.device,
            dtype=self.dtype
        )

        # Training state
        self.training_step = 0
        self.inference_mode = False

    def prepare_for_inference(self):
        """Switch all components to inference mode."""
        self.inference_mode = True
        self.eval()

        for block in self.blocks:
            block.prepare_for_inference()

        if self.engram_memory:
            self.engram_memory.prepare_for_inference()

        self.output_head.proj.prepare_for_inference() if self.output_head.proj else None

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_cache: bool = False,
        return_dict: bool = True
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len] token IDs
            targets: Optional [batch, seq_len] target token IDs
            use_cache: Whether to use KV cache
            return_dict: Whether to return dict or tuple

        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: Optional scalar loss
        """
        batch, seq_len = input_ids.shape

        # Embeddings
        x = self.embeddings(input_ids)  # [B, T, D]

        # Precompute RoPE
        cos, sin = precompute_rope_cos_sin(
            self.max_seq_len,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        )
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

        # Transformer blocks
        for layer_idx, block in enumerate(self.blocks):
            # Get mask for this layer if using cache
            mask = None
            if use_cache and self.inference_mode:
                mask = self.kv_cache.get_mask(seq_len, layer_idx, self.device)

            # Engram memory (apply per layer for layers with engram)
            if self.engram_memory is not None and layer_idx in self.engram_layers:
                x = self.engram_memory(x, input_ids, layer_idx)

            # Block forward (no engram token IDs needed since EngramMemory already applied)
            x = block(
                x,
                self.cos,
                self.sin,
                mask=mask,
                kv_cache=self.kv_cache.caches[layer_idx] if use_cache else None,
                engram=None
            )

        # Final norm
        x = self.final_norm(x)

        # Output head
        logits = self.output_head(x)  # [B, T, V]

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-1
            )

        if return_dict:
            return {'logits': logits, 'loss': loss}
        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_tokens: list[int] | None = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            input_ids: [batch, seq_len] starting tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            stop_tokens: List of token IDs to stop at
            use_cache: Whether to use KV cache

        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        self.prepare_for_inference()

        if stop_tokens is None:
            stop_tokens = []

        input_ids.shape[0]
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(
                    generated,
                    use_cache=use_cache,
                    return_dict=True
                )
                logits = outputs['logits']

                # Get logits for last token
                next_logits = logits[:, -1, :] / temperature

                # Apply top-k
                if top_k > 0:
                    topk_logits, topk_indices = torch.topk(next_logits, top_k)
                    next_logits = torch.full_like(next_logits, float('-inf'))
                    next_logits.scatter_(1, topk_indices, topk_logits)

                # Apply top-p (nucleus)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumsum > top_p
                    mask[:, 1:] = mask[:, :-1].clone()
                    mask[:, 0] = False
                    sorted_logits[mask] = float('-inf')
                    next_logits.scatter_(1, sorted_indices, sorted_logits)

                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append
                generated = torch.cat([generated, next_token], dim=1)

                # Check stop tokens
                if stop_tokens and next_token.item() in stop_tokens:
                    break

                # Check max length
                if generated.shape[1] >= self.max_seq_len:
                    break

        return generated

    def reset_kv_cache(self):
        """Reset KV cache for new sequence."""
        self.kv_cache.reset()

    def get_memory_usage(self) -> dict:
        """Get model memory usage."""
        return {
            'kv_cache': self.kv_cache.get_total_memory_usage(),
            'model_params': sum(p.numel() for p in self.parameters()),
            'model_params_mb': sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 * 1024),
        }

    def save_pretrained(self, path: str):
        """Save model weights."""
        torch.save(self.state_dict(), path)

    @classmethod
    def from_config(cls, config_path: str) -> 'PsudoTransformer':
        """Load model from config file."""
        config = OmegaConf.load(config_path)
        return cls(config)

    @classmethod
    def from_pretrained(cls, config_path: str, weights_path: str) -> 'PsudoTransformer':
        """Load model from config and weights."""
        model = cls.from_config(config_path)
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        return model


def create_model(config: DictConfig | dict) -> PsudoTransformer:
    """Create PsudoTransformer from config."""
    if isinstance(config, dict):
        config = OmegaConf.create(config)
    return PsudoTransformer(config)


def load_model_from_checkpoint(
    config_path: str,
    checkpoint_path: str,
    device: torch.device | None = None
) -> PsudoTransformer:
    """Load model from checkpoint."""
    config = OmegaConf.load(config_path)
    model = PsudoTransformer(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device or 'cpu'))
    return model


if __name__ == "__main__":
    # Quick test

    from omegaconf import OmegaConf

    config = OmegaConf.create({
        'model': {
            'vocab_size': 8192,
            'd_model': 512,
            'n_layers': 4,
            'n_heads': 8,
            'head_dim': 64,
            'd_ff': 2048,
            'max_seq_len': 1024,
            'dropout': 0.1,
            'weight_bits': 1.58,
            'n_experts': 8,
            'top_k': 2,
            'resident_slots': 4,
            'engram_layers': [1, 3],
            'engram_slots': 8192,
            'engram_heads': 4,
            'engram_orders': [2, 3],
            'max_window': 256,
            'sink_tokens': 1,
            'tie_embeddings': True,
        }
    })

    model = PsudoTransformer(config)

    # Test forward
    input_ids = torch.randint(0, 8192, (2, 32))
    outputs = model(input_ids)

    if outputs['loss'] is not None:
        pass

    # Test generation
    model.prepare_for_inference()
    generated = model.generate(
        torch.tensor([[1]]),  # BOS token
        max_new_tokens=10,
        temperature=0.8
    )

    # Memory usage

