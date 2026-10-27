"""
Mixture of Experts (MoE) for psudo-Engine.

Implements Top-2 routing with capacity factor, load balancing,
and integration with weight paging for SSD-backed expert storage.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..weight_paging.ternary import TernaryLinear

if TYPE_CHECKING:
    from ..weight_paging.weight_store import WeightStore


@dataclass
class MoEConfig:
    """Configuration for MoE layer."""
    d_model: int
    n_experts: int = 64
    top_k: int = 2
    capacity_factor: float = 1.25
    expert_hidden: int | None = None
    dropout: float = 0.1
    use_ternary: bool = False
    router_aux_loss_coef: float = 0.01
    z_loss_coef: float = 0.001


class TopKRouter(nn.Module):
    """
    Top-K Router for Mixture of Experts.

    Routes tokens to top-k experts with capacity constraints
    and computes load balancing losses.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        dropout: float = 0.1,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # Router network
        self.router = nn.Linear(d_model, n_experts, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        # Initialize router weights
        nn.init.normal_(self.router.weight, std=0.01)

        # Statistics
        self.register_buffer('tokens_per_expert', torch.zeros(n_experts), persistent=False)
        self.register_buffer('router_z_loss', torch.tensor(0.0), persistent=False)
        self.register_buffer('router_aux_loss', torch.tensor(0.0), persistent=False)

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            expert_weights: [batch, seq_len, top_k] - routing weights
            expert_indices: [batch, seq_len, top_k] - expert indices
            router_logits: [batch, seq_len, n_experts] - raw router logits
        """
        batch, seq_len, _ = x.shape

        # Compute router logits
        router_logits = self.router(x)  # [B, T, E]

        # Apply top-k routing
        if self.training:
            # Add noise for exploration during training
            noise = torch.randn_like(router_logits) * 0.1
            router_logits = router_logits + noise

        # Get top-k experts
        topk_weights, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)

        # Normalize weights (softmax over top-k)
        topk_weights = F.softmax(topk_weights, dim=-1)

        # Compute auxiliary losses for load balancing
        self._compute_losses(router_logits, topk_indices)

        return topk_weights, topk_indices, router_logits

    def _compute_losses(
        self,
        router_logits: torch.Tensor,
        topk_indices: torch.Tensor
    ):
        """Compute load balancing losses."""
        batch, seq_len, n_experts = router_logits.shape

        # Compute fraction of tokens routed to each expert
        router_probs = F.softmax(router_logits, dim=-1)  # [B, T, E]
        expert_counts = router_probs.mean(dim=(0, 1))  # [E]

        # Uniform target distribution
        uniform = torch.ones_like(expert_counts) / n_experts

        # Auxiliary loss: encourage uniform expert utilization
        aux_loss = (expert_counts * uniform).sum() * n_experts
        self.router_aux_loss = aux_loss.detach()

        # Z-loss: encourage router logits to stay small
        z_loss = router_logits.logsumexp(dim=-1).mean() ** 2
        self.router_z_loss = z_loss.detach()

        # Track tokens per expert (for monitoring)
        if not self.training:
            with torch.no_grad():
                flat_indices = topk_indices.view(-1)
                counts = torch.bincount(flat_indices, minlength=n_experts).float()
                self.tokens_per_expert = counts / counts.sum()

    def get_losses(self) -> dict[str, torch.Tensor]:
        """Get router losses."""
        return {
            'aux_loss': self.router_aux_loss,
            'z_loss': self.router_z_loss,
        }

    def get_expert_utilization(self) -> torch.Tensor:
        """Get expert utilization statistics."""
        return self.tokens_per_expert


class Expert(nn.Module):
    """
    Single Expert Network (Feed-Forward).

    Each expert is a SwiGLU feed-forward network.
    """

    def __init__(
        self,
        d_model: int,
        expert_hidden: int,
        dropout: float = 0.1,
        use_ternary: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.d_model = d_model
        self.expert_hidden = expert_hidden
        self.use_ternary = use_ternary

        if use_ternary:
            self.w1 = TernaryLinear(d_model, expert_hidden, bias=False, device=device, dtype=dtype)
            self.w2 = TernaryLinear(expert_hidden, d_model, bias=False, device=device, dtype=dtype)
            self.w3 = TernaryLinear(d_model, expert_hidden, bias=False, device=device, dtype=dtype)
        else:
            self.w1 = nn.Linear(d_model, expert_hidden, bias=False, device=device, dtype=dtype)
            self.w2 = nn.Linear(expert_hidden, d_model, bias=False, device=device, dtype=dtype)
            self.w3 = nn.Linear(d_model, expert_hidden, bias=False, device=device, dtype=dtype)

        self.dropout = nn.Dropout(dropout)
        self.inference_mode = False

    def prepare_for_inference(self):
        """Prepare for inference."""
        self.inference_mode = True
        if self.use_ternary:
            self.w1.prepare_for_inference()
            self.w2.prepare_for_inference()
            self.w3.prepare_for_inference()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward pass."""
        w1_out = self.w1(x)
        w3_out = self.w3(x)
        hidden = F.silu(w1_out) * w3_out
        output = self.w2(hidden)
        output = self.dropout(output)
        return output


class MoELayer(nn.Module):
    """
    Mixture of Experts Layer.

    Combines Top-K router with expert networks.
    Supports expert paging for large-scale MoE.
    """

    def __init__(
        self,
        config: MoEConfig,
        layer_idx: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.capacity_factor = config.capacity_factor
        self.expert_hidden = config.expert_hidden or (4 * config.d_model)

        # Router
        self.router = TopKRouter(
            config.d_model,
            config.n_experts,
            config.top_k,
            config.capacity_factor,
            config.dropout,
            device=device,
            dtype=dtype
        )

        # Experts (resident in memory)
        # For paged experts, we'll load them on demand
        self.experts = nn.ModuleList([
            Expert(
                config.d_model,
                self.expert_hidden,
                config.dropout,
                config.use_ternary,
                device=device,
                dtype=dtype
            )
            for _ in range(config.n_experts)
        ])

        self.inference_mode = False

    def prepare_for_inference(self):
        """Prepare for inference."""
        self.inference_mode = True
        self.router.dropout.p = 0.0
        for expert in self.experts:
            expert.prepare_for_inference()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE layer.

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
            aux_loss: load balancing loss
        """
        batch, seq_len, _ = x.shape

        # Reshape for routing
        x_flat = x.view(-1, self.d_model)  # [B*T, D]

        # Route tokens
        expert_weights, expert_indices, router_logits = self.router(x_flat)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process each expert
        for expert_idx in range(self.n_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)  # [B*T]

            if not expert_mask.any():
                continue

            # Get tokens for this expert
            expert_input = x_flat[expert_mask]  # [num_tokens, D]

            # Get weights for this expert
            expert_weight_mask = (expert_indices == expert_idx)  # [num_tokens, top_k]
            expert_weights_for_expert = expert_weights[expert_weight_mask]  # [num_tokens]

            # Expert forward pass
            expert_output = self.experts[expert_idx](expert_input)  # [num_tokens, D]

            # Weight and accumulate
            weighted_output = expert_output * expert_weights_for_expert.unsqueeze(-1)
            output[expert_mask] += weighted_output

        # Reshape back
        output = output.view(batch, seq_len, self.d_model)

        return output

    def get_router_losses(self) -> dict[str, torch.Tensor]:
        """Get router auxiliary losses."""
        return self.router.get_losses()

    def get_expert_utilization(self) -> torch.Tensor:
        """Get expert utilization statistics."""
        return self.router.get_expert_utilization()


class PagedMoELayer(MoELayer):
    """
    MoE Layer with paged expert loading from disk.

    Only keeps a subset of experts in VRAM, loads others on demand.
    """

    def __init__(
        self,
        config: MoEConfig,
        layer_idx: int,
        pager: "ExpertPager",
        resident_slots: int = 32,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        # Create minimal config for parent
        moe_config = MoEConfig(
            d_model=config.d_model,
            n_experts=config.n_experts,
            top_k=config.top_k,
            capacity_factor=config.capacity_factor,
            expert_hidden=config.expert_hidden,
            dropout=config.dropout,
            use_ternary=config.use_ternary,
        )

        # Initialize without creating all experts
        nn.Module.__init__(self)

        self.config = moe_config
        self.layer_idx = layer_idx
        self.pager = pager
        self.resident_slots = resident_slots
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        # Router
        self.router = TopKRouter(
            config.d_model,
            config.n_experts,
            config.top_k,
            config.capacity_factor,
            config.dropout,
            device=self.device,
            dtype=dtype
        )

        # Resident experts cache (LRU)
        self._resident_experts: dict[int, Expert] = {}
        self._resident_order: list[int] = []  # LRU order

        self.inference_mode = False

    def _ensure_expert_loaded(self, expert_idx: int) -> Expert:
        """Ensure expert is loaded in resident cache."""
        if expert_idx in self._resident_experts:
            # Move to front (most recently used)
            self._resident_order.remove(expert_idx)
            self._resident_order.insert(0, expert_idx)
            return self._resident_experts[expert_idx]

        # Load expert from pager
        expert_tensors = self.pager.load_expert(self.layer_idx, expert_idx)

        # Create expert and load weights
        expert = Expert(
            self.config.d_model,
            self.config.expert_hidden or (4 * self.config.d_model),
            self.config.dropout,
            self.config.use_ternary,
            device=self.device,
            dtype=self.dtype
        )

        # Load weights
        with torch.no_grad():
            expert.w1.weight.copy_(expert_tensors['w1'])
            expert.w2.weight.copy_(expert_tensors['w2'])
            expert.w3.weight.copy_(expert_tensors['w3'])

        if self.config.use_ternary:
            expert.prepare_for_inference()

        # Add to resident cache
        self._resident_experts[expert_idx] = expert
        self._resident_order.insert(0, expert_idx)

        # Evict LRU if over capacity
        if len(self._resident_experts) > self.resident_slots:
            lru_idx = self._resident_order.pop()
            del self._resident_experts[lru_idx]

        return expert

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with paged experts."""
        batch, seq_len, _ = x.shape
        x_flat = x.view(-1, self.d_model)

        # Route tokens
        expert_weights, expert_indices, router_logits = self.router(x_flat)

        output = torch.zeros_like(x_flat)

        # Get unique experts needed for this batch
        needed_experts = expert_indices.unique().tolist()

        # Load needed experts
        loaded_experts = {}
        for exp_idx in needed_experts:
            loaded_experts[exp_idx] = self._ensure_expert_loaded(exp_idx)

        # Process each expert
        for expert_idx, expert in loaded_experts.items():
            expert_mask = (expert_indices == expert_idx).any(dim=-1)

            if not expert_mask.any():
                continue

            expert_input = x_flat[expert_mask]
            expert_weight_mask = (expert_indices == expert_idx)
            expert_weights_for_expert = expert_weights[expert_weight_mask]

            expert_output = expert(expert_input)
            weighted_output = expert_output * expert_weights_for_expert.unsqueeze(-1)
            output[expert_mask] += weighted_output

        output = output.view(batch, seq_len, self.d_model)
        return output


class ExpertPager:
    """
    Manages expert weights on disk with VRAM paging.

    Integrates with WeightStore for disk-backed expert storage.
    """

    def __init__(
        self,
        weight_store: "WeightStore",
        n_layers: int,
        n_experts: int,
        expert_hidden: int,
        d_model: int,
        resident_slots: int = 32,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        self.weight_store = weight_store
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.expert_hidden = expert_hidden
        self.d_model = d_model
        self.resident_slots = resident_slots
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        # Resident expert cache: (layer_idx, expert_idx) -> expert tensors
        self._resident: dict[tuple, dict[str, torch.Tensor]] = {}
        self._resident_order: list[tuple] = []

    def load_expert(
        self,
        layer_idx: int,
        expert_idx: int
    ) -> dict[str, torch.Tensor]:
        """Load expert weights from disk."""
        key = (layer_idx, expert_idx)

        if key in self._resident:
            # Move to front
            self._resident_order.remove(key)
            self._resident_order.insert(0, key)
            return self._resident[key]

        # Load from weight store
        expert_tensors = self.weight_store.load_layer(
            layer_idx * self.n_experts + expert_idx
        )

        # Convert to device
        for k, v in expert_tensors.items():
            expert_tensors[k] = v.to(self.device, dtype=self.dtype)

        # Add to resident cache
        self._resident[key] = expert_tensors
        self._resident_order.insert(0, key)

        # Evict LRU if needed
        if len(self._resident) > self.resident_slots:
            lru_key = self._resident_order.pop()
            del self._resident[lru_key]

        return expert_tensors

    def save_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        tensors: dict[str, torch.Tensor]
    ):
        """Save expert weights to disk."""
        key = (layer_idx, expert_idx)

        # Save to weight store
        self.weight_store.save_layer(
            layer_idx * self.n_experts + expert_idx,
            tensors
        )

        # Update resident cache
        self._resident[key] = {k: v.cpu() for k, v in tensors.items()}
        if key in self._resident_order:
            self._resident_order.remove(key)
        self._resident_order.insert(0, key)

    def evict_layer(self, layer_idx: int):
        """Evict all experts for a layer."""
        keys_to_remove = [k for k in self._resident if k[0] == layer_idx]
        for key in keys_to_remove:
            del self._resident[key]
            if key in self._resident_order:
                self._resident_order.remove(key)

    def get_stats(self) -> dict:
        """Get pager statistics."""
        return {
            'resident_experts': len(self._resident),
            'resident_slots': self.resident_slots,
            'total_experts': self.n_layers * self.n_experts,
        }


def create_moe_layer(config: dict, device: torch.device | None = None) -> MoELayer:
    """Create MoELayer from config."""
    moe_config = MoEConfig(
        d_model=config.get('d_model', 512),
        n_experts=config.get('n_experts', 64),
        top_k=config.get('top_k', 2),
        capacity_factor=config.get('capacity_factor', 1.25),
        expert_hidden=config.get('expert_hidden'),
        dropout=config.get('dropout', 0.1),
        use_ternary=config.get('use_ternary', False),
        router_aux_loss_coef=config.get('router_aux_loss_coef', 0.01),
        z_loss_coef=config.get('z_loss_coef', 0.001),
    )
    layer_idx = config.get('layer_idx', 0)
    return MoELayer(moe_config, layer_idx, device=device, dtype=config.get('dtype', torch.bfloat16))


if __name__ == "__main__":
    # Quick test

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16

    config = MoEConfig(
        d_model=512,
        n_experts=8,
        top_k=2,
        expert_hidden=2048,
        use_ternary=True
    )

    moe = MoELayer(config, layer_idx=0, device=device, dtype=dtype)
    moe.prepare_for_inference()

    x = torch.randn(2, 32, 512, device=device, dtype=dtype)

    # Training mode
    moe.train()
    out = moe(x)

    # Inference mode
    moe.eval()
    out = moe(x)

    # Test expert utilization
    util = moe.get_expert_utilization()

