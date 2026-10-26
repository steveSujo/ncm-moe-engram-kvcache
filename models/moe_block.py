"""
MoE Block (replaces MLP in transformer)

Top-2 routing with capacity factor, entropy loss for load balancing.
Integrates with MoEPager for SSD expert loading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
from utils.expert_io import MoEPager, swiglu_forward


class Top2Router(nn.Module):
    """
    Top-2 router with entropy regularization.
    
    Computes routing weights and selects top-2 experts per token.
    """
    
    def __init__(self, d_model: int, n_experts: int, capacity_factor: float = 1.25):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        
        # Router: linear projection to expert logits
        self.router = nn.Linear(d_model, n_experts, bias=False)
        
        # Initialize to small values for balanced start
        nn.init.normal_(self.router.weight, std=0.01)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_model] or [B*T, d_model]
        Returns:
            expert_weights: [B, T, top_k] - softmax weights for selected experts
            expert_indices: [B, T, top_k] - selected expert indices
            router_logits: [B, T, n_experts] - all logits (for entropy loss)
        """
        # Flatten batch and sequence
        orig_shape = x.shape[:-1]
        x_flat = x.view(-1, x.shape[-1])  # [B*T, d_model]
        
        # Router logits
        logits = self.router(x_flat)  # [B*T, n_experts]
        
        # Top-2 selection
        top_k = 2
        top_logits, top_indices = torch.topk(logits, top_k, dim=-1)  # [B*T, 2]
        top_weights = F.softmax(top_logits, dim=-1)  # [B*T, 2]
        
        # Reshape back
        expert_weights = top_weights.view(*orig_shape, top_k)
        expert_indices = top_indices.view(*orig_shape, top_k)
        router_logits = logits.view(*orig_shape, self.n_experts)
        
        return expert_weights, expert_indices, router_logits
    
    def entropy_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy loss for load balancing.
        Encourages uniform expert utilization.
        
        Args:
            router_logits: [B, T, n_experts] or [B*T, n_experts]
        Returns:
            entropy_loss: scalar
        """
        # Average probability distribution over all tokens
        probs = F.softmax(router_logits, dim=-1)  # [B*T, n_experts]
        mean_probs = probs.mean(dim=0)  # [n_experts]
        
        # Entropy of mean distribution
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
        max_entropy = torch.log(torch.tensor(self.n_experts, dtype=torch.float, device=router_logits.device))
        
        # Normalized entropy (0 = collapsed, 1 = uniform)
        normalized_entropy = entropy / max_entropy
        
        # Loss: encourage high entropy
        return 1.0 - normalized_entropy


class MoEBlock(nn.Module):
    """
    Mixture of Experts block replacing standard MLP.
    
    Each token routes to top-2 experts. Experts loaded on-demand from SSD.
    """
    
    def __init__(
        self,
        config,
        layer_idx: int,
        pager: MoEPager,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.pager = pager
        self.capacity_factor = capacity_factor
        
        d_model = config.n_embd
        n_experts = 64  # fixed for now
        
        # Router (always resident)
        self.router = Top2Router(d_model, n_experts, capacity_factor)
        
        # Expert norm (per-expert or shared - using shared for simplicity)
        self.norm = nn.RMSNorm(d_model)
        
        # Store router loss for training
        self.router_loss = 0.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            out: [B, T, d_model]
        """
        B, T, D = x.shape
        device = x.device
        
        # Pre-norm
        x_norm = self.norm(x)  # [B, T, d_model]
        
        # Route tokens
        expert_weights, expert_indices, router_logits = self.router(x_norm)
        
        # Compute router entropy loss
        self.router_loss = self.router.entropy_loss(router_logits)
        
        # Get unique experts needed for this layer
        unique_experts = expert_indices.unique().tolist()
        
        # Ensure experts are resident
        expert_tensors = self.pager.ensure_resident(self.layer_idx, unique_experts)
        
        # Flatten for processing
        x_flat = x_norm.view(-1, D)  # [B*T, d_model]
        expert_weights_flat = expert_weights.view(-1, 2)  # [B*T, 2]
        expert_indices_flat = expert_indices.view(-1, 2)  # [B*T, 2]
        
        # Initialize output
        out_flat = torch.zeros_like(x_flat)
        
        # Process each expert
        for i in range(2):
            expert_id = expert_indices_flat[:, i]  # [B*T]
            weight = expert_weights_flat[:, i]     # [B*T]
            
            # Group tokens by expert
            for eid in unique_experts:
                mask = (expert_id == eid)
                if not mask.any():
                    continue
                
                # Get tokens for this expert
                expert_x = x_flat[mask]           # [N_e, d_model]
                expert_w = weight[mask].unsqueeze(-1)  # [N_e, 1]
                
                # SwiGLU forward
                expert_out = swiglu_forward(expert_x, expert_tensors[eid])
                
                # Weighted add
                out_flat[mask] += expert_w * expert_out
        
        # Reshape
        out = out_flat.view(B, T, D)
        
        return out
    
    def extra_repr(self):
        return f"layer={self.layer_idx}, experts={self.pager.n_experts}"


class MoEBlockNoPager(nn.Module):
    """
    MoE Block with all experts in memory (for testing/small models).
    """
    
    def __init__(
        self,
        config,
        layer_idx: int,
        n_experts: int = 64,
        capacity_factor: float = 1.25,
        expert_hidden: int = 3072,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        
        d_model = config.n_embd
        
        # Router
        self.router = Top2Router(d_model, n_experts, capacity_factor)
        
        # All experts in memory
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_hidden, bias=False),
                nn.SiLU(),
                nn.Linear(expert_hidden, d_model, bias=False),
            ) for _ in range(n_experts)
        ])
        
        # Initialize experts
        for expert in self.experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
        
        self.norm = nn.RMSNorm(d_model)
        self.router_loss = 0.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        x_norm = self.norm(x)
        expert_weights, expert_indices, router_logits = self.router(x_norm)
        self.router_loss = self.router.entropy_loss(router_logits)
        
        x_flat = x_norm.view(-1, D)
        expert_weights_flat = expert_weights.view(-1, 2)
        expert_indices_flat = expert_indices.view(-1, 2)
        
        out_flat = torch.zeros_like(x_flat)
        
        for i in range(2):
            expert_id = expert_indices_flat[:, i]
            weight = expert_weights_flat[:, i].unsqueeze(-1)
            
            for eid in range(self.n_experts):
                mask = (expert_id == eid)
                if not mask.any():
                    continue
                
                expert_x = x_flat[mask]
                expert_out = self.experts[eid](expert_x)
                out_flat[mask] += weight[mask] * expert_out
        
        return out_flat.view(B, T, D)


if __name__ == "__main__":
    # Quick test with NoPager version
    import torch
    from omegaconf import OmegaConf
    
    cfg = OmegaConf.create({
        'n_embd': 768,
        'n_layer': 12,
    })
    
    moe = MoEBlockNoPager(cfg, layer_idx=0, n_experts=8, expert_hidden=3072)
    x = torch.randn(2, 128, 768)
    out = moe(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"Router loss: {moe.router_loss:.4f}")
    print("MoEBlock test passed!")