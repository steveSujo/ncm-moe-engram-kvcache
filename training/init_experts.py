"""
Expert Initialization from Dense Checkpoint (k-means distillation)

Loads dense NCM generator checkpoint, extracts MLP weights per layer,
clusters into 64 experts using k-means, saves to SSD.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from omegaconf import OmegaConf
import os
from utils.expert_io import MoEPager


def init_experts_from_dense(
    dense_checkpoint: str = "artifacts/tinystories/generator.pt",
    expert_dir: str = "experts",
    n_experts: int = 64,
    n_layers: int = 12,
    d_model: int = 768,
    expert_hidden: int = 3072,
    random_state: int = 42,
):
    """
    Initialize MoE experts from dense NCM generator checkpoint.
    """
    
    print(f"Loading dense checkpoint from {dense_checkpoint}...")
    ckpt = torch.load(dense_checkpoint, map_location='cpu')
    state_dict = ckpt['model']
    
    # Create pager for saving
    pager = MoEPager(
        expert_dir=expert_dir,
        resident_slots=0,
        n_layers=n_layers,
        n_experts=n_experts,
        d_model=d_model,
        expert_hidden=expert_hidden,
    )
    
    for layer_idx in range(n_layers):
        print(f"Processing layer {layer_idx}...")
        
        # Extract dense MLP weights
        c_fc = state_dict[f'transformer.blocks.{layer_idx}.mlp.c_fc.weight']  # [3072, 768]
        c_fc_bias = state_dict.get(f'transformer.blocks.{layer_idx}.mlp.c_fc.bias')
        c_proj = state_dict[f'transformer.blocks.{layer_idx}.mlp.c_proj.weight']  # [768, 3072]
        c_proj_bias = state_dict.get(f'transformer.blocks.{layer_idx}.mlp.c_proj.bias')
        
        print(f"  c_fc shape: {c_fc.shape}, c_proj shape: {c_proj.shape}")
        
        # Convert to numpy for k-means
        fc_weight = c_fc.numpy().astype(np.float32)  # [hidden, d_model]
        proj_weight = c_proj.numpy().astype(np.float32)  # [d_model, hidden]
        
        # Concatenate fc and proj^T for clustering
        # fc: [hidden, d_model], proj^T: [hidden, d_model]
        proj_weight_t = proj_weight.T  # [hidden, d_model]
        print(f"  fc_weight shape: {fc_weight.shape}, proj_weight_t shape: {proj_weight_t.shape}")
        
        combined = np.concatenate([fc_weight, proj_weight_t], axis=1)  # [hidden, 2*d_model]
        print(f"  Combined shape: {combined.shape}")
        
        print(f"  Clustering {combined.shape[0]} hidden units into {n_experts} experts...")
        kmeans = KMeans(n_clusters=n_experts, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(combined)
        centroids = kmeans.cluster_centers_  # [n_experts, 2*d_model]
        print(f"  Centroids shape: {centroids.shape}")
        
        # Split centroids back
        fc_centroids = centroids[:, :d_model]  # [n_experts, d_model]
        proj_centroids = centroids[:, d_model:].T  # [d_model, n_experts]
        print(f"  fc_centroids shape: {fc_centroids.shape}, proj_centroids shape: {proj_centroids.shape}")
        
        # For each expert, create full SwiGLU weights
        for e in range(n_experts):
            # Base weights from centroids
            gate_base = fc_centroids[e:e+1]  # [1, d_model]
            up_base = fc_centroids[e:e+1]    # [1, d_model] (same init)
            down_base = proj_centroids[:, e:e+1]  # [d_model, 1]
            
            print(f"  Expert {e}: gate_base shape: {gate_base.shape}")
            
            # Expand to expert_hidden with noise
            gate_weight = gate_base.repeat(expert_hidden, axis=0) + np.random.randn(expert_hidden, d_model).astype(np.float32) * 0.01
            up_weight = up_base.repeat(expert_hidden, axis=0) + np.random.randn(expert_hidden, d_model).astype(np.float32) * 0.01
            down_weight = down_base.repeat(expert_hidden, axis=1) + np.random.randn(d_model, expert_hidden).astype(np.float32) * 0.01
            
            print(f"  Expert {e}: gate_weight shape: {gate_weight.shape}")
            
            # Convert to torch
            expert_tensors = {
                "gate": torch.from_numpy(gate_weight).to(torch.bfloat16),
                "up": torch.from_numpy(up_weight).to(torch.bfloat16),
                "down": torch.from_numpy(down_weight).to(torch.bfloat16),
            }
            
            # Add biases if present
            if c_fc_bias is not None:
                bias_base = c_fc_bias.numpy()[cluster_labels == e]
                if len(bias_base) > 0:
                    gate_bias = torch.from_numpy(np.mean(bias_base) * np.ones(expert_hidden, dtype=np.float32)).to(torch.bfloat16)
                    up_bias = torch.from_numpy(np.mean(bias_base) * np.ones(expert_hidden, dtype=np.float32)).to(torch.bfloat16)
                else:
                    gate_bias = torch.zeros(expert_hidden, dtype=torch.bfloat16)
                    up_bias = torch.zeros(expert_hidden, dtype=torch.bfloat16)
                expert_tensors["gate_bias"] = gate_bias
                expert_tensors["up_bias"] = up_bias
            
            if c_proj_bias is not None:
                down_bias = torch.from_numpy(c_proj_bias.numpy().astype(np.float32)).to(torch.bfloat16)
                expert_tensors["down_bias"] = down_bias
            
            pager.save_expert(layer_idx, e, expert_tensors)
        
        print(f"  Layer {layer_idx}: saved {n_experts} experts")
    
    print(f"\nAll experts saved to {expert_dir}")
    print(f"Index: {os.path.join(expert_dir, 'expert_index.json')}")
    
    # Verify
    total_size = 0
    for layer_idx in range(n_layers):
        for e in range(n_experts):
            path = pager.index.get_path(layer_idx, e)
            total_size += os.path.getsize(path)
    print(f"Total expert storage: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    os.makedirs("experts", exist_ok=True)
    init_experts_from_dense()
    print("Expert initialization complete!")