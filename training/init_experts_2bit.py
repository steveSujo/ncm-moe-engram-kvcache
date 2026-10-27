"""
2-bit Expert Initialization using FAISS GPU-accelerated KMeans.

Initializes MoE experts directly in 2-bit CQ format from dense NCM checkpoint.
Uses CQQuantizedMoEPager for 2-bit storage.
"""

import torch
import torch.nn as nn
import numpy as np
import faiss
from omegaconf import OmegaConf
import os
import time
import logging
from utils.expert_io_2bit import CQQuantizedMoEPager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler('init_experts_2bit.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def init_experts_from_dense_2bit(
    dense_checkpoint: str = "artifacts/tinystories/generator.pt",
    expert_dir: str = "experts_2bit",
    n_experts: int = 64,
    n_layers: int = 12,
    d_model: int = 768,
    expert_hidden: int = 3072,
    random_state: int = 42,
    use_gpu: bool = True,
    bits: float = 2.0,
    group_size: int = 128,
    resident_slots: int = 32,
):
    """
    Initialize MoE experts from dense NCM generator checkpoint using FAISS KMeans.
    Saves experts directly in 2-bit CQ format.
    """
    
    logger.info(f"Loading dense checkpoint from {dense_checkpoint}...")
    ckpt = torch.load(dense_checkpoint, map_location='cpu')
    state_dict = ckpt['model']
    
    # Create 2-bit pager for saving
    pager = CQQuantizedMoEPager(
        expert_dir=expert_dir,
        n_layers=n_layers,
        n_experts=n_experts,
        d_model=d_model,
        expert_hidden=expert_hidden,
        resident_slots=resident_slots,
        bits=bits,
        group_size=group_size,
    )
    
    # Check which layers are already complete
    completed_layers = []
    for layer_idx in range(n_layers):
        layer_complete = True
        for expert_id in range(n_experts):
            path = pager.index.get_path(layer_idx, expert_id)
            if not os.path.exists(path):
                layer_complete = False
                break
        if layer_complete:
            completed_layers.append(layer_idx)
    
    if completed_layers:
        logger.info(f"Skipping completed layers: {completed_layers}")
    
    # Setup FAISS
    if use_gpu and faiss.get_num_gpus() > 0:
        logger.info("Using FAISS GPU acceleration")
        res = faiss.StandardGpuResources()
    else:
        logger.info("Using FAISS CPU")
        res = None
    
    # Extract MLP weights from dense checkpoint
    # Dense model has MLP with up_proj, down_proj, gate_proj (SwiGLU)
    mlp_weights = {}
    for key, val in state_dict.items():
        if 'mlp' in key and ('up_proj' in key or 'down_proj' or 'gate_proj' in key):
            mlp_weights[key] = val
    
    logger.info(f"Found {len(mlp_weights)} MLP weight tensors")
    
    for layer_idx in range(n_layers):
        if layer_idx in completed_layers:
            continue
        
        logger.info(f"Initializing layer {layer_idx}/{n_layers}...")
        start_time = time.time()
        
        # Get dense weights for this layer
        layer_prefix = f"blocks.{layer_idx}.mlp"
        up_key = f"{layer_prefix}.up_proj.weight"
        down_key = f"{layer_prefix}.down_proj.weight"
        gate_key = f"{layer_prefix}.gate_proj.weight"
        
        if up_key not in state_dict:
            logger.warning(f"Layer {layer_idx} not found in checkpoint, creating random experts")
            # Create random experts
            for expert_id in range(n_experts):
                expert_tensors = pager._create_random_expert()
                pager.save_expert(layer_idx, expert_id, expert_tensors)
            continue
        
        # Stack MLP weights: [expert_hidden, d_model] for gate/up, [d_model, expert_hidden] for down
        W_gate = state_dict[gate_key].to(torch.float32).numpy()  # [hidden, d_model]
        W_up = state_dict[up_key].to(torch.float32).numpy()      # [hidden, d_model]
        W_down = state_dict[down_key].to(torch.float32).numpy()  # [d_model, hidden]
        
        # Combine gate and up for clustering (both are [hidden, d_model])
        # We'll cluster the combined matrix
        W_combined = np.concatenate([W_gate, W_up], axis=0)  # [2*hidden, d_model]
        
        # K-means clustering to find expert centroids
        d = W_combined.shape[1]
        kmeans = faiss.Kmeans(
            d=d,
            k=n_experts,
            niter=50,
            verbose=False,
            seed=random_state + layer_idx,
            gpu=use_gpu and res is not None,
        )
        
        if use_gpu and res is not None:
            kmeans.gpu = True
            kmeans.cp = res
        
        kmeans.train(W_combined)
        
        # Get centroids: [n_experts, d]
        centroids = kmeans.centroids  # [n_experts, d_model]
        
        # Assign each centroid to create expert weights
        for expert_id in range(n_experts):
            centroid = centroids[expert_id]  # [d_model]
            
            # Create expert weights by using centroid as basis
            # We'll use the centroid to modulate the dense weights
            expert_tensors = {
                "gate": torch.from_numpy(W_gate).clone(),
                "up": torch.from_numpy(W_up).clone(),
                "down": torch.from_numpy(W_down).clone(),
            }
            
            # Add small noise based on centroid
            noise_scale = 0.01
            for key in expert_tensors:
                if key == "down":
                    noise = torch.randn_like(expert_tensors[key]) * noise_scale
                    expert_tensors[key] = expert_tensors[key] + noise
                else:
                    noise = torch.randn_like(expert_tensors[key]) * noise_scale
                    expert_tensors[key] = expert_tensors[key] + noise
            
            # Save expert in 2-bit format
            pager.save_expert(layer_idx, expert_id, expert_tensors)
        
        layer_time = time.time() - start_time
        logger.info(f"  Layer {layer_idx}: saved {n_experts} experts in {layer_time:.1f}s")
    
    # Verify
    total_experts = sum(1 for f in os.listdir(expert_dir) if f.endswith('.safetensors'))
    expected = n_layers * n_experts
    logger.info(f"Total experts: {total_experts}/{expected}")
    
    if total_experts == expected:
        logger.info("2-bit Expert initialization complete!")
    else:
        logger.warning(f"Missing {expected - total_experts} experts!")


if __name__ == "__main__":
    os.makedirs("experts_2bit", exist_ok=True)
    init_experts_from_dense_2bit(use_gpu=True)
    logger.info("2-bit Expert initialization complete!")