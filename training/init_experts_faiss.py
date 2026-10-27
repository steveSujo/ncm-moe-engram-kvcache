"""
Expert initialization using FAISS GPU-accelerated KMeans.
Much faster than sklearn KMeans on CPU.
"""

import faiss
import logging
import numpy as np
import os
import time
import torch
from utils.expert_io import MoEPager


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler('expert_init_faiss.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def init_experts_from_dense_faiss(
    dense_checkpoint: str = "artifacts/tinystories/generator.pt",
    expert_dir: str = "experts",
    n_experts: int = 64,
    n_layers: int = 12,
    d_model: int = 768,
    expert_hidden: int = 3072,
    random_state: int = 42,
    use_gpu: bool = True,
):
    """
    Initialize MoE experts from dense NCM generator checkpoint using FAISS KMeans.
    """
    
    logger.info(f"Loading dense checkpoint from {dense_checkpoint}...")
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

    # Check which layers are already complete
    completed_layers = []
    for layer_idx in range(n_layers):
        layer_complete = True
        for e in range(n_experts):
            path = pager.index.get_path(layer_idx, e)
            if not os.path.exists(path):
                layer_complete = False
                break
        if layer_complete:
            completed_layers.append(layer_idx)
    
    if completed_layers:
        logger.info(f"Skipping completed layers: {completed_layers}")
    
    for layer_idx in range(n_layers):
        if layer_idx in completed_layers:
            logger.info(f"Skipping layer {layer_idx} (already complete)")
            continue
            
        layer_start = time.time()
        logger.info(f"Processing layer {layer_idx}...")
        
        # Extract dense MLP weights
        c_fc = state_dict[f'transformer.blocks.{layer_idx}.mlp.c_fc.weight']  # [3072, 768]
        c_fc_bias = state_dict.get(f'transformer.blocks.{layer_idx}.mlp.c_fc.bias')
        c_proj = state_dict[f'transformer.blocks.{layer_idx}.mlp.c_proj.weight']  # [768, 3072]
        c_proj_bias = state_dict.get(f'transformer.blocks.{layer_idx}.mlp.c_proj.bias')
        
        logger.info(f"  c_fc shape: {c_fc.shape}, c_proj shape: {c_proj.shape}")
        
        # Convert to numpy for k-means
        fc_weight = c_fc.numpy().astype(np.float32)  # [hidden, d_model]
        proj_weight = c_proj.numpy().astype(np.float32)  # [d_model, hidden]
        
        # Concatenate fc and proj^T for clustering
        proj_weight_t = proj_weight.T  # [hidden, d_model]
        logger.info(f"  fc_weight shape: {fc_weight.shape}, proj_weight_t shape: {proj_weight_t.shape}")
        
        combined = np.concatenate([fc_weight, proj_weight_t], axis=1)  # [hidden, 2*d_model]
        logger.info(f"  Combined shape: {combined.shape}")
        
        logger.info(f"  Clustering {combined.shape[0]} hidden units into {n_experts} experts with FAISS...")
        cluster_start = time.time()
        
        # FAISS KMeans
        dim = combined.shape[1]
        kmeans = faiss.Kmeans(dim, n_experts, niter=20, nredo=5, verbose=True, seed=random_state, gpu=use_gpu)
        kmeans.train(combined)
        cluster_labels = kmeans.index.search(combined, 1)[1].flatten()
        centroids = kmeans.centroids  # [n_experts, 2*d_model]
        
        logger.info(f"  FAISS KMeans completed in {time.time() - cluster_start:.1f}s, Centroids shape: {centroids.shape}")
        
        # Split centroids back
        fc_centroids = centroids[:, :d_model]  # [n_experts, d_model]
        proj_centroids = centroids[:, d_model:].T  # [d_model, n_experts]
        logger.info(f"  fc_centroids shape: {fc_centroids.shape}, proj_centroids shape: {proj_centroids.shape}")
        
        # For each expert, create full SwiGLU weights
        for e in range(n_experts):
            # Base weights from centroids
            gate_base = fc_centroids[e:e+1]  # [1, d_model]
            up_base = fc_centroids[e:e+1]    # [1, d_model] (same init)
            down_base = proj_centroids[:, e:e+1]  # [d_model, 1]
            
            # Expand to expert_hidden with noise
            gate_weight = gate_base.repeat(expert_hidden, axis=0) + np.random.randn(expert_hidden, d_model).astype(np.float32) * 0.01
            up_weight = up_base.repeat(expert_hidden, axis=0) + np.random.randn(expert_hidden, d_model).astype(np.float32) * 0.01
            down_weight = down_base.repeat(expert_hidden, axis=1) + np.random.randn(d_model, expert_hidden).astype(np.float32) * 0.01
            
            # Biases
            gate_bias = np.zeros(expert_hidden, dtype=np.float32)
            up_bias = np.zeros(expert_hidden, dtype=np.float32)
            down_bias = np.zeros(d_model, dtype=np.float32)
            
            # Convert to bfloat16 tensors
            gate_t = torch.from_numpy(gate_weight).to(torch.bfloat16)
            up_t = torch.from_numpy(up_weight).to(torch.bfloat16)
            down_t = torch.from_numpy(down_weight).to(torch.bfloat16)
            gate_bias_t = torch.from_numpy(gate_bias).to(torch.bfloat16)
            up_bias_t = torch.from_numpy(up_bias).to(torch.bfloat16)
            down_bias_t = torch.from_numpy(down_bias).to(torch.bfloat16)
            
            # Save via pager (will write to SSD and update index)
            tensors = {
                'gate_weight': gate_t,
                'up_weight': up_t,
                'down_weight': down_t,
                'gate_bias': gate_bias_t,
                'up_bias': up_bias_t,
                'down_bias': down_bias_t,
            }
            pager.save_expert(layer_idx=layer_idx, expert_id=e, tensors=tensors)
            
            if (e + 1) % 16 == 0:
                logger.info(f"  Layer {layer_idx}: saved {e+1}/{n_experts} experts")
        
        layer_time = time.time() - layer_start
        logger.info(f"  Layer {layer_idx}: saved {n_experts} experts in {layer_time:.1f}s")
    
    # Verify
    total_experts = sum(1 for f in os.listdir(expert_dir) if f.endswith('.safetensors'))
    expected = n_layers * n_experts
    logger.info(f"Total experts: {total_experts}/{expected}")
    
    if total_experts == expected:
        logger.info("Expert initialization complete!")
    else:
        logger.warning(f"Missing {expected - total_experts} experts!")


if __name__ == "__main__":
    os.makedirs("experts", exist_ok=True)
    init_experts_from_dense_faiss(use_gpu=True)
    logger.info("Expert initialization complete!")