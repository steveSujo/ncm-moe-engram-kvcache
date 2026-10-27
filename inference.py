#!/usr/bin/env python
"""
Inference script for MoE + Engram + KV Cache NCM Generator.
Loads trained model with paged experts and runs generation.
"""

import os
import sys
import torch
import argparse
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenization.vqvae_tokenizer import VQVAETokenizer
from models.moe_engram_generator import MoEEngramNextConceptTransformer
from models.moe_block import MoEBlock
from utils.expert_io import MoEPager
def load_vqvae(checkpoint_dir: str = "artifacts/tinystories"):
    """Load frozen VQ-VAE tokenizer."""
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    cfg = OmegaConf.load(config_path)
    tokenizer = VQVAETokenizer.from_config(cfg.tokenizer, base_dir=checkpoint_dir)
    tokenizer.vqvae.eval()
    for p in tokenizer.vqvae.parameters():
        p.requires_grad = False
    return tokenizer, cfg.model


def load_model(
    model_config,
    vqvae_local,
    expert_dir: str = "experts",
    resident_slots: int = 32,
    device: str = "cuda",
    checkpoint: str = None,
):
    """Load integrated model with paged experts."""
    
    # Create pager for expert loading
    pager = MoEPager(
        expert_dir=expert_dir,
        resident_slots=resident_slots,
        n_layers=model_config.n_layer,
        n_experts=64,
        d_model=model_config.n_embd,
        expert_hidden=3072,
        device=torch.device(device),
    )
    
    # Build model with shared pager
    model = MoEEngramNextConceptTransformer(
        vqvae_local=vqvae_local,
        config=model_config,
        expert_dir=expert_dir,
        resident_slots=resident_slots,
        pager=pager,
        device=torch.device(device),
    )
    
    # Load checkpoint if provided
    if checkpoint and os.path.exists(checkpoint):
        print(f"Loading checkpoint from {checkpoint}...")
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict, strict=False)
        print(f"  Step: {state_dict.get('step', 'unknown')}")
    
    model.to(device)
    model.eval()
    
    return model, pager

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_levels: int = 11,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    cfg_scale: float = 0.0,
    device: str = "cuda",
):
    """Generate text from prompt."""
    
    # Encode prompt
    if hasattr(tokenizer, 'encode'):
        prompt_tokens = tokenizer.encode(prompt)
    else:
        # Fallback: use VQ-VAE tokenizer
        prompt_tokens = tokenizer.vqvae.get_ncm_inputs(prompt)
    
    # Convert to tensor
    if isinstance(prompt_tokens, list):
        prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0)
    
    prompt_tokens = prompt_tokens.to(device)
    
    print(f"Prompt tokens shape: {prompt_tokens.shape}")
    
    # Generate using model's generate method
    generated = model.generate(
        prompt_tokens,
        l0=None,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        cfg_scale=cfg_scale,
        max_levels=max_levels,
    )
    
    print(f"Generated shape: {generated.shape}")
    
    # Decode - we get level 0 tokens (unigram)
    # For full decoding, would need VQ-VAE decoder
    return generated


def main():
    parser = argparse.ArgumentParser(description="MoE+Engram NCM Inference")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt text")
    parser.add_argument("--expert-dir", type=str, default="experts", help="Expert weights directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--resident-slots", type=int, default=32, help="Resident expert slots")
    parser.add_argument("--max-levels", type=int, default=11, help="Max levels to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--cfg-scale", type=float, default=0.0, help="CFG scale")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/tinystories", help="Artifact directory")
    args = parser.parse_args()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    print(f"Device: {args.device}")
    print(f"Resident slots: {args.resident_slots}")
    print(f"Expert dir: {args.expert_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer, model_config = load_vqvae(args.artifact_dir)
    print(f"  Model config: {model_config.n_layer} layers, {model_config.n_embd} dim, {model_config.n_head} heads")
    
    # Load model
    print("\nLoading model...")
    model, pager = load_model(
        model_config=model_config,
        vqvae_local=tokenizer.vqvae,
        expert_dir=args.expert_dir,
        resident_slots=args.resident_slots,
        device=args.device,
        checkpoint=args.checkpoint,
    )
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    print(f"Pager: {pager}")
    
    # Generate
    print(f"\nGenerating from prompt: '{args.prompt}'")
    generated = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_levels=args.max_levels,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        cfg_scale=args.cfg_scale,
        device=args.device,
    )
    
    print(f"\nGeneration complete!")
    print(f"Output shape: {generated.shape}")
    print(f"First 20 tokens: {generated[0, :20].tolist()}")
    
    # Log paging stats
    if hasattr(model, 'kv_cache'):
        print(f"KV Cache: {model.kv_cache}")


if __name__ == "__main__":
    main()