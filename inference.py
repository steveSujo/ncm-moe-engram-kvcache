#!/usr/bin/env python
"""
Inference script for MoE + Engram + KV Cache NCM Generator.
Loads trained model with paged experts and runs generation.
"""

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenization.vqvae_tokenizer import VQVAETokenizer
from models.moe_engram_generator import MoEEngramNextConceptTransformer
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
    
    # Encode prompt to BPE tokens
    if hasattr(tokenizer, 'encode'):
        prompt_bpe = tokenizer.encode(prompt)
    else:
        raise ValueError("Tokenizer must have encode method")
    
    # Convert to tensor
    if isinstance(prompt_bpe, list):
        prompt_bpe = torch.tensor(prompt_bpe, dtype=torch.long).unsqueeze(0)
    elif prompt_bpe.dim() == 1:
        prompt_bpe = prompt_bpe.unsqueeze(0)
    
    prompt_bpe = prompt_bpe.to(device)
    
    print(f"Prompt BPE shape: {prompt_bpe.shape}")
    # The model expects level 0 tokens as input
    # Use VQ-VAE to get the level 0 codes for the prompt
    with torch.no_grad():
        # For generation, we need to encode the prompt through VQ-VAE to get level codes
        # But the model's generate() expects level 0 tokens directly
        # So we'll use the unigram sampling for level 0 and then generate levels 1+
        pass
    
    # Generate multi-level codes using model
    # Model generates level 0 + max_levels-1 more levels
    generated_codes = model.generate(
        prompt_bpe[:, :1],  # Use first token as seed for level 0
        l0=None,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        cfg_scale=cfg_scale,
        max_levels=max_levels,
    )
    
    print(f"Generated codes shape: {generated_codes.shape}")
    print(f"Level 0 token IDs: {generated_codes[0].cpu().tolist()}")
    
    # Now decode using VQ-VAE decoder
    # generated_codes is [B, total_tokens] where each level has 1 token (autoregressive)
    # We need to split by level and decode
    # Each level generates 1 token per step, so we have max_levels tokens total
    # Reshape: [B, max_levels] -> split per level
    level_codes = []
    for i in range(max_levels):
        level_codes.append(generated_codes[:, i:i+1])
    
    # Decode using VQ-VAE decoder
    with torch.no_grad():
        bpe_logits = tokenizer.vqvae.decode_codes(level_codes)  # [B, T, vocab]
        bpe_tokens = bpe_logits.argmax(dim=-1)  # [B, T]
        text = tokenizer.decode(bpe_tokens[0])
    
    print(f"Decoded text: {text}")
    
    return generated_codes, text

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
    # Generate
    print(f"\nGenerating from prompt: '{args.prompt}'")
    generated_codes, generated_text = generate(
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
    print(f"Output shape: {generated_codes.shape}")
    print(f"First 20 tokens: {generated_codes[0, :20].tolist()}")
    print(f"Generated text: {generated_text}")
    
    # Log paging stats
    if hasattr(model, 'kv_cache'):
        print(f"KV Cache: {model.kv_cache}")


if __name__ == "__main__":
    main()