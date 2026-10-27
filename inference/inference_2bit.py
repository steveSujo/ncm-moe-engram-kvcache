"""
2-bit Inference for NCM-MoE-Engram-KVCache
===========================================

Native 2-bit inference using pre-quantized weights.
"""

import torch
import torch.nn as nn
import argparse
import os
import sys
from typing import Optional, List, Dict
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.moe_engram_generator_2bit import (
    MoEEngramNextConceptTransformer2Bit,
    create_moe_engram_model_2bit
)
from tokenization.vqvae_tokenizer import VQVAETokenizer


class InferenceEngine2Bit:
    """2-bit inference engine for NCM model."""
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        expert_dir: str = "experts",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        
        # Load tokenizer
        self.tokenizer = VQVAETokenizer.from_pretrained("hazyresearch/ncm-checkpoints")
        
        # Create model
        self.model = create_moe_engram_model_2bit(config_path, expert_dir)
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        
        # Move to device and prepare for inference
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.prepare_for_inference()
        self.model.eval()
        
        print(f"Model loaded on {device} with dtype {dtype}")
        print(f"Vocab size: {self.tokenizer.vocab_size}")
        print(f"Levels: {self.tokenizer.num_levels}")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
        level: Optional[torch.Tensor] = None
    ) -> str:
        """Generate text from prompt."""
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        # Default level if not provided
        if level is None:
            level = torch.zeros_like(input_ids)
        
        # Generate
        output_ids = self.model.generate(
            input_ids=input_ids,
            level=level,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        )
        
        # Decode
        generated = self.tokenizer.decode(output_ids[0].tolist())
        return generated
    
    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None
    ) -> List[str]:
        """Generate text for multiple prompts."""
        # Encode all prompts
        all_tokens = [self.tokenizer.encode(p) for p in prompts]
        max_len = max(len(t) for t in all_tokens)
        
        # Pad
        input_ids = []
        for tokens in all_tokens:
            padded = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))
            input_ids.append(padded)
        
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        level = torch.zeros_like(input_ids)
        
        # Generate
        output_ids = self.model.generate(
            input_ids=input_ids,
            level=level,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        )
        
        # Decode
        results = []
        for i in range(output_ids.shape[0]):
            generated = self.tokenizer.decode(output_ids[i].tolist())
            results.append(generated)
        
        return results
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage in GB."""
        if self.device.type == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated(self.device) / 1e9,
                'reserved': torch.cuda.memory_reserved(self.device) / 1e9,
                'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1e9
            }
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}


def main():
    parser = argparse.ArgumentParser(description="2-bit NCM Inference")
    parser.add_argument("--config", type=str, default="configs/ncm_2bit.yaml", help="Model config path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--experts", type=str, default="experts", help="Expert directory")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch", action="store_true", help="Batch generation mode")
    
    args = parser.parse_args()
    
    # Create engine
    engine = InferenceEngine2Bit(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        expert_dir=args.experts,
        device=args.device
    )
    
    # Print memory usage
    mem = engine.get_memory_usage()
    print(f"GPU Memory: {mem['allocated']:.2f} GB allocated, {mem['reserved']:.2f} GB reserved")
    
    if args.batch:
        # Interactive batch mode
        prompts = []
        print("Enter prompts (empty line to generate):")
        while True:
            prompt = input("> ")
            if not prompt:
                break
            prompts.append(prompt)
        
        if prompts:
            results = engine.generate_batch(
                prompts,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            for i, (prompt, result) in enumerate(zip(prompts, results)):
                print(f"\n--- Prompt {i+1} ---")
                print(f"Input: {prompt}")
                print(f"Output: {result}")
    else:
        # Single prompt
        result = engine.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print(f"\nInput: {args.prompt}")
        print(f"Output: {result}")
    
    # Final memory
    mem = engine.get_memory_usage()
    print(f"\nPeak GPU Memory: {mem['max_allocated']:.2f} GB")


if __name__ == "__main__":
    main()