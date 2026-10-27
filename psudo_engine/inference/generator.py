"""
Autoregressive Generator with Weight Paging for psudo-Engine.

Implements efficient text generation with paged weight loading,
KV caching, and streaming output.
"""

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ..model.psudo_transformer import PsudoTransformer, create_model
from ..weight_paging.dequantizer import BatchDequantizer
from ..weight_paging.page_controller import PageController, SequentialPageController
from ..weight_paging.weight_store import WeightStore
from .tokenizer import PsudoTokenizer


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    stop_tokens: list[int] = None
    repetition_penalty: float = 1.0
    use_cache: bool = True
    stream: bool = False

    def __post_init__(self):
        if self.stop_tokens is None:
            self.stop_tokens = []


class PsudoGenerator:
    """
    Autoregressive generator with weight paging support.

    Features:
    - KV cache for efficient generation
    - Weight paging for large models
    - Streaming output
    - Configurable sampling parameters
    """

    def __init__(
        self,
        model: PsudoTransformer,
        tokenizer: PsudoTokenizer,
        weight_store: WeightStore | None = None,
        page_controller: PageController | None = None,
        dequantizer: BatchDequantizer | None = None,
        device: torch.device | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.weight_store = weight_store
        self.page_controller = page_controller
        self.dequantizer = dequantizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device
        self.model.to(self.device)
        self.model.prepare_for_inference()

        # Generation stats
        self.stats = {
            'total_tokens': 0,
            'total_time': 0.0,
            'tokens_per_second': 0.0,
        }

    @classmethod
    def from_checkpoint(
        cls,
        config_path: str,
        checkpoint_path: str,
        weight_store_dir: str = "weights",
        device: torch.device | None = None,
    ) -> 'PsudoGenerator':
        """Create generator from checkpoint with weight paging."""
        # Load configs
        from omegaconf import OmegaConf
        config = OmegaConf.load(config_path)

        # Create model
        model = create_model(config)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

        # Create tokenizer
        tokenizer = PsudoTokenizer(vocab_size=config.model.vocab_size)

        # Create weight paging components
        weight_store = WeightStore(weight_store_dir)
        dequantizer = BatchDequantizer(device=device)
        page_controller = SequentialPageController(
            weight_store, dequantizer,
            max_vram_bytes=2_000_000_000,
            device=device
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            weight_store=weight_store,
            page_controller=page_controller,
            dequantizer=dequantizer,
            device=device,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any], device: torch.device | None = None) -> 'PsudoGenerator':
        """Create generator from config dict."""
        from omegaconf import OmegaConf
        model_config = OmegaConf.create(config)

        model = create_model(model_config)
        tokenizer = PsudoTokenizer(vocab_size=config.model.vocab_size)

        return cls(model=model, tokenizer=tokenizer, device=device)

    def generate(
        self,
        prompt: str | list[int],
        config: GenerationConfig | None = None,
        callback: Callable[[str], None] | None = None
    ) -> str | list[int]:
        """
        Generate text from prompt.

        Args:
            prompt: Input text or token IDs
            config: Generation configuration
            callback: Optional callback for streaming (receives decoded token)

        Returns:
            Generated text (if prompt is str) or token IDs (if prompt is list)
        """
        config = config or GenerationConfig()
        return_ids = isinstance(prompt, list)

        # Encode prompt
        if isinstance(prompt, str):
            input_ids = self.tokenizer.encode(prompt)
            input_ids = [self.tokenizer.bos_token_id] + input_ids
        else:
            input_ids = prompt

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Reset KV cache
        self.model.reset_kv_cache()

        # Generate
        generated = self._generate_loop(input_tensor, config, callback)

        # Update stats
        new_tokens = generated.shape[1] - input_tensor.shape[1]
        self.stats['total_tokens'] += new_tokens

        if return_ids:
            return generated[0].tolist()
        else:
            # Decode only new tokens
            new_token_ids = generated[0, input_tensor.shape[1]:].tolist()
            return self.tokenizer.decode(new_token_ids)

    def _generate_loop(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        callback: Callable[[str], None] | None = None
    ) -> torch.Tensor:
        """Core generation loop."""
        start_time = time.time()
        current_ids = input_ids.clone()

        with torch.no_grad():
            for _step in range(config.max_new_tokens):
                # Forward pass
                outputs = self.model.forward(
                    current_ids,
                    use_cache=config.use_cache,
                    return_dict=True
                )
                logits = outputs['logits']

                # Get next token logits
                next_logits = logits[:, -1, :] / config.temperature

                # Apply repetition penalty
                if config.repetition_penalty != 1.0:
                    for token_id in set(current_ids[0].tolist()):
                        next_logits[:, token_id] /= config.repetition_penalty

                # Apply top-k
                if config.top_k > 0:
                    topk_logits, topk_indices = torch.topk(next_logits, config.top_k)
                    next_logits = torch.full_like(next_logits, float('-inf'))
                    next_logits.scatter_(1, topk_indices, topk_logits)

                # Apply top-p (nucleus sampling)
                if config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumsum > config.top_p
                    mask[:, 1:] = mask[:, :-1].clone()
                    mask[:, 0] = False
                    sorted_logits[mask] = float('-inf')
                    next_logits.scatter_(1, sorted_indices, sorted_logits)

                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Check stop tokens
                if config.stop_tokens and next_token.item() in config.stop_tokens:
                    break

                # Check EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Check max sequence length
                if current_ids.shape[1] >= self.model.max_seq_len:
                    break

                # Append token
                current_ids = torch.cat([current_ids, next_token], dim=1)

                # Stream callback
                if callback and config.stream:
                    decoded = self.tokenizer.decode([next_token.item()])
                    callback(decoded)

            elapsed = time.time() - start_time
            self.stats['total_time'] += elapsed
            if self.stats['total_tokens'] > 0:
                self.stats['tokens_per_second'] = self.stats['total_tokens'] / self.stats['total_time']

        return current_ids

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None
    ) -> Iterator[str]:
        """
        Generate text with streaming output.

        Yields:
            Decoded tokens as they are generated
        """
        config = config or GenerationConfig()
        config.stream = True

        tokens_generated = []

        def internal_callback(token: str):
            tokens_generated.append(token)

        # Use a generator pattern
        result = self.generate(prompt, config)

        if isinstance(result, str):
            for token in result.split():
                yield token + " "
        else:
            for token_id in result:
                yield self.tokenizer.decode([token_id])

    def chat(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None
    ) -> str:
        """
        Chat-style generation with conversation history.

        Args:
            messages: List of {"role": "user|assistant", "content": "..."}
            config: Generation configuration

        Returns:
            Assistant response
        """
        # Format conversation
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n".join(prompt_parts) + "\nAssistant: "
        return self.generate(prompt, config)

    def get_stats(self) -> dict[str, Any]:
        """Get generation statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset generation statistics."""
        self.stats = {
            'total_tokens': 0,
            'total_time': 0.0,
            'tokens_per_second': 0.0,
        }


class BatchedGenerator:
    """
    Batched generation for multiple prompts.
    """

    def __init__(self, generator: PsudoGenerator):
        self.generator = generator

    def generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None
    ) -> list[str]:
        """Generate for multiple prompts."""
        results = []
        for prompt in prompts:
            result = self.generator.generate(prompt, config)
            results.append(result)
        return results

    def generate_batch_ids(
        self,
        prompts: list[list[int]],
        config: GenerationConfig | None = None
    ) -> list[list[int]]:
        """Generate for multiple token ID prompts."""
        results = []
        for prompt in prompts:
            result = self.generator.generate(prompt, config)
            results.append(result)
        return results


def create_generator(config: dict[str, Any], device: torch.device | None = None) -> PsudoGenerator:
    """Create PsudoGenerator from config."""
    return PsudoGenerator.from_config(config, device)


if __name__ == "__main__":
    # Quick test

    from omegaconf import OmegaConf

    config = OmegaConf.create({
        'model': {
            'vocab_size': 1000,
            'd_model': 256,
            'n_layers': 2,
            'n_heads': 4,
            'head_dim': 32,
            'd_ff': 1024,
            'n_experts': 4,
            'top_k': 2,
            'resident_slots': 2,
            'engram_layers': [1],
            'engram_slots': 1024,
            'engram_heads': 2,
            'engram_orders': [2, 3],
            'max_window': 256,
            'sink_tokens': 1,
            'weight_bits': 1.58,
            'tie_embeddings': True,
            'max_seq_len': 512,
        }
    })

    generator = PsudoGenerator.from_config(config)
    tokenizer = generator.tokenizer

    # Test generation
    prompt = "fn hello() -> string {"
    gen_config = GenerationConfig(
        max_new_tokens=20,
        temperature=0.8,
        top_k=10,
        top_p=0.9,
    )

    result = generator.generate(prompt, gen_config)

    # Test streaming
    for _token in generator.generate_stream("fn test() { return "):
        pass

    # Stats

