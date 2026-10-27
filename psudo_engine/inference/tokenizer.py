"""
BPE Tokenizer for psudo Vocabulary.

Production-ready BPE tokenizer with vocabulary management for psudo scripts.
Re-exports and extends the PsudoTokenizer from training.
"""

from pathlib import Path
from typing import Any

from ..training.psudo_dataset import PsudoTokenizer as BasePsudoTokenizer


class PsudoTokenizer(BasePsudoTokenizer):
    """
    BPE Tokenizer for psudo vocabulary.

    Extends the training tokenizer with inference-specific features:
    - Fast encoding/decoding
    - Vocabulary persistence
    - Special token handling
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        vocab_file: str | None = None,
        merges_file: str | None = None
    ):
        if vocab_file and Path(vocab_file).exists():
            self.load(vocab_file)
        else:
            super().__init__(vocab_size=vocab_size)

        # Inference optimizations
        self._encode_cache: dict[str, list[int]] = {}
        self._max_cache_size = 10000

    def encode(self, text: str) -> list[int]:
        """Encode text with caching for repeated inputs."""
        if text in self._encode_cache:
            return self._encode_cache[text]

        result = super().encode(text)

        # Cache management
        if len(self._encode_cache) < self._max_cache_size:
            self._encode_cache[text] = result

        return result

    def encode_batch(
        self,
        texts: list[str],
        max_length: int | None = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> dict[str, Any]:
        """Batch encode with padding and truncation."""
        encoded = [self.encode(text) for text in texts]

        if truncation and max_length:
            encoded = [seq[:max_length] for seq in encoded]

        if padding:
            max_len = max(len(seq) for seq in encoded) if encoded else 0
            if max_length:
                max_len = min(max_len, max_length)
            encoded = [seq + [self.pad_token_id] * (max_len - len(seq)) for seq in encoded]

        if return_tensors == "pt":
            import torch
            return {
                'input_ids': torch.tensor(encoded, dtype=torch.long),
                'attention_mask': torch.tensor(
                    [[1 if tok != self.pad_token_id else 0 for tok in seq] for seq in encoded],
                    dtype=torch.long
                )
            }

        return {'input_ids': encoded}

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """Decode token IDs to text."""
        if skip_special_tokens:
            token_ids = [
                tid for tid in token_ids
                if tid not in (self.pad_token_id, self.bos_token_id, self.eos_token_id)
            ]

        text = super().decode(token_ids)

        if clean_up_tokenization_spaces:
            # Clean up any artifact spaces
            text = text.replace(" \n", "\n").replace("\n ", "\n")

        return text

    def decode_batch(
        self,
        sequences: list[list[int]],
        **kwargs
    ) -> list[str]:
        """Decode batch of token sequences."""
        return [self.decode(seq, **kwargs) for seq in sequences]

    def save(self, path: str):
        """Save tokenizer vocabulary and merges."""
        super().save(path)

        # Also save merges file if using true BPE
        # (placeholder for full BPE implementation)

    @classmethod
    def from_pretrained(cls, path: str) -> 'PsudoTokenizer':
        """Load tokenizer from saved files."""
        vocab_file = Path(path) / "vocab.json"
        Path(path) / "merges.txt"

        if vocab_file.exists():
            return cls(vocab_file=str(vocab_file))
        else:
            # Fallback to base tokenizer
            return cls()

    def get_vocab(self) -> dict[str, int]:
        """Get vocabulary mapping."""
        return self.token_to_id.copy()

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: str | list[str],
        **kwargs
    ) -> dict[str, Any]:
        """Make tokenizer callable like HuggingFace tokenizers."""
        if isinstance(text, str):
            return self.encode_batch([text], **kwargs)
        return self.encode_batch(text, **kwargs)


def create_tokenizer(config: dict[str, Any] | None = None) -> PsudoTokenizer:
    """Create PsudoTokenizer from config."""
    config = config or {}
    return PsudoTokenizer(
        vocab_size=config.get('vocab_size', 8192),
        vocab_file=config.get('vocab_file'),
        merges_file=config.get('merges_file'),
    )


if __name__ == "__main__":
    # Quick test

    tokenizer = PsudoTokenizer(vocab_size=1000)

    # Test basic encode/decode
    text = 'fn hello() -> string {\n    return "world"\n}'
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)


    # Test batch encode
    batch = tokenizer(['fn a() { return 1 }', 'fn b() { return 2 }'], max_length=32)

    # Test callable
    result = tokenizer("fn test() { return 42 }")

