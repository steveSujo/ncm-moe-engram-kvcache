"""
Psudo Dataset for Training.

Loads, tokenizes, and prepares psudo scripts for causal language modeling.
Supports base grammar dataset and feedback loop integration.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


@dataclass
class PsudoSample:
    """A single psudo training sample."""
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    metadata: dict[str, Any] = None


class PsudoTokenizer:
    """
    BPE Tokenizer for psudo vocabulary.

    Uses a simple character-level BPE for psudo syntax.
    In production, would use a trained tokenizer.
    """

    # Special tokens
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    # Psudo keywords
    KEYWORDS = [
        "fn", "let", "if", "else", "while", "for", "in", "return",
        "true", "false", "null", "int", "float", "string", "bool",
        "list", "dict", "print", "read_file", "write_file", "len",
        "range", "map", "filter", "zip", "enumerate", "break", "continue",
    ]

    # Operators and punctuation
    OPERATORS = [
        "+", "-", "*", "/", "%", "=", "==", "!=", "<", ">", "<=", ">=",
        "&&", "||", "!", ":", ".", ",", "(", ")", "{", "}", "[", "]",
        "->", "=>", "::",
    ]

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary from keywords, operators, and characters."""
        # Special tokens
        special = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]

        # All single characters (ASCII printable)
        chars = [chr(i) for i in range(32, 127)]

        # Keywords and operators
        tokens = special + self.KEYWORDS + self.OPERATORS + chars

        # Add common multi-char tokens
        multi_char = [
            "def", "class", "import", "from", "as", "pass", "raise", "try",
            "except", "finally", "with", "lambda", "yield", "await", "async",
            "self", "super", "None", "True", "False", "and", "or", "not",
            "is", "in", "isinstance", "hasattr", "getattr", "setattr",
        ]
        tokens.extend(multi_char)

        # Build mappings
        for i, token in enumerate(tokens):
            if i >= self.vocab_size:
                break
            self.token_to_id[token] = i
            self.id_to_token[i] = token

        # Fill remaining with byte tokens
        for i in range(len(tokens), self.vocab_size):
            byte_val = i - len(tokens)
            token = f"<byte_{byte_val}>" if byte_val < 256 else f"<unk_{i}>"
            self.token_to_id[token] = i
            self.id_to_token[i] = token

        # Ensure special tokens have correct IDs
        self.pad_token_id = self.token_to_id[self.PAD_TOKEN]
        self.bos_token_id = self.token_to_id[self.BOS_TOKEN]
        self.eos_token_id = self.token_to_id[self.EOS_TOKEN]
        self.unk_token_id = self.token_to_id[self.UNK_TOKEN]

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        # Simple greedy longest-match tokenization
        tokens = []
        i = 0
        while i < len(text):
            # Try longest match first
            matched = False
            for length in range(min(20, len(text) - i), 0, -1):
                substr = text[i:i+length]
                if substr in self.token_to_id:
                    tokens.append(self.token_to_id[substr])
                    i += length
                    matched = True
                    break

            if not matched:
                # Fall back to single char
                char = text[i]
                if char in self.token_to_id:
                    tokens.append(self.token_to_id[char])
                else:
                    tokens.append(self.unk_token_id)
                i += 1

        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        chars = []
        for tid in token_ids:
            if tid in self.id_to_token:
                token = self.id_to_token[tid]
                if token.startswith("<byte_"):
                    byte_val = int(token[6:-1])
                    chars.append(chr(byte_val))
                elif not token.startswith("<"):
                    chars.append(token)
        return "".join(chars)

    def encode_batch(self, texts: list[str], max_length: int = 512) -> torch.Tensor:
        """Encode batch of texts."""
        batch = []
        for text in texts:
            ids = self.encode(text)
            if len(ids) > max_length:
                ids = ids[:max_length]
            batch.append(ids)

        # Pad to max length
        max_len = max(len(ids) for ids in batch)
        padded = []
        for ids in batch:
            padded_ids = ids + [self.pad_token_id] * (max_len - len(ids))
            padded.append(padded_ids)

        return torch.tensor(padded, dtype=torch.long)

    def __len__(self) -> int:
        return self.vocab_size

    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            'vocab_size': self.vocab_size,
            'token_to_id': self.token_to_id,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'PsudoTokenizer':
        """Load tokenizer from file."""
        with open(path) as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        tokenizer.pad_token_id = tokenizer.token_to_id[tokenizer.PAD_TOKEN]
        tokenizer.bos_token_id = tokenizer.token_to_id[tokenizer.BOS_TOKEN]
        tokenizer.eos_token_id = tokenizer.token_to_id[tokenizer.EOS_TOKEN]
        tokenizer.unk_token_id = tokenizer.token_to_id[tokenizer.UNK_TOKEN]
        return tokenizer


class PsudoDataset(Dataset):
    """
    Psudo script dataset for causal language modeling.

    Loads psudo scripts from JSONL files and tokenizes them.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PsudoTokenizer,
        max_length: int = 512,
        bos_token: bool = True,
        eos_token: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.samples = self._load_data(data_path)

    def _load_data(self, data_path: str) -> list[dict]:
        """Load data from JSONL file."""
        samples = []
        path = Path(data_path)

        if not path.exists():
            return samples

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Extract code/text from sample
        if isinstance(sample, str):
            code = sample
        elif isinstance(sample, dict):
            code = sample.get('code', sample.get('text', sample.get('content', '')))
        else:
            code = str(sample)

        # Tokenize
        token_ids = self.tokenizer.encode(code)

        # Add special tokens
        if self.bos_token:
            token_ids = [self.tokenizer.bos_token_id] + token_ids
        if self.eos_token:
            token_ids = token_ids + [self.tokenizer.eos_token_id]

        # Truncate or pad
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        # Create input/target for causal LM
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        # Pad if needed
        if len(input_ids) < self.max_length - 1:
            pad_len = self.max_length - 1 - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            target_ids = target_ids + [-1] * pad_len  # -1 for ignore_index

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(target_ids, dtype=torch.long),
        }

    def get_collate_fn(self):
        """Return collate function for DataLoader."""
        def collate_fn(batch):
            input_ids = torch.stack([item['input_ids'] for item in batch])
            labels = torch.stack([item['labels'] for item in batch])
            return {'input_ids': input_ids, 'labels': labels}
        return collate_fn


class StreamingPsudoDataset(IterableDataset):
    """
    Streaming dataset for large psudo corpora.

    Reads JSONL files line by line without loading all into memory.
    """

    def __init__(
        self,
        data_paths: list[str],
        tokenizer: PsudoTokenizer,
        max_length: int = 512,
        shuffle: bool = True,
        buffer_size: int = 10000
    ):
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle = shuffle
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # Collect all lines from all files
        all_lines = []
        for path in self.data_paths:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_lines.append((path, line))

        if self.shuffle:
            np.random.shuffle(all_lines)

        for _path, line in all_lines:
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Extract code
            if isinstance(sample, str):
                code = sample
            elif isinstance(sample, dict):
                code = sample.get('code', sample.get('text', sample.get('content', '')))
            else:
                code = str(sample)

            # Tokenize
            token_ids = self.tokenizer.encode(code)

            # Add special tokens
            token_ids = [self.tokenizer.bos_token_id] + token_ids + [self.tokenizer.eos_token_id]

            # Truncate
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]

            # Create input/target
            input_ids = token_ids[:-1]
            target_ids = token_ids[1:]

            # Pad
            if len(input_ids) < self.max_length - 1:
                pad_len = self.max_length - 1 - len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                target_ids = target_ids + [-1] * pad_len

            yield {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(target_ids, dtype=torch.long),
            }


class FeedbackDataset(Dataset):
    """
    Dataset combining base grammar data with feedback corrections.

    Feedback data comes from the transpiler harness (compiler/LSP errors).
    """

    def __init__(
        self,
        base_dataset: PsudoDataset,
        feedback_path: str,
        tokenizer: PsudoTokenizer,
        max_length: int = 512,
        feedback_ratio: float = 0.3
    ):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feedback_ratio = feedback_ratio

        # Load feedback data
        self.feedback_samples = self._load_feedback(feedback_path)

        # Compute lengths
        self.base_len = len(base_dataset)
        self.feedback_len = len(self.feedback_samples)
        self.total_len = self.base_len + self.feedback_len

    def _load_feedback(self, feedback_path: str) -> list[dict]:
        """Load feedback correction pairs."""
        samples = []
        path = Path(feedback_path)

        if not path.exists():
            return samples

        for file_path in path.glob("*.jsonl"):
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        # Expected format: {"prompt": "...", "faulty_code": "...", "error": "...", "corrected_code": "..."}
                        if all(k in sample for k in ['faulty_code', 'corrected_code']):
                            samples.append(sample)
                    except json.JSONDecodeError:
                        continue

        return samples

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Decide whether to sample from base or feedback
        if idx < self.base_len:
            return self.base_dataset[idx]
        else:
            feedback_idx = idx - self.base_len
            return self._process_feedback(self.feedback_samples[feedback_idx])

    def _process_feedback(self, sample: dict) -> dict[str, torch.Tensor]:
        """Process a feedback sample into training format."""
        # Create training example: prompt + faulty_code + error -> corrected_code
        prompt = sample.get('prompt', '')
        faulty = sample.get('faulty_code', '')
        error = sample.get('error', '')
        corrected = sample.get('corrected_code', '')

        # Format as instruction tuning
        text = f"# Task: {prompt}\n# Error: {error}\n# Faulty:\n{faulty}\n# Corrected:\n{corrected}"

        token_ids = self.tokenizer.encode(text)
        token_ids = [self.tokenizer.bos_token_id] + token_ids + [self.tokenizer.eos_token_id]

        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        if len(input_ids) < self.max_length - 1:
            pad_len = self.max_length - 1 - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            target_ids = target_ids + [-1] * pad_len

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(target_ids, dtype=torch.long),
        }

    def get_feedback_stats(self) -> dict[str, Any]:
        """Get statistics about feedback data."""
        return {
            'base_samples': self.base_len,
            'feedback_samples': self.feedback_len,
            'total_samples': self.total_len,
            'feedback_ratio': self.feedback_len / self.total_len if self.total_len > 0 else 0,
        }

    def get_collate_fn(self):
        """Return collate function for DataLoader."""
        def collate_fn(batch):
            input_ids = torch.stack([item['input_ids'] for item in batch])
            labels = torch.stack([item['labels'] for item in batch])
            return {'input_ids': input_ids, 'labels': labels}
        return collate_fn


def create_datasets(config: dict[str, Any]) -> tuple[PsudoDataset, FeedbackDataset | None]:
    """Create train/val datasets from config."""
    tokenizer = PsudoTokenizer(vocab_size=config.get('vocab_size', 8192))

    # Base dataset
    train_dataset = PsudoDataset(
        data_path=config.get('train_data_path', 'data/base_grammar.jsonl'),
        tokenizer=tokenizer,
        max_length=config.get('max_length', 512),
    )

    # Validation dataset
    val_dataset = PsudoDataset(
        data_path=config.get('val_data_path', 'data/val_grammar.jsonl'),
        tokenizer=tokenizer,
        max_length=config.get('max_length', 512),
    )

    # Feedback dataset (optional)
    feedback_dataset = None
    if config.get('feedback_enabled', False):
        feedback_dataset = FeedbackDataset(
            base_dataset=train_dataset,
            feedback_path=config.get('feedback_path', 'data/feedback_loop/'),
            tokenizer=tokenizer,
            max_length=config.get('max_length', 512),
            feedback_ratio=config.get('feedback_ratio', 0.3),
        )

    return train_dataset, val_dataset, feedback_dataset, tokenizer


def generate_synthetic_data(
    output_path: str,
    num_samples: int = 10000,
    vocab_size: int = 8192
):
    """Generate synthetic psudo scripts for initial training."""

    templates = [
        # Simple functions
        'fn add(a: int, b: int) -> int {{\n    return a + b\n}}',
        'fn multiply(x: int, y: int) -> int {{\n    return x * y\n}}',
        'fn factorial(n: int) -> int {{\n    if n <= 1 {{\n        return 1\n    }}\n    return n * factorial(n - 1)\n}}',

        # File operations
        'fn read_config(path: string) -> string {{\n    return read_file(path)\n}}',
        'fn write_output(path: string, content: string) -> () {{\n    write_file(path, content)\n}}',

        # Loops and control flow
        'fn sum_range(start: int, end: int) -> int {{\n    let total = 0\n    for i in range(start, end) {{\n        total = total + i\n    }}\n    return total\n}}',
        'fn find_max(items: list<int>) -> int {{\n    let max_val = items[0]\n    for item in items {{\n        if item > max_val {{\n            max_val = item\n        }}\n    }}\n    return max_val\n}}',

        # Data structures
        'fn count_items(items: list<string>) -> dict<string, int> {{\n    let counts = dict()\n    for item in items {{\n        if item in counts {{\n            counts[item] = counts[item] + 1\n        }} else {{\n            counts[item] = 1\n        }}\n    }}\n    return counts\n}}',

        # Conditionals
        'fn check_positive(x: int) -> bool {{\n    if x > 0 {{\n        return true\n    }} else {{\n        return false\n    }}\n}}',
        'fn classify_number(n: int) -> string {{\n    if n < 0 {{\n        return "negative"\n    }} else if n == 0 {{\n        return "zero"\n    }} else {{\n        return "positive"\n    }}\n}}',
    ]

    samples = []
    for i in range(num_samples):
        template = np.random.choice(templates)
        # Add some variation
        code = template
        samples.append({
            'code': code,
            'id': f'synthetic_{i}',
        })

    # Write to JSONL
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')



if __name__ == "__main__":
    # Quick test

    tokenizer = PsudoTokenizer(vocab_size=1000)

    # Test encode/decode
    text = 'fn hello() -> string {\n    return "world"\n}'
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    # Generate synthetic data
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "synthetic.jsonl"
        generate_synthetic_data(str(data_path), num_samples=100)

        # Test dataset
        dataset = PsudoDataset(str(data_path), tokenizer, max_length=128)

        if len(dataset) > 0:
            sample = dataset[0]

