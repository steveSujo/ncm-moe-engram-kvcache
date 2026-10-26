"""
Create dummy TinyStories dataset for testing.
"""

import torch
import pickle
import os


def create_dummy_dataset(output_path: str = "data/tinystories/train.pkl", n_samples: int = 1000, seq_len: int = 128):
    """Create dummy tokenized TinyStories dataset."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate random token sequences (vocab size 8192 for BPE)
    vocab_size = 8192
    data = []
    
    for _ in range(n_samples):
        seq = torch.randint(0, vocab_size, (seq_len,)).tolist()
        data.append(seq)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Created dummy dataset: {n_samples} samples, seq_len={seq_len}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    create_dummy_dataset()