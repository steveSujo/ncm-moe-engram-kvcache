# Makefile for NCM-MoE-Engram-KVCache

.PHONY: help install download test train generate clean

# Default target
help:
	@echo "NCM-MoE-Engram-KVCache - Available targets:"
	@echo "  install     - Create uv environment and install dependencies"
	@echo "  download    - Download NCM checkpoints from HuggingFace"
	@echo "  init-experts - Run k-means expert initialization from dense checkpoint"
	@echo "  test        - Run smoke tests"
	@echo "  train       - Start full training (50k steps)"
	@echo "  generate    - Run generation demo"
	@echo "  clean       - Remove generated artifacts (checkpoints, experts, data)"
	@echo "  clean-all   - Remove everything including checkpoints and experts"

# Environment setup
install:
	uv venv .venv
	uv pip install -r requirements.txt
# Download checkpoints
download:
	python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='hazyresearch/ncm-checkpoints', local_dir='artifacts', local_dir_use_symlinks=False)"

# Initialize experts (run once after download)
init-experts:
	PYTHONPATH=. .venv/Scripts/python.exe training/init_experts.py

# Run smoke tests
test:
	PYTHONPATH=. .venv/Scripts/python.exe training/smoke_tests.py

# Full training
train:
	PYTHONPATH=. .venv/Scripts/python.exe training/train_moe_engram.py

# Quick generation demo
generate:
	PYTHONPATH=. .venv/Scripts/python.exe -c "
import torch
from omegaconf import OmegaConf
from unittest.mock import MagicMock
from models.moe_engram_generator import MoEEngramNextConceptTransformer
from models.moe_block import MoEBlockNoPager

config = OmegaConf.load('artifacts/tinystories/config.yaml')
mock_vqvae = MagicMock()
mock_vqvae.get_codebooks.return_value = [torch.randn(4096, 384, dtype=torch.bfloat16) for _ in range(11)]
mock_vqvae.get_unigram.return_value = torch.ones(4096, dtype=torch.bfloat16) / 4096
mock_vqvae.get_ncm_inputs.return_value = (torch.randint(0, 4096, (2, 128, 11)), torch.randint(0, 4096, (2, 128)))

model = MoEEngramNextConceptTransformer(vqvae_local=mock_vqvae, config=config.model, expert_dir='experts')
for i, block in enumerate(model.blocks):
    block.mlp = MoEBlockNoPager(config.model, layer_idx=i, n_experts=8, expert_hidden=3072)

model.eval()
with torch.no_grad():
    out = model.generate(torch.tensor([[1]]), max_levels=3)
    print(f'Generated shape: {out.shape}')
    print(f'Tokens: {out.tolist()}')
"

# Clean generated files (keep checkpoints and experts)
clean:
	rm -rf data/tinystories/train.pkl
	rm -rf __pycache__
	rm -rf models/__pycache__
	rm -rf training/__pycache__
	rm -rf utils/__pycache__
	rm -rf tokenization/__pycache__
	find . -name "*.pyc" -delete

# Clean everything including checkpoints and experts
clean-all: clean
	rm -rf checkpoints/*
	rm -rf experts/*
	rm -rf artifacts/*

# Quick development cycle: test -> train -> generate
dev: test train generate

# Full setup from scratch
setup: install download init-experts test
	@echo "Setup complete! Run 'make train' to start training."