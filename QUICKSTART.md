# Quickstart Guide

Get the NCM-MoE-Engram-KVCache model running in 5 minutes.

## 1. Environment Setup (30s)

```bash
# Clone and enter
git clone <your-repo-url>
cd ncm_model

# Install dependencies
uv pip install -r requirements.txt
```

## 2. Download Checkpoints (1-2 min)

```bash
# Option A: HuggingFace CLI
hf download hazyresearch/ncm-checkpoints --local-dir artifacts

# Option B: Python (if CLI not available)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='hazyresearch/ncm-checkpoints', local_dir='artifacts', local_dir_use_symlinks=False)
"
```

Expected structure after download:
```
artifacts/
└── tinystories/
    ├── config.yaml
    ├── tokenizer.pt      # VQ-VAE tokenizer (221 MB)
    ├── generator.pt      # Dense NCM generator (356 MB)
    ├── bpe_vocab_8192.pickle
    └── bpe_merges_8192.pickle
```

## 3. Initialize Experts (2-3 min)

```bash
PYTHONPATH=. .venv/Scripts/python.exe training/init_experts.py
```

Creates `experts/` with 768 expert files (12 layers × 64 experts):
```
experts/
├── expert_index.json
├── layer0_expert0.safetensors
├── ...
└── layer11_expert63.safetensors
```

## 4. Verify Installation (30s)

```bash
PYTHONPATH=. .venv/Scripts/python.exe training/smoke_tests.py
```

Expected output:
```
============================================================
RUNNING SMOKE TESTS
============================================================

==================================================
Testing EngramModule...
  Indices shape: torch.Size([2, 128, 8]) ✓
  K, V shape: torch.Size([2, 128, 768]), torch.Size([2, 128, 768]) ✓
  Application output: torch.Size([2, 128, 768]) ✓
EngramModule test PASSED

... (6 more tests) ...

============================================================
ALL SMOKE TESTS PASSED! ✓
============================================================
```

## 5. Run a Training Step (optional, 1 min)

```bash
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

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# One step
batch = torch.randint(0, 8192, (2, 128))
x, targets = mock_vqvae.get_ncm_inputs(batch)
logits, loss = model(x[:, :, 1], level=1, targets=targets)
loss.backward()
optimizer.step()
print(f'Training step loss: {loss.item():.4f}')
"
```

## 6. Generate Text (optional, 10s)

```bash
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
    out = model.generate(torch.tensor([[1]]), max_levels=3)  # Start with BOS=1
    print(f'Generated tokens: {out.tolist()}')
    print(f'Shape: {out.shape}')
"
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: utils` | Run from repo root with `PYTHONPATH=.` |
| `CUDA out of memory` | Reduce `resident_slots` in MoEPager (default 32 → 16) |
| `Expert not found` | Re-run `init_experts.py` |
| `dtype mismatch` | Ensure model and inputs are both BF16 |
| Slow expert loading | Pre-load experts: `pager.ensure_resident(layer, list(range(64)))` |

---

## Next Steps

1. **Full training**: `PYTHONPATH=. .venv/Scripts/python.exe training/train_moe_engram.py`
2. **Monitor**: Check `checkpoints/` for periodic saves
3. **Evaluate**: Use trained checkpoint with real VQ-VAE tokenizer for generation
4. **Scale**: Increase `resident_slots` if VRAM allows, or add more experts