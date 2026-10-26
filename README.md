# NCM-MoE-Engram-KVCache

**Next-Concept Modeling (NCM) Generator with MoE Paging, Engram Memory, and Bounded KV Cache** — trained on 4GB VRAM.

Combines four complementary sparsity techniques:
- **NCM Tokenizer** (frozen): Pre-trained HazyResearch VQ-VAE + multiscale residual VQ (TinyStories)
- **NCM Generator**: Parallel coarse-to-fine transformer (11 levels, block-diagonal attention, tied heads)
- **MoE Paging** (mini-AGI style): 64 experts/layer on SSD, 32 resident in VRAM, dynamic expert loading
- **Engram Key-Value Memory** (Needle 2 / DeepSeek): Hashed n-gram lookup tables (2-gram + 3-gram) at layers 2 & 15
- **Bounded KV Cache** (Needle 2): 256-token sliding window + pinned attention sinks + INT8 quantization

**Target**: ~200M sparse params, ~32M active params/forward, fits 4GB VRAM (BF16 weights + INT8 KV/activations).

---

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA 12.1+ (or CPU for development)
- 4GB+ VRAM GPU for training

### Install
```bash
# Create environment (uv recommended)
uv venv .venv
uv pip install -r requirements.txt
```

### Download Checkpoints
```bash
# NCM tokenizer + generator checkpoints (TinyStories)
hf download hazyresearch/ncm-checkpoints --local-dir artifacts

# Or via Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='hazyresearch/ncm-checkpoints', local_dir='artifacts', local_dir_use_symlinks=False)
"
```

### Initialize Experts (one-time)
```bash
cd A:/source/py/ncm_model  # or your repo root
PYTHONPATH=. .venv/Scripts/python.exe training/init_experts.py
```
This runs k-means distillation from the dense NCM generator checkpoint, creating 12 layers × 64 experts = 768 experts on SSD (~11 GB).

### Run Smoke Tests
```bash
PYTHONPATH=. .venv/Scripts/python.exe training/smoke_tests.py
```
Expected: All 7 tests pass (~30s on CPU).

### Train
```bash
PYTHONPATH=. .venv/Scripts/python.exe training/train_moe_engram.py
```
- Batch size: 4, grad accumulation: 8
- Target: 50k steps
- Checkpoints saved every 1000 steps to `checkpoints/`
- Resident experts checkpointed every 1000 steps

### Generate
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
    out = model.generate(torch.randint(0, 4096, (1, 1)), max_levels=3)
    print(f'Generated: {out.shape}')
"
```

---

## Architecture Details

### Engram Memory (Needle 2 / DeepSeek style)
```
Tokens [B, T] 
  → multi-head n-gram hash (orders [2,3], 4 heads, 8192 slots, FNV-1a)
  → embedding lookup [num_tables, slots, 128] → concat → [B, T, 1024]
  → K/V projections → k_t, v_t [B, T, 768]
  → gate = sigmoid(RMSNorm(h)^T RMSNorm(k) / sqrt(d)) 
  → gated_v = gate * v
  → causal depthwise Conv1D (kernel=4, dilation=3, SiLU) + residual
  → add to residual stream at layers 2, 15
```
- 2 sites × (2 orders × 4 heads × 8192 slots × 128) ≈ 33.6M params (BF16: ~67 MB VRAM)

### MoE Paging
- 12 layers × 64 experts = 768 experts on SSD
- Each expert: SwiGLU (gate/up/down), hidden=3072 (4×)
- Resident pool: 32 experts in VRAM (~470 MB BF16)
- Router: `nn.Linear(768, 64)` per layer (always resident)
- Top-2 routing with capacity factor 1.25
- Entropy loss for load balancing

### Bounded KV Cache
- 256-token sliding window per layer
- Pinned attention sinks (prefix_len=1 from NCM config)
- INT8 per-channel quantization for K/V storage
- Dynamic causal mask handles extended KV length

### Two-Timescale Optimizer
```python
trunk_params      → LR 3e-4,  WD 0.1   (attn, embeddings, heads, engram projs, routers)
engram_table_params → LR 1e-4,  WD 0.01  (Engram embedding tables)
expert_params     → LR 3e-3,  WD 0.1   (MoE expert weights, 10× trunk)
```

---

## Project Structure

```
ncm_model/
├── artifacts/                    # Downloaded checkpoints (not in repo)
│   └── tinystories/
│       ├── config.yaml
│       ├── tokenizer.pt
│       ├── generator.pt
│       ├── bpe_vocab_8192.pickle
│       └── bpe_merges_8192.pickle
├── checkpoints/                  # Training checkpoints (not in repo)
├── data/
│   └── tinystories/
│       └── train.pkl            # Pre-tokenized dataset (dummy by default)
├── experts/                      # MoE expert weights (not in repo, created by init)
│   ├── expert_index.json
│   └── layer{X}_expert{Y}.safetensors
├── models/
│   ├── __init__.py
│   ├── common.py                # Block, SelfAttention, MLP, RMSNorm
│   ├── engram.py                # EngramModule, hashed n-gram lookup
│   ├── kv_cache.py              # BoundedKVCache, INT8 quantization
│   ├── moe_block.py             # MoEBlock, Top2Router
│   └── moe_engram_generator.py  # Integrated MoEEngramNextConceptTransformer
├── tokenization/
│   └── vqvae_tokenizer.py       # Frozen VQ-VAE tokenizer wrapper
├── training/
│   ├── init_experts.py          # K-means expert initialization
│   ├── train_moe_engram.py      # Main training loop
│   ├── smoke_tests.py           # Verification tests
│   └── create_dummy_dataset.py  # Dummy dataset creator
└── utils/
    └── expert_io.py             # MoEPager, ExpertIndex, safetensors I/O
```

---

## Configuration Reference

### Model Config (`artifacts/tinystories/config.yaml`)
```yaml
model:
  n_layer: 12
  n_head: 12
  n_embd: 768
  vocab_size: [4096] * 11      # 11 levels
  context_length: 129
  prefix_len: 1
  tie_heads_to_codebook: true
  use_rope: true
  rope_base: 100000
  use_qk_norm: true
  use_level_encoding: true
  attn_pattern: block_diagonal
  dropout: 0.1
  bias: true
```

### Engram Config (Needle 2 defaults)
```python
{
    "orders": [2, 3],           # bi-gram + tri-gram
    "slots": 8192,              # hash table size
    "sub_dim": 128,             # per-head embedding dim
    "conv_taps": 4,             # causal conv depth
    "conv_dilation": 3,         # dilation = max(order)
    "num_heads": 4,             # multi-head hashing
}
```

### MoE Config
```python
num_experts = 64
top_k = 2
capacity_factor = 1.25
expert_hidden = 3072          # 4× d_model
resident_slots = 32           # VRAM budget
```

### KV Cache Config
```python
max_window = 256              # sliding window
sink_tokens = 1               # attention sinks (prefix_len)
n_kv_heads = 12               # MHA for pretrained compatibility
head_dim = 64                 # n_embd / n_head
```

---

## Training Milestones

| Step | Expected |
|------|----------|
| 500  | Loss decreasing, router entropy > 3.5, Engram gate α ∈ [0.3, 0.7] |
| 5k   | Loss ~5-6, expert utilization balanced (CV < 0.3) |
| 20k  | Loss ~4, coherent generation |
| 50k  | Loss ~3.5, competitive genppl vs dense NCM baseline |

---

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU VRAM | ≥ 4 GB (tested on 4GB) |
| System RAM | ≥ 16 GB (for expert SSD paging + dataset) |
| SSD | ≥ 20 GB free (experts + checkpoints) |
| CPU | AVX2+ for k-means init |

### VRAM Budget (BF16)
| Component | Size |
|-----------|------|
| Tokenizer (frozen) | ~50 MB |
| Trunk (attn + emb + heads) | ~500 MB |
| Engram tables | ~67 MB |
| Resident experts (32) | ~470 MB |
| Activations + KV cache (INT8) | ~1.2 GB |
| Gradients + optimizer states | ~1.5 GB |
| **Total** | **~3.8 GB** |

---

## Extending / Customizing

### Change Engram Layers
```python
model = MoEEngramNextConceptTransformer(
    ...,
    engram_config={"orders": [2,3], "slots": 8192, ...},
)
# Engram layers auto-set to [2, 15] or [2, n_layer-1]
```

### Adjust MoE Residency
```python
# For lower VRAM:
pager = MoEPager(expert_dir, resident_slots=16, ...)

# For higher VRAM:
pager = MoEPager(expert_dir, resident_slots=64, ...)
```

### Freeze Engram Tables
```python
for name, param in model.named_parameters():
    if 'engram.embeddings' in name:
        param.requires_grad = False
# Reduces VRAM, only trains projections
```

### Use Different Dataset
Replace `data/tinystories/train.pkl` with pre-tokenized sequences (list of int lists, length 128).

---

## Troubleshooting

### OOM on 4GB GPU
```python
# Reduce resident experts
resident_slots = 16

# Offload Engram tables to CPU (deterministic indices allow prefetch)
# Modify EngramModule to keep embeddings on CPU

# Reduce KV window
kv_cache_config = {"max_window": 128, ...}

# Use FP16 KV cache instead of INT8 (2× KV memory)
```

### Router Collapse (entropy → 0)
```python
# Increase entropy loss weight
loss = loss + 0.05 * router_loss  # default 0.01

# Temporarily increase top_k
router = Top2Router(..., top_k=3)

# Add expert dropout during training
```

### Slow Expert Loading (Windows)
```python
# Pre-load all experts to RAM at startup
for layer in range(12):
    for expert in range(64):
        pager.ensure_resident(layer, [expert])

# Or use torch.load(mmap=True) instead of safetensors
```

### Dtype Mismatch (BF16 vs FP32)
```python
# Ensure model params are BF16
model = model.to(dtype=torch.bfloat16)

# Or cast inputs in forward
x = x.to(dtype=next(model.parameters()).dtype)
```

---

## References

- **NCM**: HazyResearch/ncm — Next-Concept Modeling with parallel coarse-to-fine generation
- **Needle 2**: Cactus Compute's three-tier caching (sliding KV + Engram + Cactus Quants)
- **DeepSeek Engram**: arXiv:2601.07372 — Conditional memory via hashed n-gram lookup
- **mini-AGI MoE**: SSD-backed expert paging with dynamic residency

---

## License

MIT License — see LICENSE for details.

---

## Citation

```bibtex
@software{ncm_moe_engram_kvcache,
  title = {NCM Generator with MoE Paging, Engram Memory, and Bounded KV Cache},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-repo/ncm-moe-engram-kvcache}
}
```