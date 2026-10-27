# Topic Map — Paper to Code File Mapping

This document maps each research paper to the specific code files in the NCM-MoE-Engram-KVCache project that implement its ideas.

## Legend

| Symbol | Meaning |
|--------|---------|
| 📄 | Primary implementation |
| 🔗 | Secondary / supporting |
| ⚙️ | Configuration / utilities |

---

## Next-Concept Modeling (NCM)

| Paper | Code File | Component |
|-------|-----------|-----------|
| HazyResearch 2023 (NCM) | `models/next_concept.py` 📄 | Original NCM reference implementation |
| HazyResearch 2023 (NCM) | `models/moe_engram_generator.py` 📄 | Our MoE+Engram adapted NCM |
| HazyResearch 2023 (NCM) | `tokenization/vqvae_tokenizer.py` 📄 | Frozen VQ-VAE tokenizer wrapper |
| van den Oord 2017 (VQ-VAE) | `tokenization/vqvae_tokenizer.py` 📄 | VQ-VAE encoder/decoder |
| Lee 2022 (Residual VQ) | `tokenization/vqvae_tokenizer.py` 📄 | 12-level residual quantization |
| HazyResearch 2023 (NCM) | `models/common.py` 🔗 | Block, SelfAttention, RMSNorm |

**Key NCM Design Decisions in Code**:
- `prefix_len = 1` (sink token for hierarchical generation)
- `vocab_size = [4096] × 11` (11 levels, 4096 each)
- `levels = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]` (641 total)
- Block-diagonal attention (each level attends within level)
- Tied heads: `lm_heads.weight = codebook_embeddings`

---

## Mixture of Experts (MoE)

| Paper | Code File | Component |
|-------|-----------|-----------|
| Shazeer 2017 (MoE) | `models/moe_block.py` 📄 | `Top2Router`, `MoEBlock` |
| Shazeer 2017 (MoE) | `models/moe_block.py` 🔗 | Entropy loss, capacity factor |
| Fedus 2021 (Switch) | `models/moe_block.py` 🔗 | Top-2 routing, expert dropout |
| Zoph 2022 (MoE Design) | `training/init_experts.py` 📄 | K-means expert initialization |
| Zoph 2022 (MoE Design) | `models/moe_engram_generator.py` 🔗 | Expert config (64 experts, 3072 hidden) |
| Lialin 2023 (4GB Training) | `utils/expert_io.py` 📄 | `MoEPager`, LRU cache (32 resident) |
| Roller 2021 (Hash Layers) | `utils/expert_io.py` 🔗 | Deterministic expert assignment |

**Key MoE Config in Code**:
```python
# models/moe_engram_generator.py
num_experts = 64
top_k = 2
capacity_factor = 1.25
expert_hidden = 3072  # 4 × n_embd

# utils/expert_io.py
resident_slots = 32  # 32 experts in VRAM at once
expert_dir = "experts"  # SSD storage
```

---

## Engram Memory

| Paper | Code File | Component |
|-------|-----------|-----------|
| Min 2023 (Needle) | `models/engram.py` 📄 | `EngramModule`, `fnv1a_hash`, `engram_indices` |
| Min 2023 (Needle) | `models/engram.py` 📄 | `needle2_engram_config` |
| Min 2023 (Needle) | `models/moe_engram_generator.py` 🔗 | `engram_layers = [2, 11]` |
| Yang 2024 (Needle 2) | `models/engram.py` 🔗 | MHC/HadamardMLP NOT implemented |
| Krotov 2016 (Dense Assoc) | `models/engram.py` 🔗 | Gating mechanism concept |
| Ramsauer 2020 (Hopfield) | `models/engram.py` 🔗 | Attention-as-memory concept |

**Engram Config in Code**:
```python
# models/engram.py: needle2_engram_config()
{
    "orders": [2, 3],           # bigram + trigram
    "num_heads": 4,             # 4 independent hash tables
    "sub_dim": 128,             # per-head embedding dim
    "slots": 8192,              # hash table size (power of 2)
    "conv_taps": 4,             # Conv1D kernel size
    "conv_dilation": 3,         # dilation for temporal smoothing
}

# models/moe_engram_generator.py
engram_layers = [2, 11]  # adapted from [2, 15] for 12-layer model
```

---

## KV Cache Optimization

| Paper | Code File | Component |
|-------|-----------|-----------|
| Xiao 2023 (StreamingLLM) | `models/kv_cache.py` 📄 | `BoundedKVCache`, sliding window |
| Xiao 2023 (StreamingLLM) | `models/kv_cache.py` 📄 | Sink tokens (prefix_len=1) |
| Liu 2024 / Han 2024 (KV Quant) | `models/kv_cache.py` 📄 | `quantize_int8`, `dequantize_int8` |
| Ge 2023 (CacheGen) | `models/kv_cache.py` 🔗 | Compression concept |
| Zhang 2023 (H2O) | `models/kv_cache.py` 🔗 | Heavy-hitter concept |

**KV Cache Config in Code**:
```python
# models/kv_cache.py
max_window = 256          # sliding window
sink_tokens = 1           # prefix_len=1 (NCM sink)
n_kv_heads = 12           # MHA (not GQA) for pretrained compat
head_dim = 64             # n_embd / n_head

# INT8 per-channel quantization
quantize_int8(x) -> (x_q, scale)
dequantize_int8(x_q, scale) -> x_dq
```

---

## Memory-Constrained Training

| Paper | Code File | Component |
|-------|-----------|-----------|
| Lialin 2023 (4GB Training) | `training/train_moe_engram.py` 📄 | Gradient accumulation, AMP |
| Lialin 2023 (4GB Training) | `utils/expert_io.py` 📄 | Expert paging (32 resident) |
| Rajbhandari 2020 (ZeRO) | `training/train_moe_engram.py` 🔗 | Optimizer state management |
| Dao 2022 (FlashAttention) | N/A | Not used (standard attention) |
| Ramesh 2021 (Limited Memory) | `training/train_moe_engram.py` 📄 | Gradient accumulation, clipping |
| Shazeer 2020 (SwiGLU) | `models/moe_block.py` 📄 | `SwiGLU` expert activation |
| Touvron 2023 (LLaMA) | `models/common.py` 📄 | `RMSNorm`, `RoPE` |

**Training Config in Code**:
```python
# training/train_moe_engram.py
batch_size = 4
accum_steps = 8          # effective batch = 32
max_steps = 50000
warmup_steps = 1000
gradient_clip = 1.0

# Two-timescale optimizer
trunk_lr = 3e-4      # attention, embeddings, heads, routers
engram_lr = 1e-4     # engram tables (slow)
expert_lr = 3e-3     # experts (10× trunk)

# Mixed precision
dtype = torch.bfloat16
```

---

## Cross-Cutting Configuration

| Config File | Purpose |
|-------------|---------|
| `artifacts/tinystories/config.yaml` | HazyResearch NCM config (frozen) |
| `models/moe_engram_generator.py` | Model architecture assembly |
| `training/train_moe_engram.py` | Training loop, optimizer, data |
| `utils/expert_io.py` | Expert I/O, paging, indexing |
| `tokenization/vqvae_tokenizer.py` | Frozen tokenizer interface |

---

## Dependency Graph

```
artifacts/tinystories/config.yaml
    │
    ├── tokenization/vqvae_tokenizer.py (frozen tokenizer)
    │       ├── VQ-VAE encoder/decoder (van den Oord 2017, Lee 2022)
    │       └── get_ncm_inputs() → multi-level codes
    │
    ├── models/moe_engram_generator.py (main model)
    │       ├── models/common.py (Block, SelfAttention, RMSNorm)
    │       ├── models/moe_block.py (MoEBlock, Top2Router) ← Shazeer 2017
    │       ├── models/engram.py (EngramModule) ← Min 2023
    │       ├── models/kv_cache.py (BoundedKVCache) ← Xiao 2023
    │       └── utils/expert_io.py (MoEPager) ← Lialin 2023
    │
    └── training/train_moe_engram.py (training loop)
            ├── Gradient accumulation (Ramesh 2021)
            ├── Two-timescale optimizer (custom)
            ├── Mixed precision AMP (Dao 2022)
            └── Expert checkpointing (Lialin 2023)
```

---

## Verification Checklist

- [x] Every major code file linked to ≥1 paper
- [x] Every paper cited in research notes linked to code
- [x] Key config values traced to literature
- [x] Design decisions documented with source