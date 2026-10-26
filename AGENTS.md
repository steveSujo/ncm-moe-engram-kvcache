# Agent Instructions for NCM-MoE-Engram-KVCache

This repository contains a Next-Concept Modeling (NCM) generator trained on 4GB VRAM using four complementary sparsity techniques:
1. Frozen NCM tokenizer (HazyResearch VQ-VAE + residual VQ)
2. MoE paging (64 experts/layer on SSD, 32 resident)
3. Engram memory (Needle 2 style hashed n-gram lookup)
4. Bounded KV cache (256-token window + INT8)

## Quick Start for Agents

```bash
# Environment
uv venv .venv
set UV_CACHE_DIR=A:\uvcache && uv pip install -r requirements.txt

# Download checkpoints
hf download hazyresearch/ncm-checkpoints --local-dir artifacts

# Initialize experts (one-time, ~60 min on CPU)
PYTHONPATH=. .venv/Scripts/python.exe training/init_experts.py

# Run smoke tests
PYTHONPATH=. .venv/Scripts/python.exe training/smoke_tests.py

# Full training (requires GPU)
PYTHONPATH=. .venv/Scripts/python.exe training/train_moe_engram.py
```

## Key Files

| Purpose | File |
|---------|------|
| Main training | `training/train_moe_engram.py` |
| Expert init | `training/init_experts.py` |
| Integrated model | `models/moe_engram_generator.py` |
| MoE block + router | `models/moe_block.py` |
| Engram memory | `models/engram.py` |
| Bounded KV cache | `models/kv_cache.py` |
| Expert I/O (paging) | `utils/expert_io.py` |
| Frozen tokenizer | `tokenization/vqvae_tokenizer.py` |
| Base NCM | `models/common.py`, `models/next_concept.py` |
| Smoke tests | `training/smoke_tests.py` |

## Rules (MANDATORY)

Read `.omp/RULES.md` before any operation. Key rules:
- **Disk space guard**: Check free space before installs/builds. NEVER run cleanup commands yourself.
- **Logging**: All long-running scripts write to log files (`expert_init.log`, `training.log`)
- **Conventional commits**: `type(scope): description`
- **Resume from logs**: Scripts skip completed work on restart

## Common Tasks

### Modify Engram Layers
Edit `models/moe_engram_generator.py` line ~80: `engram_layers = [2, 11]`

### Change MoE Config
Edit `models/moe_block.py` and `models/moe_engram_generator.py`:
- `num_experts = 64`
- `top_k = 2`
- `resident_slots = 32`

### Adjust KV Cache
Edit `models/kv_cache.py` `BoundedKVCache.__init__`:
- `max_window = 256`
- `sink_tokens = 1`

### Change Optimizer LRs
Edit `training/train_moe_engram.py` `get_optimizer()`:
- Trunk: 3e-4
- Engram tables: 1e-4
- Experts: 3e-3

## Testing

```bash
# All tests
PYTHONPATH=. .venv/Scripts/python.exe -m pytest training/ -v

# Specific test file
PYTHONPATH=. .venv/Scripts/python.exe -m pytest training/test_init_experts.py -v
PYTHONPATH=. .venv/Scripts/python.exe -m pytest training/test_train.py -v

# Smoke tests only
PYTHONPATH=. .venv/Scripts/python.exe training/smoke_tests.py
```

## Logs

- `expert_init.log` — expert initialization progress
- `training.log` — training progress
- Check these if script appears stuck

## Hardware Requirements

- GPU: 6GB+ VRAM (tested on 6.44 GB GTX 1660 Ti)
- RAM: 16GB+ system
- SSD: 20GB+ free for experts + checkpoints
- Disk: 10GB+ free on C: for uv cache

## Architecture Decisions

See `notes/DECISIONS.md` for detailed rationale on all major choices.

---

*This file is for LLM agents working on this repository.*