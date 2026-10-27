# Agent Instructions for psudo-Engine

This repository contains **psudo-Engine** — a paged-weight hybrid coding agent trained on 4GB VRAM using four complementary sparsity techniques:
1. **Weight Paging** (mini-AGI style): 64 MoE experts/layer on SSD, 32 resident in VRAM
2. **Ternary Quantization**: BitNet 1.58b ({-1, 0, +1}) with STE gradients
3. **Engram Memory** (Needle 2 style): Hashed n-gram lookup (2-gram + 3-gram)
4. **Bounded KV Cache** (Needle 2): 256-token sliding window + pinned sinks + INT8

---

## Quick Start for Agents

```bash
# Environment
uv venv .venv
uv pip install -r requirements.txt

# Optional: for synthetic data generation
uv pip install numpy

# Quick test (CPU, random weights)
PYTHONPATH=. .venv/Scripts/python.exe -c "
from psudo_engine.inference.cli import main
import sys
sys.argv = ['cli.py', '--prompt', 'fn hello() {', '--max-tokens', '20']
main()
"

# Run tests
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/ -v

# Run benchmarks
PYTHONPATH=. .venv/Scripts/python.exe psudo_engine/benchmarks/benchmark.py
```

---

## Key Files

| Purpose | File |
|---------|------|
| **Model Config** | `config/model.yaml` |
| **Training Config** | `config/training.yaml` |
| **Transpiler Config** | `config/transpiler.yaml` |
| **Shared Schema** | `psudo_engine/data/psudo-spec.json` |
| **Sync Protocol** | `docs/sync_protocol.md` |

| Component | File |
|-----------|------|
| **Main Training** | `psudo_engine/training/train.py` |
| **Ternary Trainer** | `psudo_engine/training/ternary_trainer.py` |
| **PsudoTransformer** | `psudo_engine/model/psudo_transformer.py` |
| **MoE Block + Router** | `psudo_engine/model/moe.py` |
| **Engram Memory** | `psudo_engine/model/engram.py` |
| **Bounded KV Cache** | `psudo_engine/model/kv_cache.py` |
| **Transformer Blocks** | `psudo_engine/model/blocks.py` |
| **Attention** | `psudo_engine/model/attention.py` |
| **Weight Paging** | `psudo_engine/weight_paging/` |
| **Context/Retrieval** | `psudo_engine/context/` |
| **Training Data** | `psudo_engine/training/psudo_dataset.py` |
| **Feedback Dataset** | `psudo_engine/training/feedback_dataset.py` |
| **Curriculum** | `psudo_engine/training/curriculum.py` |
| **Inference Generator** | `psudo_engine/inference/generator.py` |
| **Tokenizer** | `psudo_engine/inference/tokenizer.py` |
| **CLI** | `psudo_engine/inference/cli.py` |

---

## Rules (MANDATORY)

Read `.omp/RULES.md` before any operation. Key rules:
- **Disk space guard**: Check free space before installs/builds. NEVER run cleanup commands yourself.
- **Logging**: All long-running scripts write to log files (`expert_init.log`, `training.log`)
- **Conventional commits**: `type(scope): description`
- **Resume from logs**: Scripts skip completed work on restart

---

## Common Tasks

### Modify Model Config
Edit `config/model.yaml`:
```yaml
model:
  vocab_size: 8192
  d_model: 512
  n_layers: 24
  n_heads: 8
  head_dim: 64
  n_experts: 64
  top_k: 2
  resident_slots: 32
  engram_layers: [4, 12, 20]
  engram_slots: 8192
  engram_orders: [2, 3]
  engram_heads: 4
  max_window: 256
  sink_tokens: 1
  weight_bits: 1.58  # ternary
```

### Change MoE Config
Edit `psudo_engine/model/moe.py`:
- `n_experts = 64`
- `top_k = 2`
- `capacity_factor = 1.25`

### Adjust KV Cache
Edit `psudo_engine/model/kv_cache.py`:
- `max_window = 256`
- `sink_tokens = 1`

### Change Optimizer LRs
Edit `psudo_engine/training/ternary_trainer.py`:
```python
trunk_lr: 3e-4      # attention, embeddings
engram_lr: 1e-4     # engram tables
expert_lr: 3e-3     # MoE experts (10× trunk)
```

---

## Testing

```bash
# All tests
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/ -v

# Integration tests (13 tests)
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/test_integration.py -v

# Unit tests
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/test_weight_paging.py -v
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/test_context.py -v
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/test_model.py -v

# Benchmarks
PYTHONPATH=. .venv/Scripts/python.exe psudo_engine/benchmarks/benchmark.py
```

---

## Logs

- `logs/training_YYYYMMDD_HHMMSS.log` — training progress
- Check these if script appears stuck

---

## Hardware Requirements

- **GPU**: 4GB+ VRAM (tested on GTX 1660 Ti 6GB)
- **RAM**: 16GB+ system
- **SSD**: 20GB+ free for experts + checkpoints
- **Disk**: 10GB+ free on C: for uv cache

---

## Architecture Decisions

See `ARCHITECTURE.md` for deep technical documentation.
See `docs/sync_protocol.md` for psudo-engine ↔ psudo-harness sync protocol.

---

*This file is for LLM agents working on this repository.*