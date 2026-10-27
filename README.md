# psudo-Engine

**Paged-weight hybrid coding agent** — a 4GB VRAM coding agent that generates `psudo` pseudocode, transpiles to Rust, and learns from compiler/LSP feedback.

## Overview

psudo-Engine integrates four complementary techniques to fit a coding agent in 4GB VRAM:

| Technique | Purpose |
|-----------|---------|
| **Weight Paging** (mini-AGI style) | 64 MoE experts/layer on SSD, 32 resident in VRAM, dynamic expert loading |
| **Engram Memory** (Needle 2 style) | Hashed n-gram lookup (2-gram + 3-gram), 4 heads, 8192 slots, gated injection |
| **Bounded KV Cache** (Needle 2) | 256-token sliding window + pinned attention sinks + INT8 quantization |
| **Ternary Quantization** | BitNet 1.58b ({-1, 0, +1}) with STE gradients, ~9x compression |

**Target**: ~30-50M params, ternary precision, trained exclusively on `psudo` script generation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        psudo-Engine                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Weight Paging │  │ Engram Memory    │  │ Bounded KV Cache  │  │
│  │ (mini-AGI)   │  │ (Needle 2)       │  │ (Needle 2)        │  │
│  └──────┬───────┘  └────────┬─────────┘  └────────┬──────────┘  │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              PsudoTransformer (30-50M params)             │  │
│  │  ┌────────────────────────────────────────────────────┐   │  │
│  │  │  Embeddings → [Block × 24] → Norm → Output Head   │   │  │
│  │  │  Block: RMSNorm → Attention → Engram? → FFN (MoE) │   │  │
│  │  └────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ psudo-harness │  │ psudo-spec.json  │  │ Deterministic    │  │
│  │ (transpiler)  │  │ (shared contract)│  │ Transpiler       │  │
│  └──────────────┘  └──────────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA 12.1+ (or CPU for development)
- 4GB+ VRAM GPU for training/inference

### Install
```bash
# Create environment (uv recommended)
uv venv .venv
uv pip install -r requirements.txt

# Optional: install numpy for synthetic data generation
uv pip install numpy
```

### Quick Test (CPU)
```bash
PYTHONPATH=. .venv/Scripts/python.exe -c "
from psudo_engine.inference.cli import main
import sys
sys.argv = ['cli.py', '--prompt', 'fn main() {', '--max-tokens', '20']
main()
"
```

### Run Tests
```bash
# Unit + integration tests
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/ -v

# Or just integration tests
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/test_integration.py -v
```

### Run Benchmarks
```bash
PYTHONPATH=. .venv/Scripts/python.exe psudo_engine/benchmarks/benchmark.py
```

---

## Project Structure

```
psudo-Engine/
├── config/                     # Configuration files
│   ├── model.yaml              # Model architecture
│   ├── training.yaml           # Training hyperparameters
│   └── transpiler.yaml         # Transpiler config
├── psudo_engine/               # Main package
│   ├── config/                 # Config loading
│   ├── weight_paging/          # Weight paging subsystem
│   │   ├── weight_store.py     # Layer chunk serialization
│   │   ├── ternary.py          # Ternary quantization (BitNet 1.58b)
│   │   ├── dequantizer.py      # Batch dequant on VRAM entry
│   │   └── page_controller.py  # LRU eviction page controller
│   ├── context/                # Context & retrieval subsystem
│   │   ├── ast_indexer.py      # tree-sitter Python/Rust parsing
│   │   ├── syntax_hub.py       # Markdown rule files + indexing
│   │   ├── retriever.py        # BM25 + all-MiniLM-L6-v2 embeddings
│   │   └── context_builder.py  # Token-budgeted context assembly
│   ├── model/                  # Model architecture
│   │   ├── blocks.py           # RMSNorm, Attention, TransformerBlock
│   │   ├── attention.py        # Causal attention + BoundedKVCache
│   │   ├── moe.py              # Top-2 router + paged expert execution
│   │   ├── engram.py           # Hashed n-gram lookup (2/3-gram)
│   │   ├── kv_cache.py         # BoundedKVCache (256-window + INT8)
│   │   └── psudo_transformer.py # Full model assembly
│   ├── training/               # Training pipeline
│   │   ├── psudo_dataset.py    # Load/tokenize psudo scripts
│   │   ├── feedback_dataset.py # Merge base + feedback_loop
│   │   ├── ternary_trainer.py  # STE forward/backward training
│   │   ├── curriculum.py       # Difficulty progression
│   │   ├── train.py            # Main training loop
│   │   └── schema_check.py     # Schema version validation
│   ├── inference/              # Inference pipeline
│   │   ├── tokenizer.py        # BPE tokenizer for psudo vocab
│   │   ├── generator.py        # Autoregressive with weight paging
│   │   └── cli.py              # Demo CLI script
│   └── __init__.py
├── config/                     # YAML configs
│   ├── model.yaml
│   ├── training.yaml
│   └── transpiler.yaml
├── data/                       # Data directory
│   ├── psudo-spec.json         # Shared schema (grammar, builtins, AST)
│   └── feedback_loop/          # Feedback loop corrections
├── docs/                       # Documentation
│   └── sync_protocol.md        # Sync protocol documentation
├── benchmarks/                 # Benchmark suite
│   └── benchmark.py            # VRAM, latency, quality benchmarks
├── tests/                      # Test suite
│   ├── test_weight_paging.py
│   ├── test_context.py
│   ├── test_model.py
│   └── test_integration.py
├── psudo_engine/               # Package entry point
└── requirements.txt
```

---

## Configuration

### Model Config (`config/model.yaml`)
```yaml
model:
  vocab_size: 8192        # psudo token vocabulary
  d_model: 512
  n_layers: 24
  n_heads: 8
  head_dim: 64
  d_ff: 2048
  n_experts: 64           # per layer
  top_k: 2
  resident_slots: 32      # VRAM resident experts
  engram_layers: [4, 12, 20]
  engram_slots: 8192
  engram_orders: [2, 3]
  engram_heads: 4
  max_window: 256
  sink_tokens: 1
  weight_bits: 1.58       # ternary
  kv_bits: 8
  act_bits: 8
  tie_embeddings: true
```

### Training Config (`config/training.yaml`)
```yaml
training:
  batch_size: 4
  grad_accum: 8
  max_steps: 50000
  checkpoint_every: 1000
  eval_every: 500
  log_every: 10
  optimizer:
    trunk_lr: 3e-4      # attention, embeddings
    engram_lr: 1e-4     # engram tables
    expert_lr: 3e-3     # MoE experts
    weight_decay: 0.1
    betas: [0.9, 0.95]
  grad_clip: 1.0
  amp: true
  dtype: bfloat16
  curriculum:
    enabled: true
    warmup_steps: 1000
  feedback:
    enabled: true
    feedback_path: "data/feedback_loop/"
    mix_ratio: 0.3
  schema_version: "1.0.0"
```

### Transpiler Config (`config/transpiler.yaml`)
```yaml
transpiler:
  spec_version: "1.0.0"
  grammar:
    entry_point: "program"
  builtins:
    - name: "read_file"
      signature: "(path: string) -> string"
    - name: "write_file"
      signature: "(path: string, content: string) -> ()"
    # ... more builtins
  rust:
    edition: "2021"
    crate_name: "psudo_generated"
    dependencies:
      - name: "tokio"
        version: "1.0"
        features: ["full"]
      # ... more deps
```

---

## Dual-Repository Architecture

psudo-Engine uses a **dual-repository** design to separate training from execution:

| Repository | Purpose | Key Components |
|------------|---------|----------------|
| **psudo-engine** (this repo) | Training & Model | Model definition, ternary layers, training scripts, datasets, test inference |
| **psudo-harness** (separate repo) | Execution & Transpiler | Deterministic transpiler, `psudo` parser/AST compiler, agent loop, LSP client |

### Shared Contract: `psudo-spec.json`

Both repos import the same `psudo-spec.json` (in `psudo_engine/data/psudo-spec.json`), defining:
- Grammar subset (`psudo.exe -s`)
- Valid builtins (`read_file`, `write_file`, `print`, `len`, `range`, etc.)
- AST node types
- Type system (primitives, containers, special types)
- Feedback format (correction pairs)

**Sync Protocol**: See `docs/sync_protocol.md` for the complete protocol.

---

## Reflexion Correction Loop

```
[Model generates psudo] → [Transpiler outputs Rust] → [cargo check / LSP Diagnostics]
                                                                      │
                                               (Detects Type/Syntax Error)
                                                                      ▼
                              [Added to Dataset: (Error + Faulty Code)] ◄── [Harness formats correction pair]
```

1. **Execution & Failure**: Agent attempts task → transpiler converts `psudo` → Rust → `cargo check` throws error
2. **Harness Capture**: `psudo-harness` catches stderr / LSP diagnostic JSON
3. **Correction Formatting**: Harness formats failure into training sample per shared schema
4. **Dataset Ingestion**: Harness appends to local `.jsonl` feedback buffer → synced to `psudo-engine/data/feedback_loop/`

---

## Training

```bash
# Prepare data (requires numpy)
uv pip install numpy
PYTHONPATH=. .venv/Scripts/python.exe -m psudo_engine.training.train --prepare-data

# Train
PYTHONPATH=. .venv/Scripts/python.exe -m psudo_engine.training.train

# With overrides
PYTHONPATH=. .venv/Scripts/python.exe -m psudo_engine.training.train --max-steps 1000 --batch-size 8
```

### Key Training Features
- **Two-timescale optimizer**: Trunk (3e-4), Engram (1e-4), Experts (3e-3)
- **STE gradients** for ternary weights
- **Curriculum learning**: Simple → complex psudo scripts
- **Feedback integration**: Mixes base grammar + compiler/LSP corrections (configurable ratio)
- **Checkpointing**: Every 1000 steps, resume support
- **Schema version check**: Ensures training/transpiler spec alignment

---

## Inference / Generation

```bash
# CLI demo
PYTHONPATH=. .venv/Scripts/python.exe psudo_engine/inference/cli.py --prompt "fn main() {"

# Or programmatically
from psudo_engine.inference.generator import PsudoGenerator, GenerationConfig
from psudo_engine.model.psudo_transformer import create_model
from psudo_engine.inference.tokenizer import PsudoTokenizer

model = create_model(config)
tokenizer = PsudoTokenizer(vocab_size=8192)
generator = PsudoGenerator(model=model, tokenizer=tokenizer)

result = generator.generate("fn main() {", GenerationConfig(max_new_tokens=50))
print(result)
```

### Generation Config
```python
GenerationConfig(
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    stop_tokens=[eos_token_id],
    repetition_penalty=1.0,
    use_cache=True,
    stream=False
)
```

---

## Benchmarks

```bash
PYTHONPATH=. .venv/Scripts/python.exe psudo_engine/benchmarks/benchmark.py
```

Outputs:
- **VRAM**: Peak/average usage (MB)
- **Latency**: Tokens/sec, prefill ms, decode ms/token
- **Quality**: Syntax validity rate, compile success rate (with transpiler)

Example output:
```
small (256d, 2L): 21 MB VRAM, 47 tok/s, 21ms/token
medium (512d, 12L): 286 MB VRAM, 11 tok/s, 89ms/token
```

---

## Tests

```bash
# All tests
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/ -v

# Integration tests only
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/test_integration.py -v

# Unit tests per module
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/test_weight_paging.py -v
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/test_context.py -v
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/test_model.py -v
```

### Current Test Status (13 integration tests)
| Test | Status |
|------|--------|
| test_full_generation_pipeline | ⚠️ Expected fail (untrained model) |
| test_chat_mode | ✅ Passing |
| test_train_step | ✅ Passing |
| test_evaluation | ✅ Passing |
| test_checkpoint_save_load | ⚠️ Windows file-lock issue |
| test_curriculum_progression | ✅ Passing |
| test_synthetic_data_generation | ✅ Passing |
| test_syntax_hub_retrieval | ✅ Passing |
| test_hybrid_retriever | ✅ Passing |
| test_context_builder | ✅ Passing |
| test_synthetic_data_generation | ✅ Passing |
| test_dataset_creation | ✅ Passing |
| test_feedback_dataset | ✅ Passing |

**3 expected failures**: untrained model generation, Windows file-locking, missing numpy for synthetic data.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB | 6-8 GB |
| System RAM | 16 GB | 32 GB |
| SSD | 20 GB free | 50 GB free |
| CPU | AVX2+ | AVX-512 |

**VRAM Budget (BF16 weights + INT8 KV/activations)**
| Component | 256d/2L | 512d/12L |
|-----------|---------|----------|
| Model weights (ternary) | ~15 MB | ~180 MB |
| Engram tables | ~20 MB | ~67 MB |
| Resident experts | ~20 MB | ~470 MB |
| KV cache (INT8) | ~10 MB | ~60 MB |
| Activations/grads | ~50 MB | ~400 MB |
| **Total** | **~21 MB** | **~286 MB** |

---

## Extending

### Add New Builtin
1. Add to `config/transpiler.yaml` builtins list
2. Add to `psudo_engine/data/psudo-spec.json` builtins
3. Update `SyntaxHub` default rules (`psudo/syntax.md`)
4. Implement in `psudo-harness` transpiler

### Custom Engram Configuration
```python
config = OmegaConf.load("config/model.yaml")
config.model.engram_layers = [4, 12, 20]
config.model.engram_slots = 16384
config.model.engram_orders = [2, 3, 4]  # add 4-gram
```

---

## License

MIT License - see LICENSE file for details.

---

## Related Projects

- **psudo-harness** (separate repo): Transpiler, LSP client, agent loop
- **psudo-spec** (shared): `psudo-spec.json` schema definition

---

*Built for 4GB VRAM coding agents. Trained on psudo, runs on Rust.*