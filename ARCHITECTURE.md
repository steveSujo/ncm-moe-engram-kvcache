# psudo-Engine Architecture Documentation

Deep technical documentation for the psudo-Engine paged-weight hybrid coding agent.

## Table of Contents
1. [Overall Architecture](#overall-architecture)
2. [Component Details](#component-details)
3. [Data Flow](#data-flow)
4. [Memory Management](#memory-management)
5. [Training Pipeline](#training-pipeline)
6. [Generation Pipeline](#generation-pipeline)
6. [Dual-Repository Sync Protocol](#dual-repository-sync-protocol)

---

## Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           psudo-Engine                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐      │
│  │  Weight      │    │   Engram         │    │   Bounded KV       │      │
│  │  Paging      │    │   Memory         │    │   Cache            │      │
│  │  (mini-AGI)  │    │   (Needle 2)     │    │   (Needle 2)       │      │
│  └──────┬───────┘    └────────┬─────────┘    └────────┬────────────┘      │
│         │                     │                    │                    │
│         ▼                     ▼                    ▼                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    PsudoTransformer (30-50M params)              │  │
│  │                                                                   │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │  Embeddings → [Block × 24] → Norm → Output Head            │  │  │
│  │  │  Block: RMSNorm → Attention → Engram? → FFN (MoE)          │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│         │                     │                     │                │
│         ▼                     ▼                     ▼                │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐    │
│  │ psudo-harness │  │ psudo-spec.json  │  │ Deterministic      │    │
│  │ (transpiler)  │  │ (shared contract)│  │ Transpiler        │    │
│  └──────────────┘    └──────────────────┘    └──────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Specifications:**
- **Model Size**: ~30-50M parameters (ternary)
- **Precision**: Ternary weights ({-1, 0, +1}) with STE gradients
- **Target Hardware**: 4GB VRAM (GTX 1660 Ti), 16GB RAM, NVMe SSD
- **Task**: Causal psudo script generation

---

## Component Details

### 1. Weight Paging Subsystem (mini-AGI Style)

**Files**: `psudo_engine/weight_paging/`

| File | Purpose |
|------|---------|
| `weight_store.py` | Layer chunk serialization (.bin per layer), metadata index |
| `ternary.py` | BitNet 1.58b ternary quantization ({-1, 0, +1}) with STE |
| `dequantizer.py` | Batch dequantization to INT8/BF16 on VRAM entry |
| `page_controller.py` | LRU eviction, sequential prefetch |

#### Ternary Quantization (BitNet 1.58b)
- **Values**: {-1, 0, +1} with per-channel scaling factor
- **Forward**: `w_q = round(w / scale).clamp(-1, 1)`
- **Backward**: STE - gradients pass through as `grad * scale`
- **Compression**: ~1.58 bits/param (log₂(3)) ≈ 9x vs BF16

#### Page Controller
- **Resident slots**: 32 experts in VRAM (configurable)
- **Eviction**: LRU with sequential prefetch
- **Disk format**: .bin per layer, safetensors-compatible

---

### 2. Context & Retrieval Subsystem (Needle 2 Style)

**Files**: `psudo_engine/context/`

| File | Purpose |
|------|---------|
| `ast_indexer.py` | tree-sitter Python/Rust parsing → function signatures |
| `syntax_hub.py` | Markdown syntax rules (Python, Rust, psudo builtins) |
| `retriever.py` | Hybrid BM25 + all-MiniLM-L6-v2 embeddings (RRF) |
| `context_builder.py` | Token-budgeted (<1500 tokens) context assembly |

#### Syntax Hub
- **Sources**: Modular Markdown files (`syntax_rules/{python,rust,psudo,generic}/`)
- **Categories**: builtin, control_flow, type, pattern, idiom, stdlib
- **Indexing**: BM25 (keyword) + all-MiniLM-L6-v2 (semantic) → RRF fusion

#### Context Builder
- **Budget**: <1500 tokens per prompt
- **Sources**: Task description + relevant syntax rules + AST signatures
- **Prioritization**: By rule priority + retrieval score

---

### 3. Model Architecture

**Files**: `psudo_engine/model/`

| File | Purpose |
|------|---------|
| `blocks.py` | RMSNorm, MultiHeadAttention, SwiGLU, TransformerBlock |
| `attention.py` | Causal attention with BoundedKVCache |
| `moe.py` | Top-2 router + paged expert execution |
| `engram.py` | Hashed n-gram lookup (2-gram + 3-gram) |
| `kv_cache.py` | BoundedKVCache (256-window + INT8 + sinks) |
| `psudo_transformer.py` | Full model assembly |

#### PsudoTransformer (30-50M params)

**Config** (`config/model.yaml`):
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
```

#### TransformerBlock
```
Input x [B, T, D]
  → RMSNorm → MultiHeadAttention (with KVCache) → + residual
  → Engram injection (if layer in engram_layers)
  → RMSNorm → SwiGLU (MoE) → + residual
  → Output [B, T, D]
```

#### MultiHeadAttention
- **QKV**: Combined projection → split → RoPE → KVCache.update()
- **RoPE**: Precomputed cos/sin tables, applied per-head
- **KVCache**: BoundedKVCache with INT8 quantization, 256-window + 1 sink

#### MoE Layer (Top-2 Router)
```
Router: Linear(d_model, n_experts) → Top-2 → softmax
Experts: SwiGLU (gate/up/down) - loaded on-demand via PageController
Load balancing: Entropy loss + Z-loss
```

#### Engram Memory (Needle 2 Style)
```
Orders: [2, 3] (bi-gram + tri-gram)
Heads: 4
Slots: 8192 per head
Sub-dim: 128 (512/4)
Gating: sigmoid(x @ gate_proj) per head×order
Output: num_heads × num_orders × sub_dim → output_proj → d_model
```

---

### 4. Bounded KV Cache (Needle 2)

**File**: `psudo_engine/model/kv_cache.py`

| Feature | Spec |
|---------|------|
| Window | 256 tokens |
| Sink tokens | 1 (first token pinned) |
| Quantization | INT8 per-head per-channel |
| Dequant | (q.to(fp32) * scale).to(bfloat16) |

**Autoregressive Mask**:
```
key_len = cache_len + query_len  (capped at max_window)
mask = causal_mask(query_len, key_len)  # [1, 1, q_len, key_len]
```

---

## Data Flow

### Training Forward Pass
```
Input tokens [B, T]
  → Embeddings → [B, T, D]
  → for each layer:
      if layer in engram_layers:
          x = EngramMemory(x, tokens, layer_idx)  # additive injection
      x = Block(x, cos, sin, mask, kv_cache)
  → FinalNorm → OutputHead → logits [B, T, V]
  → CrossEntropy(shift_logits, shift_labels)
```

### Inference / Generation
```
Prompt → Tokenize → [B, T]
  → model.reset_kv_cache()
  → for each new token:
      logits = model(current_ids[-1:], use_cache=True)
      next_token = sample(logits[-1] / temp)
      append to sequence
      if eos or max_len: break
```

### Training with Feedback
```
psudo script → Transpiler → Rust → cargo check
    ↓ (error)
LSP/Rustc diagnostics → Harness formats correction pair
    ↓
psudo-engine/data/feedback_loop/corrections_xxx.jsonl
    ↓
Trainer: batch = mix(base_grammar + feedback) → train_step()
```

---

## Memory Management

### VRAM Budget (4GB Target)

| Component | 256d/2L | 512d/12L | 512d/24L |
|-----------|---------|----------|----------|
| Ternary weights | ~15 MB | ~180 MB | ~360 MB |
| Engram tables | ~20 MB | ~67 MB | ~67 MB |
| Resident experts | ~20 MB | ~470 MB | ~470 MB |
| KV Cache (INT8) | ~10 MB | ~60 MB | ~120 MB |
| Activations/grads | ~50 MB | ~400 MB | ~800 MB |
| **Total** | **~21 MB** | **~286 MB** | **~650 MB** |

### Expert Paging
- **Disk**: 64 experts × 24 layers × 3 tensors × BF16 ≈ 11 GB SSD
- **VRAM**: 32 resident experts (LRU)
- **Load**: `pager.ensure_resident(layer, expert_ids)` on-demand

### KV Cache
- **Disk**: None (all in VRAM, INT8)
- **Max window**: 256 tokens per layer
- **Sink**: 1 token pinned permanently

---

## Training Pipeline

### Two-Timescale Optimizer
```python
param_groups = [
    {'params': trunk_params,     'lr': 3e-4},   # attention, embeddings
    {'params': engram_params,    'lr': 1e-4},   # engram tables
    {'params': expert_params,    'lr': 3e-3},   # MoE experts (10×)
]
optimizer = AdamW(param_groups, weight_decay=0.1, betas=(0.9, 0.95))
```

### STE Gradient Flow (Ternary)
```python
# Forward: w_q = ternary_quantize(w)
# Backward: grad_w = grad_output * scale  (STE)
```

### Gradient Accumulation
```python
loss = loss / grad_accum  # 8
loss.backward()
if (step + 1) % grad_accum == 0:
    clip_grad_norm_(1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

### Curriculum Learning
```
Stage 1 (0-2k):   Basic syntax, simple functions
Stage 2 (2k-5k):  Control flow, data structures  
Stage 3 (5k-10k): File I/O, higher-order functions
Stage 4 (10k+):   Full programs, error handling
```

### Feedback Integration
```
Dataset = (1 - ratio) * BaseGrammar + ratio * FeedbackCorrections
Default ratio = 0.3
```

---

## Generation Pipeline

### Autoregressive Decoding
```python
generator.generate(
    prompt="fn hello() -> string {",
    config=GenerationConfig(
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
        use_cache=True
    )
)
```

### Sampling Pipeline
```
logits = model(last_token, use_cache=True)
logits = logits / temperature
logits = top_k(logits, k)  # if k > 0
logits = top_p(logits, p)  # if p < 1.0
logits = repetition_penalty(logits, prev_tokens)
probs = softmax(logits)
token = multinomial(probs)
```

### KV Cache Management
- `model.reset_kv_cache()` at sequence start
- `kv_cache.get_mask(seq_len, layer_idx, device)` per layer
- `kv_cache.update_layer(layer_idx, k, v)` per block
- `model.reset_kv_cache()` between sequences

---

## Dual-Repository Sync Protocol

### Repositories
| Repo | Purpose | Key Files |
|------|---------|-----------|
| **psudo-engine** | Training & Model | Model, ternary layers, training scripts, datasets |
| **psudo-harness** | Execution & Transpiler | Transpiler, LSP client, agent loop |

### Shared Contract: `psudo-spec.json`
```
{
  "version": "1.0.0",
  "grammar": { "entry_point": "program", ... },
  "builtins": [ {"name": "read_file", "signature": "(path: string) -> string"}, ... ],
  "ast_nodes": { "FunctionDef": {...}, "IfStmt": {...}, ... },
  "type_system": { "primitives": [...], "containers": [...], ... },
  "transpiler": { "target": "rust", "dependencies": [...] }
}
```

### Version Locking
```yaml
# Both repos: config/training.yaml and config/transpiler.yaml
schema_version: "1.0.0"
```

### Reflexion Loop
```
1. Model generates psudo script
2. Harness transpiles → Rust
3. cargo check / rust-analyzer LSP
4. On error: Harness captures stderr/LSP JSON
5. Harness formats correction pair:
   { prompt, faulty_code, error_message, corrected_code }
6. Appended to feedback_loop/corrections_xxx.jsonl
7. Periodic sync → psudo-engine/data/feedback_loop/
8. Trainer mixes feedback into training batches (ratio 0.3)
```

### Sync Procedure
```bash
# 1. Update schema
git submodule update --remote  # or vendor copy

# 2. Bump version in psudo-spec.json
# "version": "1.1.0"

# 3. Update both repos
# psudo-engine: config.schema_version = "1.1.0"
# psudo-harness: transpiler.spec_version = "1.1.0"

# 4. Coordinated release
git tag v1.1.0-engine
git tag v1.1.0-harness
```

---

## Benchmarks

### VRAM Usage (GTX 1660 Ti 6GB)
| Model | Peak VRAM | Tokens/sec | ms/token |
|-------|-----------|------------|----------|
| Small (256d, 2L) | 21 MB | 47 | 21 ms |
| Medium (512d, 12L) | 286 MB | 11 | 89 ms |

### Latency Breakdown
| Phase | Small | Medium |
|-------|-------|--------|
| Prefill | 24 ms | 118 ms |
| Decode/token | 21 ms | 89 ms |

---

## Testing

```bash
# All tests
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/ -v

# Integration tests (13 tests, 10 passing)
pytest psudo_engine/tests/test_integration.py -v

# Unit tests
pytest psudo_engine/tests/test_weight_paging.py -v
pytest psudo_engine/tests/test_context.py -v
pytest psudo_engine/tests/test_model.py -v

# Benchmarks
PYTHONPATH=. .venv/Scripts/python.exe psudo_engine/benchmarks/benchmark.py
```

---

## File Index

| Path | Description |
|------|-------------|
| `psudo_engine/weight_paging/weight_store.py` | Layer chunk serialization |
| `psudo_engine/weight_paging/ternary.py` | Ternary quantization (BitNet 1.58b) |
| `psudo_engine/weight_paging/dequantizer.py` | Batch dequant on VRAM entry |
| `psudo_engine/weight_paging/page_controller.py` | LRU eviction page controller |
| `psudo_engine/context/ast_indexer.py` | tree-sitter Python/Rust parsing |
| `psudo_engine/context/syntax_hub.py` | Markdown rule files + indexing |
| `psudo_engine/context/retriever.py` | BM25 + embedding retrieval |
| `psudo_engine/context/context_builder.py` | Token-budgeted context assembly |
| `psudo_engine/model/blocks.py` | RMSNorm, Attention, TransformerBlock |
| `psudo_engine/model/attention.py` | Causal attention + BoundedKVCache |
| `psudo_engine/model/moe.py` | Top-2 router + paged experts |
| `psudo_engine/model/engram.py` | Hashed n-gram (2/3-gram) |
| `psudo_engine/model/kv_cache.py` | BoundedKVCache (256-window + INT8) |
| `psudo_engine/model/psudo_transformer.py` | Full model assembly |
| `psudo_engine/training/psudo_dataset.py` | psudo script dataset |
| `psudo_engine/training/feedback_dataset.py` | Base + feedback merge |
| `psudo_engine/training/ternary_trainer.py` | STE training loop |
| `psudo_engine/training/curriculum.py` | Difficulty progression |
| `psudo_engine/training/train.py` | Main training entry |
| `psudo_engine/training/schema_check.py` | Schema version validation |
| `psudo_engine/inference/tokenizer.py` | BPE tokenizer |
| `psudo_engine/inference/generator.py` | Autoregressive generator |
| `psudo_engine/inference/cli.py` | Demo CLI |
| `psudo_engine/benchmarks/benchmark.py` | Benchmark suite |
| `psudo_engine/data/psudo-spec.json` | Shared schema |
| `docs/sync_protocol.md` | Sync protocol docs |
| `tests/test_integration.py` | 13 integration tests |