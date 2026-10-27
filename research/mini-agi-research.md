# mini-AGI Research Notes

## Overview

**Repository**: https://github.com/volotat/mini-AGI  
**Type**: Mixture of Experts (MoE) Language Model with Expert Paging  
**Key Innovation**: Three-tier expert storage (Disk → RAM → VRAM) with LRU paging, enabling massive expert pools on limited VRAM

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        mini-AGI Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │    Disk      │    │    RAM       │    │    VRAM      │      │
│  │  (All Experts)│◄───│ (LRU Cache)  │◄───│ (Resident     │      │
│  │  .npy files  │    │  CPU tensors │    │  Experts)    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│        ▲                   ▲                   ▲                │
│        │                   │                   │                │
│        └───────────────────┴───────────────────┘                │
│                    Three-Tier Storage                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Expert System (`minagi/pool.py`, `minagi/paged.py`)

**Expert Architecture** (`pool.py:Expert`):
```
Expert = SwiGLU(w1: [d_ff, d_model], w3: [d_ff, d_model], w2: [d_model, d_ff])
```

**SharedPool** (`pool.py:SharedPool`):
- One pool of experts reachable from every depth (layer)
- No fixed size - grows dynamically
- `gate` parameter per expert for routing
- `n_experts()` returns current count

**PagedPool** (`paged.py:PagedPool`):
- VRAM-limited pool with fixed `resident` slots
- Only `resident` experts exist as CUDA parameters at any moment
- Disk holds every expert, RAM caches recent, VRAM holds active
- LRU eviction between RAM/VRAM tiers
- Adam moments stored **with expert** (not slot) - critical for correct momentum

### 2. Expert Paging System (`paged.py`)

#### Three-Tier Architecture (`paged.py:Tiers`)
```
Disk (.npy files)     RAM (LRU Cache)      VRAM (Resident Slots)
  │                       │                      │
  │  All experts          │  Recently used       │  Active experts
  │  Weights + Moments    │  CPU tensors         │  CUDA Parameters
  ▼                       ▼                      ▼
```

#### Key Classes

| Class | Responsibility |
|-------|---------------|
| `Tiers` | Disk↔RAM↔VRAM tier management, LRU, write-behind |
| `PagedPool` | VRAM slot management, expert admission/eviction |
| `PagedPool.admit()` | Expert selection via text voting |
| `PagedPool._own_moments()` | Optimizer moment management |

#### Paging Algorithm (`paged.py:PagedPool.admit`)

```
For each forward's first pass:
  1. Text votes for experts via router (accumulates since position 0)
  2. Experts with highest votes admitted until `resident` slots full
  3. Temperature=0: greedy top-k; Temperature>0: probabilistic draw
  4. Admitted experts loaded to VRAM slots
  4. Characters route among admitted experts only
```

#### Optimizer Moment Management (Critical)

```python
# paged.py:55-60
# Adam keeps two moments per parameter, belonging to the EXPERT, not the slot
# Stepping an expert with wrong moments = stranger's momentum
# Solution: Every expert on card stepped with ITS OWN moments

def _own_moments(self):
    """Before each optimizer step, ensure each slot has its expert's moments"""
    for slot, expert_id in enumerate(self.slots):
        if not self._own[slot]:
            # Load expert's moments from RAM/disk into slot
            self._load_moments_into_slot(slot, expert_id)
            self._own[slot] = True
```

---

## 2. Attention System (`model.py`)

### Blocked Attention (`model.py:_blocked_recompute`)

```python
# Problem: Full attention matrix O(T²) causes OOM on large contexts
# Solution: Blocked computation with gradient checkpointing

def blocked_attention(q, k, v, P, block=512):
    """
    q: [B, H, T, D] queries
    k, v: [B, H, P+T, D] keys/values (with prefix P)
    Computes attention in blocks of size `block`
    Uses gradient checkpointing to avoid storing full attention matrix
    """
```

**Key Features**:
- Blocked computation: processes attention in blocks of 512
- Gradient checkpointing: recomputes forward in backward pass
- Supports prefix attention (prefix length `P`)
- RoPE (Rotary Positional Embeddings) support
- KV cache support for autoregressive generation

### KV Cache (`model.py:Attention`)

```python
class Attention(nn.Module):
    def forward(self, x, ..., kv_cache=None, ...):
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)
            # Ensures dtype consistency
            k = k.to(q.dtype)
            v = v.to(q.dtype)
```

---

## 3. MoE Routing (`pool.py:PooledMLP`)

```python
class PooledMLP(nn.Module):
    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        
        # Router logits: [B*T, n_experts]
        router_logits = self.gate(x.view(-1, D)).float()
        
        # Top-k routing (top-2 by default)
        top2_probs, top2_indices = torch.topk(
            F.softmax(router_logits, dim=-1), k=2, dim=-1)
        
        # Load experts for this batch
        expert_ids = top2_indices.view(-1).unique()
        self.pool.load_experts(expert_ids.tolist())
        
        # Process through experts
        # ... expert computation ...
```

**Load Balancing** (`pool.py:258-270`):
```python
# Balance loss in loss function
balance_loss = self.balance * (expert_usage * (expert_usage / uniform).log()).sum()
# Charges router for probability on each expert proportional to expert's 
# share of recent admissions
```

---

## 4. Precision System (`precision.py`)

### Compute Dtype Management
```python
# Automatic dtype selection based on GPU
def gpu_bf16_native(device=None):
    """True if GPU has native bf16 arithmetic (Ampere+/RDNA2+)"""
    
def gpu_fp32_instead(device=None):
    """True if bf16 configured but GPU lacks native bf16 and has <14GB VRAM"""
```

**Autocast**: All forwards run inside `torch.autocast(device, dtype=compute_dtype())`

### Moment Storage (`precision.py:219-240`)
```python
def pack_bf16(t):
    """Store bf16 tensor as int16 (numpy has no bf16)"""
    return t.to(torch.bfloat16).view(torch.int16).numpy()

def unpack_bf16(a):
    """Load bf16 array: passes float32 through, converts int16→bf16"""
    if a.dtype == np.int16:
        return torch.from_numpy(a).view(torch.bfloat16).to(torch.float32)
    return a  # already float32 (old format)

def as_bf16(a):
    """Load moment: int16→bf16→fp32 for Adam"""
    return torch.from_numpy(a).view(torch.bfloat16).to(torch.float32)
```

**Storage Format**: 
- Weights: FP32 (full precision)
- Adam moments (m, v): BF16 stored as int16 → halves storage
- On load: moments widened to FP32 for Adam step

---

## 5. Blocked Attention (`model.py`)

### Memory-Efficient Attention
```python
def _blocked_recompute(q, k, v, P, block):
    """Blocked attention with gradient checkpointing"""
    # Splits Q into blocks of size `block` (default 512)
    # Each block checkpointed → recomputed in backward
    # Avoids materializing full [B, H, T, T] attention matrix
```

**Memory Savings**:
- Standard attention: O(B × H × T²) memory for attention matrix
- Blocked: O(B × H × T × block) + recomputation overhead
- Enables 4096+ context on 16GB VRAM

### RoPE (Rotary Positional Embeddings)
```python
def build_rope(block, head_dim, theta, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2) / head_dim))
    t = torch.arange(max_len)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()
```

---

## 6. Live Learning (`live.py`)

### Continuous Learning from Conversation
```python
class LiveLearner:
    """One optimizer step per `chunk` characters of conversation stream"""
    
    # Single continuous stream: user + bot text interleaved
    # No per-exchange boundary - cursor only moves forward
    # Gradient reaches all context characters
    # Same code path as training on corpus
```

**Key Properties**:
- Single continuous stream (no exchange boundaries)
- Gradient reaches all context characters
- Same code path as corpus training
- Plasticity controller continues from training
- Held-out evaluation every few minutes

---

## 7. Expert Lifecycle Management

### Prune Clock (`paged.py:422-450`)
```python
# last_seen: text count when expert last admitted
# Prune: experts not admitted for long time are deleted
# trial period: newborns safe for `trial` steps
# dying_at: share of survival window unaddressed = dying (default 0.75)
```

### Balance Term (Load Balancing)
```python
# Loss term: balance * Σ(prob_i * log(prob_i / uniform_i))
# Charges router for probability on each expert ∝ expert's admission share
# Forces router to spread load across experts
```

### Temperature-Controlled Selection
```python
# Temperature = 0: greedy top-k from text's vote
# Temperature > 0: probabilistic draw from text's router distribution
# Row-by-row draw: at each row, draw experts from text's distribution
```

---

## 8. Live Learning / Continuous Training (`live.py`)

```python
class LiveLearner:
    """Continuous learning from conversation stream"""
    
    # Single continuous stream: user + bot text interleaved
    # No exchange boundaries - cursor only moves forward
    # Gradient reaches all context characters
    # Same code path as corpus training
    # Plasticity controller continues from training
    # Held-out evaluation every few minutes
```

**Key Insight**: No separate "inference" and "training" modes - same code path.

---

## 9. Configuration (`config.yaml`)

```yaml
# Model
vocab_size: 8192
d_model: 768
n_heads: 12
n_layers: 20
d_ff: 3072
dropout: 0.1

# MoE
n_experts: 64
resident: 16          # VRAM resident experts
ram_capacity: 256     # RAM cache size

# Training
chunk: 2048           # chars per optimizer step
batch: 4
lr: 3e-4
weight_decay: 0.1

# Paging
ram_capacity: 256
resident: 16
balance: 0.01         # load balancing loss weight
```

---

## Key Innovations Summary

| Innovation | Description | File |
|------------|-------------|------|
| **Three-tier paging** | Disk→RAM→VRAM with LRU | `paged.py:Tiers` |
| **Moment ownership** | Moments travel with expert, not slot | `paged.py:_own_moments` |
| **Blocked attention** | Checkpointed blocks avoid OOM | `model.py:_blocked_recompute` |
| **Moment storage** | Adam moments in bf16 (packed as int16) | `precision.py:pack_bf16` |
| **Live learning** | Continuous training from conversation | `live.py:LiveLearner` |
| **Load balancing** | Router entropy loss | `pool.py:balance` |
| **Temperature selection** | Greedy (0) or probabilistic (>0) | `paged.py:_draw` |
| **Prune clock** | Expert lifecycle management | `paged.py:prune` |
| **Balance term** | Router entropy regularization | `pool.py:balance` |
| **Live learning** | Continuous conversation training | `live.py:LiveLearner` |

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Model size | ~121M params (20L, 768d, 12H) |
| Experts | 64 (default) |
| Resident | 16 (VRAM) |
| RAM cache | 256 experts |
| Disk | Unlimited |
| Context | 4096 tokens |
| VRAM (16 resident) | ~4-6 GB |
| RAM cache | ~2-4 GB |

---

## Key Design Principles

1. **Moments belong to experts, not slots** - prevents momentum corruption during paging
2. **Router is the source of truth** - paging decisions follow router's vote
3. **Temperature = 0 for training, >0 for generation** - deterministic vs stochastic
4. **Blocked attention + checkpointing** → enables long context on limited VRAM
5. **Moment storage in bf16** - 50% reduction vs fp32, full exponent range
4. **Live learning = training** - no separate inference/training modes
5. **Prune clock = text count** - natural measure of expert relevance
6. **Balance term in loss** - prevents expert collapse

---

## References

- **Repository**: https://github.com/volotat/mini-AGI
- **Key Files**:
  - `minagi/paged.py` - Three-tier paging system
  - `minagi/pool.py` - Expert pool and routing
  - `minagi/model.py` - Transformer blocks, attention, MoE
  - `minagi/precision.py` - Precision management
  - `minagi/live.py` - Live learning
  - `train.py` / `serve.py` - Training and serving entry points