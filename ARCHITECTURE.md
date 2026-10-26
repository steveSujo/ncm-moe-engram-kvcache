# Architecture Documentation

Deep technical documentation for the NCM-MoE-Engram-KVCache model.

## Table of Contents
1. [Overall Architecture](#overall-architecture)
2. [Component Details](#component-details)
3. [Data Flow](#data-flow)
4. [Memory Management](#memory-management)
5. [Training Pipeline](#training-pipeline)
6. [Generation Pipeline](#generation-pipeline)

---

## Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NCM-MoE-Engram-KVCache Model                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Input      │    │  Embeddings  │    │  Transformer │                  │
│  │  Tokens      │───▶│  (wte, wle,  │───▶│  Blocks      │                  │
│  │  [B, T]      │    │   wpe)       │    │  ×12         │                  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                  │
│                                                  │                          │
│                    ┌─────────────────────────────┼─────────────────────┐   │
│                    ▼                             ▼                     ▼   │
│             ┌─────────────┐              ┌─────────────┐         ┌─────────┐│
│             │   Engram    │              │   MoE FFN   │         │  Self   ││
│             │  (layers    │              │  (all       │         │ Attention││
│             │   2, 15)    │              │   layers)   │         │  + KV   ││
│             └──────┬──────┘              └──────┬──────┘         │  Cache  ││
│                    │                            │                └────┬────┘│
│                    └────────────────────────────┘                     │      │
│                                         │                             │      │
│                                         ▼                             │      │
│                              ┌─────────────────────┐                 │      │
│                              │    Layer Norm       │                 │      │
│                              │    + LM Heads       │                 │      │
│                              │    (per level)      │                 │      │
│                              └──────────┬──────────┘                 │      │
│                                         │                             │      │
│                    ┌────────────────────┴────────────────────┐       │      │
│                    ▼                                         ▼       ▼      │
│           ┌───────────────┐                          ┌───────────────┐      │
│           │  Next Token   │                          │  Engram KV    │      │
│           │  Logits       │                          │  (for next    │      │
│           │  [B, T, V]    │                          │  layer)       │      │
│           └───────────────┘                          └───────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. NCM Tokenizer (Frozen)

**File**: `tokenization/vqvae_tokenizer.py` → wraps `models/vqvae.py`

**Architecture**:
- BPE tokenizer (8192 vocab) → VQ-VAE encoder (6 layers, 384 dim)
- Multiscale residual VQ with 11 codebooks (4096 entries each, 384 dim)
- Levels: [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128] (manual levels)
- EMA codebook updates, commitment cost 0.25

**Interface**:
```python
tokenizer = VQVAETokenizer.from_config(config.tokenizer)
x, targets = tokenizer.vqvae.get_ncm_inputs(batch_tokens)  # x: [B, L, D], targets: [B, 128]
codebooks = tokenizer.vqvae.get_codebooks()  # List[11] of [4096, 384]
unigram = tokenizer.vqvae.get_unigram()  # [4096]
```

**Frozen**: All params `requires_grad=False`, `eval()` mode.

---

### 2. Engram Memory Module

**File**: `models/engram.py`

**Needle 2 / DeepSeek Style Hashed n-gram Lookup**

#### Hash Function (FNV-1a, 32-bit)
```python
fnv_prime = 16777619
offset_basis = 2166136261 + seed * 12345 + head * 67890
for byte in 4_bytes:
    h_val ^= byte
    h_val = (h_val * fnv_prime) & 0xFFFFFFFF
index = h_val % num_slots
```

#### Multi-Head n-gram Indices
- Orders: [2, 3] (bi-gram + tri-gram)
- 4 heads per order → 8 hash tables total
- 8192 slots per table
- 128-dim sub-embeddings per head
- Total embedding dim: 8 × 128 = 1024

#### Engram Forward
```python
# tokens: [B, T]
indices = engram_indices(tokens, orders=[2,3], num_heads=4, num_slots=8192)  # [B, T, 8]

# Lookup embeddings
embs = [F.embedding(indices[:,:,i], table_i) for i in range(8)]  # 8 × [B, T, 128]
concat_emb = torch.cat(embs, dim=-1)  # [B, T, 1024]

# Project to K, V
k = k_proj(concat_emb)  # [B, T, 768]
v = v_proj(concat_emb)  # [B, T, 768]
```

#### Gated Fusion
```python
# At layers 2 and 15
x_norm = rmsnorm_x(x)  # [B, T, 768]
k_norm = rmsnorm_k(k)  # [B, T, 768]
gate = sigmoid(sum(x_norm * k_norm, dim=-1, keepdim=True) / sqrt(768))  # [B, T, 1]
gated_v = gate * v  # [B, T, 768]

# Causal depthwise conv
conv_in = gated_v.transpose(1, 2)  # [B, 768, T]
conv_in = pad(conv_in, (pad_left, 0))  # causal padding
conv_out = silu(depthwise_conv1d(conv_in))  # [B, 768, T]
out = x + conv_out.transpose(1, 2)  # residual add
```

**Memory**: 2 sites × 8 tables × 8192 slots × 128 dim = 16.8M params/site × 2 = 33.6M params (BF16: 67 MB)

---

### 3. Bounded KV Cache

**File**: `models/kv_cache.py`

**Needle 2 Style: 256-token sliding window + pinned sinks**

#### Structure
```
Layer KV Cache:
┌─────────────────────────────────────────────────────────────┐
│  Sink (pinned)     │  Rolling Window (max 256)              │
│  [B, H, 1, 64]     │  [B, H, W, 64]  where W ≤ 256         │
│  INT8 + scale      │  INT8 + scale                          │
└─────────────────────────────────────────────────────────────┘
Total KV length = 1 + min(seq_len, 256)
```

#### INT8 Quantization
```python
# Per-head per-channel
scale = x.abs().amax(dim=(0, 2)) / 127.0  # [H, D]
q = (x / scale.unsqueeze(0).unsqueeze(2)).round().clamp(-128, 127).to(torch.int8)

# Dequantize
x = (q.to(float32) * scale.unsqueeze(0).unsqueeze(2)).to(bfloat16)
```

#### Update Logic
```python
def update(layer_idx, k, v):  # k,v: [B, H, T_new, D]
    if first_forward:
        sink_k = k[:, :, :1]      # pin first token
        sink_v = v[:, :, :1]
        window_k = k[:, :, 1:]
        window_v = v[:, :, 1:]
    else:
        window_k = cat([window_k, k], dim=2)
        window_v = cat([window_v, v], dim=2)
        if window_k.size(2) > 256:
            window_k = window_k[:, :, -256:]
            window_v = window_v[:, :, -256:]
    
    # Return full KV for attention
    k_full = cat([dequantize(sink_k), dequantize(window_k)], dim=2)
    v_full = cat([dequantize(sink_v), dequantize(window_v)], dim=2)
    return k_full, v_full
```

#### Causal Mask
```python
# Dynamic mask for query_len × (sink + window + query_len)
kv_len = 1 + window_len + query_len
mask = tril(ones(query_len, kv_len))  # [1, 1, query_len, kv_len]
```

---

### 4. MoE FFN with SSD Paging

**Files**: `models/moe_block.py`, `utils/expert_io.py`

#### Expert Architecture (SwiGLU)
```python
# Expert weights: gate, up, down
gate = Linear(d_model, expert_hidden, bias=False)  # [3072, 768]
up   = Linear(d_model, expert_hidden, bias=False)  # [3072, 768]
down = Linear(expert_hidden, d_model, bias=False)  # [768, 3072]

# Forward
gate_out = silu(x @ gate.T)  # [*, 3072]
up_out   = x @ up.T          # [*, 3072]
hidden   = gate_out * up_out # [*, 3072]
out      = hidden @ down.T   # [*, 768]
```

#### Top-2 Router
```python
router = Linear(d_model, 64, bias=False)  # [64, 768]

logits = x @ router.T  # [B*T, 64]
top_logits, top_indices = topk(logits, k=2, dim=-1)  # [B*T, 2]
weights = softmax(top_logits, dim=-1)  # [B*T, 2]
```

#### Entropy Loss
```python
probs = softmax(logits, dim=-1)  # [B*T, 64]
mean_probs = probs.mean(dim=0)   # [64]
entropy = -sum(mean_probs * log(mean_probs + 1e-8))
max_entropy = log(64)
normalized_entropy = entropy / max_entropy  # 0=collapsed, 1=uniform
loss = 1.0 - normalized_entropy  # encourage high entropy
```

#### SSD Paging (MoEPager)
```python
# Expert storage: 12 layers × 64 experts = 768 files
# Each expert: 3 tensors (gate, up, down) in BF16
# Size per expert: 3 × 768 × 3072 × 2 bytes ≈ 14.7 MB
# Total SSD: ~11.3 GB

# Resident pool: LRU cache of 32 experts in VRAM
resident = OrderedDict[(layer, expert)] = {gate, up, down tensors}

def ensure_resident(layer_idx, expert_ids):
    for eid in expert_ids:
        if (layer_idx, eid) in resident:
            move_to_end()  # MRU
        else:
            if len(resident) >= 32:
                evict_lru()  # remove from VRAM
            load_from_ssd()  # safetensors mmap
            resident[key] = tensors
    return {eid: resident[(layer_idx, eid)] for eid in expert_ids}
```

#### MoE Block Forward
```python
def forward(x):  # [B, T, 768]
    x_norm = rmsnorm(x)
    weights, indices, logits = router(x_norm)  # [B, T, 2], [B, T, 2], [B, T, 64]
    router_loss = entropy_loss(logits)
    
    # Collect unique experts needed
    unique_experts = indices.unique().tolist()
    expert_tensors = pager.ensure_resident(layer_idx, unique_experts)
    
    # Process per expert
    out = zeros_like(x_norm)
    for k in range(2):
        eid = indices[:, :, k]  # [B, T]
        w = weights[:, :, k].unsqueeze(-1)  # [B, T, 1]
        for expert_id in unique_experts:
            mask = (eid == expert_id)
            if mask.any():
                expert_x = x_norm[mask]
                expert_out = swiglu_forward(expert_x, expert_tensors[expert_id])
                out[mask] += w[mask] * expert_out
    
    return out
```

---

### 5. Integrated Transformer Block

**File**: `models/moe_engram_generator.py`

```python
class MoEEngramNextConceptTransformer(nn.Module):
    def __init__(self, vqvae_local, config, expert_dir, ...):
        # Embeddings
        self.wte = Embedding(vocab_0, 768)
        self.wle = Embedding(11, 768)      # level encoding
        self.wpe = Embedding(129, 768)     # positional
        
        # 12 Blocks with MoE FFN
        self.blocks = ModuleList([Block(config) for _ in range(12)])
        for i, block in enumerate(blocks):
            block.mlp = MoEBlock(config, layer_idx=i, pager=pager)
        
        # Engram at layers 2, 15
        self.engram_layers = [2, 15]
        self.engrams = ModuleList([EngramModule(...) for _ in range(2)])
        
        # KV Cache
        self.kv_cache = MultiLayerKVCache(n_layers=12, max_window=256, ...)
        
        # Output heads (tied to codebooks via 384-dim projection)
        self.proj_to_codebook = Linear(768, 384)
        self.lm_heads = ModuleList([Linear(384, 4096) for _ in range(11)])
    
    def forward(idx, level=0, targets=None, kv_cache=None):
        # idx: [B, T] token indices for current level
        
        # Embeddings
        x = wte(idx) + wle(level) + wpe(pos)  # [B, T, 768]
        
        cache = kv_cache or self.kv_cache
        
        for i, block in enumerate(blocks):
            # Engram injection BEFORE attention at layers 2, 15
            if i in engram_layers:
                engram_out = engrams[engram_idx].apply_engam(x, idx)
                x = engram_out
            
            # Block forward with KV cache
            x = block(x, kv_cache=cache, layer_idx=i)
        
        x = ln_f(x)
        x = proj_to_codebook(x)  # 768 → 384
        logits = lm_heads[level](x)  # [B, T, 4096]
        
        loss = CE(logits, targets) if targets else None
        return logits, loss
```

---

## Data Flow

### Training Forward Pass
```
Batch [B, 128] (BPE tokens)
    │
    ▼
VQ-VAE.get_ncm_inputs()
    │
    ├─▶ x: [B, 128, 11]  (multi-level codes)
    └─▶ targets: [B, 128] (level-0 tokens)
    
For each level 0..10:
    level_tokens = x[:, :, level]  # [B, 128]
    logits, loss = model(level_tokens, level=level, targets=targets)
    total_loss += loss
    
    # Router losses from all MoE blocks
    router_loss = mean(block.mlp.router_loss for block in blocks)
    total_loss += 0.01 * router_loss
```

### Generation (Autoregressive, Level-by-Level)
```
Level 0: Sample from unigram → l0 [B, 1]

For level in 1..10:
    # Build context from all previous levels
    ctx = cat([l0, l1, ..., l_{level-1}], dim=-1)  # [B, sum_T]
    
    # Forward with KV cache
    logits, _ = model(ctx, level=level, kv_cache=kv_cache)
    
    # Sample next token
    logits = logits[:, -1] / temp
    next_token = sample(logits)
    generated.append(next_token)

Return cat(generated)  # [B, 11] tokens across all levels
```

---

## Memory Management

### VRAM Allocation (4GB Budget)

| Component | Precision | Size | Notes |
|-----------|-----------|------|-------|
| Tokenizer (VQ-VAE) | BF16 | ~50 MB | Frozen |
| Embeddings (wte, wle, wpe) | BF16 | ~30 MB | |
| Transformer trunk (attn + LN) | BF16 | ~400 MB | 12 layers |
| Engram tables (2 sites) | BF16 | ~67 MB | 33.6M params |
| Engram projections | BF16 | ~10 MB | |
| MoE routers (12 × 64 × 768) | BF16 | ~12 MB | |
| Resident experts (32) | BF16 | ~470 MB | Dynamic |
| KV cache (INT8) | INT8 | ~200 MB | 12 layers × 256 × 12 × 64 |
| Activations (BF16) | BF16 | ~1.0 GB | Peak during forward |
| Gradients | BF16 | ~500 MB | |
| Optimizer states (AdamW) | FP32 | ~1.0 GB | 2× params |
| **Total** | | **~3.8 GB** | < 4 GB |

### SSD Storage
- 768 experts × 14.7 MB = ~11.3 GB
- Safetensors format for fast mmap loading
- JSON index for metadata

### Expert Initialization (K-means Distillation)
```python
# From dense NCM checkpoint
dense_fc = state_dict['transformer.blocks.{i}.mlp.c_fc.weight']  # [3072, 768]
dense_proj = state_dict['transformer.blocks.{i}.mlp.c_proj.weight']  # [768, 3072]

# Combine for clustering
combined = cat([fc, proj.T], axis=1)  # [3072, 1536]
kmeans = KMeans(n_clusters=64).fit(combined)

# Split centroids
fc_centroids = centroids[:, :768]  # [64, 768]
proj_centroids = centroids[:, 768:].T  # [768, 64]

# Expand to full expert size with noise
for e in 0..63:
    gate = fc_centroids[e].repeat(3072, 1) + noise
    up   = fc_centroids[e].repeat(3072, 1) + noise
    down = proj_centroids[:, e].repeat(1, 3072) + noise
```

---

## Training Pipeline

### Two-Timescale Optimizer
```python
optimizer = AdamW([
    {'params': trunk_params,      'lr': 3e-4,  'wd': 0.1},
    {'params': engram_table_params, 'lr': 1e-4,  'wd': 0.01},
    {'params': expert_params,     'lr': 3e-3,  'wd': 0.1},
], betas=(0.9, 0.95), fused=True)

scheduler = LambdaLR(optimizer, lr_lambda):
    warmup_steps=1000: linear
    50000 steps: cosine decay
```

### Gradient Clipping
```python
# Only trunk + engram + currently routed experts
routed_experts = model.get_routed_expert_params()
clip_grad_norm_(trunk + engram + routed_experts, max_norm=1.0)
```

### Checkpointing
```python
# Every 1000 steps
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'step': step,
}, f'checkpoints/step_{step}.pt')

# Also save resident experts to SSD
for block in model.blocks:
    block.mlp.pager.save_all_resident()
```

---

## Generation Pipeline

### NCM Multi-Level Generation
```python
def generate(idx, l0=None, max_levels=11):
    kv_cache.reset()
    
    if l0 is None:
        l0 = sample(unigram)  # [B, 1]
    
    generated = [l0]
    
    for level in range(1, max_levels):
        # Hierarchical conditioning: all previous levels
        ctx = cat(generated, dim=-1)  # [B, sum_T]
        
        logits, _ = model(ctx, level=level, kv_cache=kv_cache)
        logits = logits[:, -1] / temperature
        
        # Top-k / top-p sampling
        if top_k > 0: logits = top_k_filter(logits, top_k)
        if top_p < 1.0: logits = top_p_filter(logits, top_p)
        
        next_token = multinomial(softmax(logits), 1)
        generated.append(next_token)
    
    return cat(generated, dim=-1)  # [B, 11]
```

### Key Differences from Standard AR
1. **Level-by-level**: 11 forward passes per sample (not token-by-token)
2. **Hierarchical context**: Each level sees all coarser levels
3. **KV cache per level**: Resets between levels, accumulates within level
4. **Engram at fixed layers**: Always layers 2, 15 regardless of level

---

## Extension Points

### Adding New Engram Sites
```python
model = MoEEngramNextConceptTransformer(
    ...,
    engram_config={"orders": [2,3], "slots": 8192, ...},
)
# Modify self.engram_layers in __init__
```

### Custom Routing
```python
class CustomRouter(nn.Module):
    def forward(self, x):
        # Your routing logic
        return weights, indices, logits
```

### Different Quantization
```python
# In kv_cache.py
def quantize_int4(x):  # 4-bit
    # ...
    
def dequantize_int4(q, scale):
    # ...
```

### Distributed Training
- MoE expert parallelism across GPUs
- Each GPU holds subset of experts
- All-to-all for routed tokens

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Forward pass (1 level, B=4, T=128) | ~200ms CPU, ~50ms GPU |
| Expert load (cache miss) | ~5ms SSD → VRAM |
| Engram forward | ~10ms |
| KV cache update | ~2ms |
| Total training step (accum=8) | ~2s GPU |
| Generation (11 levels) | ~2s GPU |

---

## Debugging Hooks

```python
# Log router statistics
router_entropy = block.mlp.router_loss
expert_usage = pager.get_memory_usage()

# Log Engram gate statistics
gate_mean = engram.gate.mean().item()  # Should be ~0.3-0.7

# Log KV cache stats
window_len = kv_cache.get_window_len(layer_idx)
total_len = kv_cache.get_total_len(layer_idx)

# Monitor VRAM
allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
```