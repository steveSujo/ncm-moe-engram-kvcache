# KV Cache Architecture in NCM-MoE-Engram-KVCache

## Overview

The KV (Key-Value) Cache is a critical optimization for autoregressive generation in transformer models. Instead of recomputing key and value projections for all previous tokens at each generation step, we cache them and only compute for the new token.

In this NCM-MoE-Engram model, the KV cache serves **12 transformer layers** with **Multi-Head Attention (MHA)** across **11 generation levels** (levels 1-11, skipping level 0 which is the prefix).

---

## Architecture Overview

```mermaid
graph TB
    subgraph "Generation Step t"
        direction TB
        Input[New Token] --> Embed[Token + Pos Embed]
        Embed --> Blocks[12 Transformer Blocks]
        Blocks --> Proj[Proj to Codebook]
        Proj --> Heads[11 Level Heads]
        Heads --> Logits[Logits per Level]
        Logits --> Sample[Sample Next Token]
        Sample --> Next[Next Token]
    end

    subgraph "KV Cache System"
        direction TB
        Cache[(KV Cache\n12 Layers × 12 Heads\n256 Window + 1 Sink)]
        Blocks -.->|Read K,V| Cache
        Blocks -.->|Write K,V| Cache
        Engram[Engram Memory] -.->|K,V Fusion| Blocks
    end

    subgraph "Per-Layer KV Cache"
        direction LR
        Layer1[Layer 0] --> Cache1[(K,V)]
        Layer2[Layer 1] --> Cache2[(K,V)]
        Layer3[Layer 2] --> Cache3[(K,V)]
        LayerN[...] --> CacheN[(K,V)]
        Layer12[Layer 11] --> Cache12[(K,V)]
    end

    Blocks --> Layer1
    Blocks --> Layer2
    Blocks --> Layer3
    Blocks --> LayerN
    Blocks --> Layer12
```

---

## KV Cache Data Structure

```mermaid
classDiagram
    class MultiLayerKVCache {
        +n_layers: int = 12
        +max_window: int = 256
        +sink_tokens: int = 1
        +n_kv_heads: int = 12
        +head_dim: int = 64
        +k_cache: Tensor[B, H, T, D]
        +v_cache: Tensor[B, H, T, D]
        +cache_len: int
        +update(layer_idx, k, v) K,V
        +reset()
        +get(layer_idx) K,V
    }

    class BoundedKVCache {
        +max_window: int = 256
        +sink_tokens: int = 1
        +k_cache: Tensor[B, H, T, D]
        +v_cache: Tensor[B, H, T, D]
        +update(k, v) K,V
        +get() K,V
        +reset()
    }

    MultiLayerKVCache "1" *-- "12" BoundedKVCache : contains
```

### Tensor Shapes

| Component | Shape | Description |
|-----------|-------|-------------|
| `k_cache` per layer | `[B, H, T_cached, D]` | Key cache per layer |
| `v_cache` per layer | `[B, H, T_cached, D]` | Value cache per layer |
| `B` | 1-4 | Batch size |
| `H` | 12 | Number of KV heads (MHA) |
| `T_cached` | ≤ 257 | Cached sequence length (256 window + 1 sink) |
| `D` | 64 | Head dimension (768/12) |

---

## KV Cache in Generation Flow

```mermaid
sequenceDiagram
    participant User
    participant Generator as Generator
    participant KVCache as MultiLayerKVCache
    participant Blocks as 12 Transformer Blocks
    participant Attention as SelfAttention

    User->>Generator: Prompt "Once upon a time"
    Generator->>KVCache: reset()
    Generator->>Tokenizer: Encode prompt
    Generator->>Blocks: Forward prompt (prefill)
    
    loop For each generation step
        Generator->>Blocks: Forward new token
        Blocks->>Attention: SelfAttention.forward()
        Attention->>KVCache: update(layer_idx, k, v)
        KVCache-->>Attention: Return updated K,V
        Attention->>Blocks: Return output
        Blocks->>Generator: Logits
        Generator->>Generator: Sample next token
        Generator->>Blocks: Next iteration
    end
    
    Generator->>User: Generated text
```

---

## Attention with KV Cache

```mermaid
flowchart TD
    Input[x: B,T,C] --> QKV[QKV Projection]
    QKV --> Q[Q: B,H,T,D]
    QKV --> K[K: B,H,T,D]
    QKV --> V[V: B,H,T,D]
    
    K --> QKNorm[QK Norm]
    V --> QKNorm
    
    subgraph "KV Cache Update"
        K --> Cache[KV Cache]
        V --> Cache
        Cache --> K_cached
        Cache --> V_cached
    end
    
    K_cached --> ConcatK[Concat K]
    V_cached --> ConcatV[Concat V]
    
    Q --> RoPE[RoPE]
    K_cached --> RoPE
    
    ConcatK --> Attn[Attention Scores]
    Q --> Attn
    
    Attn --> Mask[Causal Mask]
    Mask --> Softmax
    Softmax --> Dropout
    Dropout --> Y[Y = Attn @ V]
    Y --> Proj[Output Projection]
    Y --> Output
    
    Engram[Engram K,V] -.-> K
    Engram -.-> V
    KVCache[KV Cache] -.-> K
    KVCache -.-> V
```

---

## Sliding Window + Attention Sink

```mermaid
graph LR
    subgraph "Cache Window (256 tokens)"
        direction LR
        Sink[Sink: 1 token] --> Recent[Recent 256 tokens]
    end
    
    subgraph "Attention Pattern"
        direction TB
        Query[Query Token] --> Sink
        Query --> Recent
        Sink -.->|Full Attention| Sink
        Recent -.->|Sliding Window| Recent
    end
    
    classDef sink fill:#f9f,stroke:#333
    classDef recent fill:#bbf,stroke:#333
    
    class Sink sink
    class Recent recent
```

### Configuration
```python
# In models/kv_cache.py
max_window = 256      # Sliding window size
sink_tokens = 1       # Prefix length (NCM prefix_len=1)
n_kv_heads = 12       # MHA (not GQA) for NCM compatibility
head_dim = 64         # 768 / 12
```

### Why 1 Sink Token?
- NCM uses `prefix_len=1` for hierarchical generation
- The first token acts as attention sink (StreamingLLM pattern)
- Prevents attention degradation in long sequences

---

## INT8 Quantization (BF16 Model)

```mermaid
flowchart LR
    K[K: BF16] --> Quant[INT8 Quantize]
    V[V: BF16] --> Quant
    
    Quant --> K_q[K_q: INT8]
    Quant --> V_q[V_q: INT8]
    Quant --> K_scale[Scale: FP32]
    Quant --> V_scale[FP32]
    
    K_q --> Store[(KV Cache)]
    V_q --> Store
    K_scale --> Store
    V_scale --> Store
    
    Store --> Dequant[Dequantize]
    Dequant --> K_dq[K: BF16]
    Dequant --> V_dq[V: BF16]
    
    K_dq --> Attention
    V_dq --> Attention
```

### Quantization Details
```python
# models/kv_cache.py
def quantize_int8(x):
    # Per-head per-channel asymmetric quantization
    scale = x.abs().max(dim=-2, keepdim=True).values / 127
    x_q = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
    return x_q, scale

def dequantize_int8(x_q, scale):
    return x_q.to(torch.bfloat16) * scale
```

- **Per-head per-channel**: Each head/channel gets independent scale
- **Storage**: INT8 (1 byte) + FP32 scale (4 bytes) = 5 bytes vs 2 bytes BF16
- **Savings**: ~2× compression for KV cache

---

## 2-bit CQ Quantization (Native 2-bit Branch)

```mermaid
flowchart LR
    K[K: BF16] --> CQ[CQ Quantize]
    V[V: BF16] --> CQ
    
    CQ --> Hadamard[Hadamard Rotate]
    Hadamard --> Normalize[Normalize]
    Normalize --> Codebook[Codebook Lookup]
    Codebook --> Deq[Dequantize]
    Deq --> K_dq[K: BF16]
    Deq --> V_dq[V: BF16]
    
    K_dq --> Attention
    V_dq --> Attention
```

### CQ Pipeline
```python
def cq_quantize_kv(k, v, bits=2.0, group_size=64):
    # 1. Reshape to groups
    # 2. Hadamard rotation (spreads information)
    # 3. Normalize to unit sphere
    # 3. Codebook lookup (Lloyd-Max for Gaussian)
    # 4. Dequantize: codebook * norm
    # 5. Inverse Hadamard
    # 4. Return quantized + norms for dequantization
    return k_q, k_norm, v_q, v_norm
```

### 2-bit Codebook (Lloyd-Max for Gaussian)
| Bits | Levels | Example Values |
|------|--------|----------------|
| 2-bit | 4 | [-0.134, -0.040, 0.040, 0.134] |
| 1.58-bit (ternary) | 3 | [-1.224, 0, +1.224] |

---

## Cache State During Generation

```mermaid
stateDiagram-v2
    [*] --> Reset: reset()
    Reset --> Prefill: Forward prompt
    Prefill --> Step1: Generate token 1
    Step1 --> Step2: Generate token 2
    Step2 --> Step3: Generate token 3
    Step3 --> ...: Continue
    ... --> Full: Window full (257 tokens)
    Full --> Slide: Evict oldest, add new
    Slide --> Slide: Continue sliding
    Slide --> Done: Generation complete
    Done --> Reset: reset()
```

### Cache State at Step N
| Step | Cache Length | Sink | Window | Total |
|------|-------------|------|--------|-------|
| 1 (prefill) | 128 | 1 | 127 | 128 |
| 1 | 129 | 1 | 128 | 129 |
| ... | ... | 1 | 256 | 257 |
| 129+ | 257 | 1 | 256 | 257 (sliding) |

---

## Integration Points in Model

```mermaid
graph TB
    subgraph "MoEEngramNextConceptTransformer"
        Forward[forward()] --> MultiLevel{forward_multilevel}
        MultiLevel --> ProjFrom[proj_from_codebook]
        ProjFrom --> EmbAdd[Add Embeddings]
        EmbAdd --> Blocks[12 Blocks]
        
        subgraph "Block i"
            Block --> LN1[LayerNorm 1]
            LN1 --> Attn[SelfAttention]
            Attn --> KV[KV Cache]
            Attn --> Engram[Engram KV]
            Attn --> LN2[LayerNorm 2]
            LN2 --> MoE[MoE Block]
        end
        
        Blocks --> LNf[Final LayerNorm]
        LNf --> ProjTo[proj_to_codebook]
        ProjTo --> Heads[11 Level Heads]
        Heads --> Logits
    end
    
    KVCache[KV Cache] -.->|Read/Write| Blocks
    Engram[Engram Memory] -.->|K,V Fusion| Blocks
```

### Key Integration Points

| Location | Component | KV Cache Interaction |
|----------|-----------|---------------------|
| `SelfAttention.forward()` | Attention | `kv_cache.update(layer_idx, k, v)` |
| `Block.forward()` | Block | Passes `kv_cache` to attention |
| `Block2Bit.forward()` | 2-bit Block | Same interface |
| `generate()` | Generation | `kv_cache.reset()` at start |
| `forward_single_level()` | Single-level | Passes `kv_cache` to blocks |

---

## Code References

| File | Key Functions |
|------|---------------|
| `models/kv_cache.py` | `BoundedKVCache`, `MultiLayerKVCache`, `quantize_int8`, `dequantize_int8` |
| `models/common.py` | `SelfAttention.forward()` - KV cache integration |
| `models/moe_engram_generator.py` | `forward_multilevel()`, `forward_single_level()`, `generate()` |
| `models/moe_engram_generator_2bit.py` | 2-bit versions with CQ quantization |
| `models/kv_cache.py` | `quantize_int8()`, `dequantize_int8()`, `cq_quantize_kv()` |

---

## Performance Characteristics

| Metric | BF16 (INT8 KV) | 2-bit Native |
|--------|----------------|--------------|
| KV Cache Size | ~4.5 MB | ~1.1 MB |
| Quantization Overhead | Low (INT8) | Medium (CQ) |
| Dequant Latency | Negligible | Low (Hadamard) |
| Quality Loss | <0.1 PPL | <0.5 PPL target |
| VRAM Savings | 2× | 4× |

---

## Debugging KV Cache

```python
# Check cache state
model.kv_cache.cache_len  # Current length
model.kv_cache.get(layer_idx)  # Get K,V for layer

# Verify shapes
k, v = model.kv_cache.get(0)
print(f"K: {k.shape}, V: {v.shape}")  # [B, H, T, D]

# Reset between generations
model.kv_cache.reset()

# Check for NaN
assert not torch.isnan(model.kv_cache.k_cache).any()
```

---

## Summary

The KV cache in NCM-MoE-Engram-KVCache:

1. **Per-layer**: 12 independent caches (one per transformer layer)
2. **MHA**: 12 heads × 64 dim = 768 dim per layer
3. **Window**: 256 tokens + 1 sink token = 257 max
4. **Quantization**: INT8 (BF16 model) / 2-bit CQ (native 2-bit)
5. **Integration**: Transparent in `SelfAttention.forward()` via `kv_cache.update()`
6. **Generation**: Reset at start, accumulates during autoregressive decoding
7. **Sliding**: Automatic eviction when window full (StreamingLLM pattern)

The KV cache is the primary memory optimization enabling long-form generation on 6GB VRAM.