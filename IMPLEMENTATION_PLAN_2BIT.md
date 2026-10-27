# Native 2-bit Training Implementation Plan (ARCHIVED)

## NCM-MoE-Engram-KVCache with Cactus Quants (Needle-style)

---

## Status: ARCHIVED / SUPERSEDED

**This plan has been superseded by the psudo-Engine rewrite (v0.2.0+)** which uses **ternary quantization (1.58-bit)** instead of 2-bit Cactus Quants.

**Reason**: Ternary quantization (BitNet 1.58b) provides:
- Better quality/stability tradeoff than 2-bit CQ
- Simpler implementation (no Hadamard transforms, no codebooks)
- Better hardware compatibility (no Hadamard overhead)
- Proven at scale (BitNet papers)

The psudo-Engine uses **ternary quantization (1.58-bit / {-1, 0, +1})** with STE instead of 2-bit CQ.

---

## Original Plan (Archived for Reference)

### Overview
**Goal**: Train NCM-MoE-Engram-KVCache natively in 2-bit precision using Cactus Quants (CQ) from Needle, achieving ~8x memory reduction vs BF16 while maintaining model quality.

**Target Hardware**: GTX 1660 Ti (6GB VRAM)  
**Target Model**: ~200M sparse params, ~32M active params/forward  
**Target Precision**: 2-bit CQ (Cactus Quants) for weights, BF16 for activations/gradients

---

### Architecture Overview

#### Components to Quantize (2-bit CQ)

| Component | Current | Target | Memory Savings |
|-----------|---------|--------|----------------|
| MoE Experts (gate/up/down) | BF16 | 2-bit CQ | 8x |
| Engram Tables | BF16 | 2-bit CQ | 8x |
| Engram Projections (k_proj, v_proj) | BF16 | 2-bit CQ | 8x |
| Attention Projections (c_attn, c_proj) | BF16 | 2-bit CQ | 8x |
| MoE Router | FP32 | FP32 | - |
| RMSNorm | FP32 | FP32 | - |
| Embeddings (wte, wle, wpe) | BF16 | BF16 | - |
| Output Heads (lm_heads) | BF16 | BF16 | - |
| Projections (proj_to/from_codebook) | BF16 | 2-bit CQ | 8x |
| Engram Embeddings | BF16 | 2-bit CQ | 8x |
| KV Cache | INT8 | 2-bit CQ | 4x |

### Memory Budget (6GB VRAM)

| Component | BF16 Baseline | 2-bit CQ | Savings |
|-----------|---------------|----------|---------|
| Model (resident) | ~431 MB | ~80 MB | 351 MB |
| Optimizer states | ~941 MB | ~235 MB | 706 MB |
| Activations (with checkpointing) | ~125 MB | ~125 MB | - |
| KV Cache (2-bit) | ~4.5 MB | ~1.1 MB | 3.4 MB |
| **Total** | **~1.5 GB** | **~441 MB** | **~1 GB** |

---

## Implementation Phases (Archived)

### Phase 1: Core Quantization Infrastructure (Week 1)
- [x] Hadamard matrix generation (group_size=128)
- [x] Lloyd-Max codebook generation (2-bit, 1.58-bit ternary)
- [x] `cq_quantize()` with Hadamard rotation + codebook lookup
- [x] `CQStraightThrough` autograd.Function (STE)
- [x] `CQQuantizedLinear` and `CQQuantizedEmbedding` modules
- [ ] Add `CQQuantizedLayerNorm` (if needed)
- [ ] Add per-head quantization for attention heads

### Phase 2: 2-bit Model Components (Week 2)
- `models/moe_engram_generator_2bit.py`
- `models/cq_quantize.py` Extensions
- `utils/expert_io.py` - 2-bit Expert Pager
- `models/kv_cache.py` - 2-bit KV Cache

### Phase 3: Training Pipeline (Week 3)
- `training/train_moe_engram_2bit.py`
- `training/init_experts_2bit.py`
- `inference_2bit.py`

### Phase 4: Training & Validation (Week 4)
- 50k steps on TinyStories
- Ablation studies (2-bit vs 4-bit vs 8-bit vs BF16)

---

## Technical Details (Archived)

### Cactus Quants (CQ) Algorithm
```python
def cq_quantize(w, bits=2.0, group_size=128):
    # 1. Pad to group_size multiple
    # 2. Reshape: [..., D] -> [..., num_groups, group_size]
    # 3. Hadamard rotation: groups @ H / sqrt(group_size)
    # 4. Normalize to unit sphere: rot / ||rot||
    # 5. Nearest neighbor in Lloyd-Max codebook
    # 6. Denormalize: deq * norm
    # 7. Inverse Hadamard: deq @ H
    # 8. Remove padding
    return dequantized_weights
```

### Straight-Through Estimator (STE)
```python
class CQStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, bits, group_size, codebook, hadamard):
        return cq_quantize(w, bits, group_size, codebook, hadamard)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None  # Identity gradient
```

---

## Why Superseded by Ternary (1.58-bit)

| Factor | 2-bit CQ | Ternary (1.58-bit) |
|--------|----------|-------------------|
| Quality | Good | Better/Equal |
| Stability | STE noise | More stable |
| Implementation | Complex (Hadamard + codebooks) | Simple (ternary + STE) |
| Hardware | Needs Hadamard kernels | Native tensor ops |
| Codebook | Lloyd-Max optimization needed | Fixed {-1, 0, +1} |
| Proven at scale | Limited | BitNet papers, proven |

The psudo-Engine uses **ternary quantization (1.58-bit)** with BitNet 1.58b scaling mechanics, which provides better quality/stability with simpler implementation.

---

## Current Implementation (psudo-Engine)

The psudo-Engine uses **ternary quantization (1.58-bit)** instead:

```python
# psudo_engine/weight_paging/ternary.py
def ternary_quantize(weight):
    # scale = mean(|weight|) per output channel
    quantized = (weight / scale).round().clamp(-1, 1)
    return quantized.to(torch.int8), scale

# STE for training
class TernaryQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        quantized, scales = ternary_quantize(weight)
        ctx.save_for_backward(weight, scales)
        return quantized, scales
    
    @staticmethod
    def backward(ctx, grad_quantized, grad_scales):
        weight, scales = ctx.saved_tensors
        grad_weight = grad_quantized.float() * scales
        return grad_weight
```

---

## Current Implementation (psudo-Engine)

The psudo-Engine uses **ternary quantization (1.58-bit)** instead:

| File | Description |
|------|-------------|
| `psudo_engine/weight_paging/ternary.py` | Ternary quantization + STE |
| `psudo_engine/model/blocks.py` | TransformerBlock with ternary support |
| `psudo_engine/model/moe.py` | MoE with ternary experts |
| `psudo_engine/model/psudo_transformer.py` | Full model assembly |
| `psudo_engine/training/ternary_trainer.py` | STE training loop |

---

*This document is archived for historical reference. The active development uses ternary quantization (1.58-bit) in psudo-Engine.*
