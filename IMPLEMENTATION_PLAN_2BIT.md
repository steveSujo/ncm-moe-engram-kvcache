# Native 2-bit Training Implementation Plan
## NCM-MoE-Engram-KVCache with Cactus Quants (Needle-style)

---

## Overview

**Goal**: Train NCM-MoE-Engram-KVCache natively in 2-bit precision using Cactus Quants (CQ) from Needle, achieving ~8x memory reduction vs BF16 while maintaining model quality.

**Target Hardware**: GTX 1660 Ti (6GB VRAM)  
**Target Model**: ~200M sparse params, ~32M active params/forward  
**Target Precision**: 2-bit CQ (Cactus Quants) for weights, BF16 for activations/gradients

---

## Architecture Overview

### Components to Quantize (2-bit CQ)

| Component | Current | Target | Memory Savings |
|-----------|---------|--------|----------------|
| MoE Experts (gate/up/down) | BF16 | 2-bit CQ | 8× |
| Engram Tables | BF16 | 2-bit CQ | 8× |
| Engram Projections (k_proj, v_proj) | BF16 | 2-bit CQ | 8× |
| Attention Projections (c_attn, c_proj) | BF16 | 2-bit CQ | 8× |
| MoE Router | FP32 | FP32 | - |
| RMSNorm | FP32 | FP32 | - |
| Embeddings (wte, wle, wpe) | BF16 | BF16 | - |
| Output Heads (lm_heads) | BF16 | BF16 | - |
| Projections (proj_to/from_codebook) | BF16 | 2-bit CQ | 8× |
| Engram Embeddings | BF16 | 2-bit CQ | 8× |
| KV Cache | INT8 | 2-bit CQ | 4× |

### Memory Budget (6GB VRAM)

| Component | BF16 Baseline | 2-bit CQ | Savings |
|-----------|---------------|----------|---------|
| Model (resident) | ~431 MB | ~80 MB | 351 MB |
| Optimizer states | ~941 MB | ~235 MB | 706 MB |
| Activations (with checkpointing) | ~125 MB | ~125 MB | - |
| KV Cache (2-bit) | ~4.5 MB | ~1.1 MB | 3.4 MB |
| **Total** | **~1.5 GB** | **~441 MB** | **~1 GB** |

---

## Implementation Phases

### Phase 1: Core Quantization Infrastructure (Week 1)

#### 1.1 Complete `models/cq_quantize.py`
- [x] Hadamard matrix generation (group_size=128)
- [x] Lloyd-Max codebook generation (2-bit, 1.58-bit ternary)
- [x] `cq_quantize()` with Hadamard rotation + codebook lookup
- [x] `CQStraightThrough` autograd.Function (STE)
- [x] `CQQuantizedLinear` and `CQQuantizedEmbedding` modules
- [ ] Add `CQQuantizedLayerNorm` (if needed)
- [ ] Add per-head quantization for attention heads

#### 1.2 Validation Tests
- [ ] Round-trip quantization error < 0.1%
- [ ] STE gradient flow verification
- [ ] SNR > 20dB for 2-bit
- [ ] No NaN/Inf in forward/backward

### Phase 2: 2-bit Model Components (Week 2)

#### 2.1 `models/moe_engram_generator_2bit.py`
- [ ] `CQQuantizedAttention` - 2-bit c_attn/c_proj with STE
- [ ] `CQQuantizedMoEBlock` - 2-bit experts with FP32 router
- [ ] `Block2Bit` - Pre-LN block with 2-bit attention + MoE
- [ ] `EngramModule2Bit` - 2-bit embeddings + projections
- [ ] `MultiLayerKVCache2Bit` - 2-bit CQ KV cache
- [ ] `MoEEngramNextConceptTransformer2Bit` - Full 2-bit model

#### 2.2 `models/cq_quantize.py` Extensions
- [ ] `CQQuantizedLayerNorm` (if needed)
- [ ] Per-head quantization for attention
- [ ] `configure_quantization()` for global settings
- [ ] Quantization-aware training utilities

#### 2.3 `utils/expert_io.py` - 2-bit Expert Pager
- [ ] `CQQuantizedMoEPager` - saves/loads experts in 2-bit CQ
- [ ] Quantize on save, dequantize on load
- [ ] LRU cache with 2-bit resident experts

#### 2.4 `models/kv_cache.py` - 2-bit KV Cache
- [ ] `MultiLayerKVCache2Bit` with 2-bit CQ storage
- [ ] `quantize_kv_cq()` / `dequantize_kv_cq()` functions
- [ ] Per-head per-channel 2-bit quantization

### Phase 3: Training Pipeline (Week 3)

#### 3.1 `training/train_moe_engram_2bit.py`
- [ ] New training script for 2-bit model
- [ ] BF16 mixed precision + 2-bit weights (STE)
- [ ] Gradient accumulation (accum=16 for effective batch 64)
- [ ] Cosine decay with warmup (1000/50000)
- [ ] Gradient clipping (max_norm=1.0)
- [ ] Periodic `empty_cache()` and checkpointing

#### 3.2 `training/init_experts_2bit.py`
- [ ] Initialize experts directly in 2-bit CQ format
- [ ] K-means on dense weights → 2-bit CQ codebook
- [ ] Save experts in 2-bit format

#### 3.3 `inference_2bit.py`
- [ ] Quantized inference with pre-quantized weights
- [ ] `prepare_for_inference()` for all quantized modules
- [ ] KV cache 2-bit quantization

### Phase 4: Training & Validation (Week 4)

#### 4.1 Training Run
- [ ] 50k steps on TinyStories
- [ ] Monitor: loss, router_loss, perplexity
- [ ] Checkpoint every 1000 steps
- [ ] Memory profiling

#### 4.2 Validation
- [ ] Compare perplexity vs BF16 baseline
- [ ] Generation quality evaluation
- [ ] Memory profiling (peak VRAM)
- [ ] Speed benchmark (steps/sec)

#### 4.3 Ablation Studies
- [ ] 2-bit vs 4-bit vs 8-bit vs BF16
- [ ] STE vs fake_quant gradient quality
- [ ] Codebook size ablation (2-bit vs 1.58-bit ternary)

---

## Technical Details

### Cactus Quants (CQ) Algorithm

```
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

### Codebook Construction

```python
# 2-bit: 4 levels (Lloyd-Max for N(0,1))
# 1.58-bit: 3 levels (ternary: {-α, 0, +α})
# Hadamard matrix: H_1 = [1]; H_{2n} = [H, H; H, -H] / sqrt(2)
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

### Expert Quantization Pipeline

```
Training:
  1. Load expert weights (FP32)
  2. cq_quantize() → 2-bit
  3. Save to SSD (2-bit format)
  
Training (QAT):
  1. Load 2-bit expert
  2. STE forward: quantized weights
  3. STE backward: identity gradient
  4. Update FP32 master weights
  4. Periodically re-quantize and save
  
Inference:
  1. Load 2-bit expert
  2. Dequantize on-the-fly (or keep quantized)
```

### KV Cache 2-bit Quantization

```python
def cq_quantize_kv(k, v, bits=2.0, group_size=64):
    # Per-head per-channel CQ quantization
    # Returns: k_q, k_norm, v_q, v_norm
    pass

def dequantize_kv_cq(k_q, k_norm, v_q, v_norm):
    # Dequantize for attention computation
    pass
```

---

## File Structure

```
models/
├── common.py                 # Shared components (RMSNorm, etc.)
├── cq_quantize.py           # CQ quantization core (DONE)
├── engram.py                # EngramModule (BF16 version)
├── engram_2bit.py           # NEW: EngramModule2Bit
├── kv_cache.py              # KV cache (INT8 + 2-bit)
├── moe_block.py             # MoEBlock (BF16 version)
├── moe_block_2bit.py        # NEW: CQQuantizedMoEBlock
├── moe_engram_generator.py  # BF16 model
├── moe_engram_generator_2bit.py  # NEW: 2-bit model
├── moe_engram_generator_2bit.py  # NEW: 2-bit model

utils/
├── expert_io.py             # BF16 pager
├── expert_io_2bit.py        # NEW: 2-bit pager

training/
├── train_moe_engram.py      # BF16 training
├── train_moe_engram_2bit.py # NEW: 2-bit training
├── init_experts_2bit.py     # NEW: 2-bit expert init

inference/
├── inference.py             # BF16 inference
├── inference_2bit.py        # NEW: 2-bit inference
```

---

## Training Configuration

### Hyperparameters (2-bit)

| Parameter | Value |
|-----------|-------|
| Batch size | 4 |
| Accumulation | 16 (effective batch 64) |
| Max steps | 50,000 |
| Warmup steps | 1,000 |
| Peak LR (trunk) | 3e-4 |
| Peak LR (engram) | 1e-4 |
| Peak LR (experts) | 3e-3 |
| Weight decay | 0.1 (trunk), 0.01 (engram), 0.1 (experts) |
| Gradient clip | 1.0 |
| Precision | BF16 activations, 2-bit CQ weights |
| Optimizer | AdamW (FP32 master weights) |

### Memory Optimizations

```python
# Environment
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Gradient checkpointing
from torch.utils.checkpoint import checkpoint
x = checkpoint(block_forward, block, x, layer_idx, use_reentrant=False)

# Periodic cleanup
if step % 100 == 0:
    torch.cuda.empty_cache()
```

---

## Validation Criteria

| Metric | Target |
|--------|--------|
| Perplexity increase vs BF16 | < 0.5 |
| Generation quality | Comparable to BF16 |
| Peak VRAM usage | < 5.5 GB |
| Training speed | > 0.5 steps/sec |
| No NaN/Inf | 0 occurrences |
| Router load balance | Entropy loss < 0.1 |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| 2-bit quality degradation | Start with 4-bit, validate, then 2-bit |
| STE gradient noise | Gradient clipping 1.0; larger accum |
| Hadamard overhead | Precompute Hadamard matrices; use `torch.matmul` |
| Engram 2-bit collision rate | Monitor hash utilization; increase slots if needed |
| Training instability | Keep trunk at 4-bit; only quantize experts/engram |
| KV cache 2-bit quality | Start with 8-bit, validate, then 2-bit |

---

## Dependencies

```bash
pip install scikit-learn  # For KMeans expert init
# PyTorch 2.5+ with CUDA 12.1
# PyTorch 2.5+ has built-in STE support via autograd.Function
```

---

## Timeline Summary

| Week | Deliverable |
|------|-------------|
| 1 | Core CQ quantization + tests |
| 2 | 2-bit model components (attention, MoE, engram, KV) |
| 3 | Training pipeline + expert init |
| 4 | Training run + validation + ablation |

---

## Branch Strategy

```
main (BF16, stable)
  └── native-2bit-training (this branch)
       ├── Phase 1: Core quantization
       ├── Phase 2: 2-bit components
       ├── Phase 3: Training pipeline
       └── Phase 4: Training run + validation
```

Merge back to `main` only after validation passes.

---

## References

1. **Needle/Cactus Quants**: Ndubuaku et al., "Needle: Automation Foundation Model for Tiny Devices", 2026
2. **CQ Quantization**: `needle/model/quantize.py` (JAX)
3. **Lloyd-Max**: Lloyd, S. (1982). "Least squares quantization in PCM"
4. **Hadamard Rotation**: Standard technique for quantization rotation
5. **STE**: Bengio et al. (2013). "Estimating or Propagating Gradients Through Stochastic Neurons"
6. **StreamingLLM**: Xiao et al. (2023). "Attention Sinks"
6. **KIVI**: Han et al. (2024). "Tuning-Free 2-bit KV Cache Quantization"

---

## Next Steps

1. **Immediate**: Complete `models/cq_quantize.py` validation tests
2. **Day 1-2**: Implement `EngramModule2Bit` and `CQQuantizedAttention`
3. **Day 3-4**: Implement `CQQuantizedMoEBlock` and `Block2Bit`
4. **Day 5**: Integrate into `MoEEngramNextConceptTransformer2Bit`
5. **Week 2**: Training pipeline and expert initialization
6. **Week 3**: Training run on TinyStories
7. **Week 3-4**: Validation, ablation, documentation