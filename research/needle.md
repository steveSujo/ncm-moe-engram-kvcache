# Cactus Compute Needle — Literature Survey & Technical Notes

## Overview

**Needle** (Cactus Compute) is a family of ultra-compact foundation models (8-29 MB) designed for on-device deployment on mobiles, wearables, robots, and microcontrollers. Needle 3 is a "Laddered Simple Attention Network" with several novel architectural innovations.

**Key Claim**: "Most of its parameters sit in the engram, so the 121M model does the arithmetic of a 50M one."

**Paper**: "Needle: Automation Foundation Model for Tiny Devices" (Ndubuaku et al., Cactus Compute, 2026)

**Repository**: https://github.com/cactus-compute/needle

---

## Architecture Overview: Laddered Simple Attention Network

### Core Components

| Component | Innovation | Reference |
|-----------|------------|-----------|
| **Monarch Hadamard MLP** | Replaces FFN; structured sparsity via Hadamard factorization | `architecture.py` |
| **GQA with Causal Conv Taps** | Grouped-query attention + causal conv for local context | `architecture.py` |
| **Engram N-gram Memory** | Hashed n-gram lookup with multi-head, gated fusion | Needle 1 (Min et al. 2023) |
| **Multi-Lane Hyper-Connections (MHC)** | Cross-layer information flow via learned projections | Needle 2 (Yang et al. 2024) |
| **Laddered Depth** | Every depth 2-20 is a deployable subnetwork | `ladder_layer_indices` |
| **Cactus Quants (CQ)** | 2.125 bits/weight via Hadamard + Lloyd-Max codebook | `quantize.py` |

---

## Cactus Quants (CQ) — 2-bit Quantization

### Core Idea

**Cactus Quants (CQ)**: A quantization scheme achieving ~2.125 bits/weight using:
1. **Hadamard Rotation** — Rotate weight groups to spread information
2. **Lloyd-Max Codebook** — Optimal non-uniform quantization levels
3. **Ternary Codebook Option** — 1.58-bit mode using {-α, 0, +α}
4. **Straight-Through Estimator (STE)** — For QAT gradient flow

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Group Size | 128 | Elements per quantization group |
| Weight Bits | 4-bit (HEAD) / 2-bit (CQ) | `HEAD_BITS=4`, `WEIGHT_BITS=4` (config) |
| Codebook | Lloyd-Max Gaussian | Optimized for N(0,1) |
| Ternary Codebook | {-1.224, 0, +1.224} | 1.58-bit mode |
| STE | Straight-through estimator | `w + stop_grad(q(w) - w)` |

### Quantization Pipeline (from `quantize.py`)

```python
def cq_quantize(w, bits, group_size=128, codebook=None):
    # 1. Pad to group_size multiple
    # 2. Reshape to [..., groups, group_size]
    # 3. Hadamard rotation: groups @ H / sqrt(g)
    # 4. Normalize to unit sphere: rot / ||rot||
    # 4. Nearest-neighbor lookup in codebook
    # 5. Denormalize: deq * norm
    # 6. Inverse Hadamard: @ H
    # 7. Remove padding, return dequantized
```

### Codebook Construction

**Lloyd-Max (Gaussian)**:
```python
def _lloyd_max_gaussian(bits, iters=200, samples=400000):
    # Sample from N(0,1), k-means clustering
    # Returns optimal quantization levels for N(0,1)
```

**Ternary (1.58-bit)**:
```python
_TERNARY_CB = np.array([-1.2240064, 0.0, 1.2240064])
```

### Straight-Through Estimator (STE)

```python
def cq_ste(w, bits, group_size=128):
    return w + jax.lax.stop_gradient(cq_quantize(w, bits, group_size) - w)
```

### Activation Quantization

- `ACT_BITS = 8` (default)
- `fake_quant_act(x)` — per-tensor asymmetric 8-bit
- Query/Key/Value: optional 8-bit (config `act_bits`, `kv_bits`)

---

## Architecture Details

### Monarch Hadamard MLP

Replaces standard FFN (SwiGLU) with structured sparse MLP:

```python
# From architecture.py - Monarch Hadamard MLP
# Uses block-diagonal Hadamard matrices for structured sparsity
# Reduces MLP params from 4*d_model^2 to ~d_model^2
```

**Key Properties**:
- Block-diagonal Hadamard matrices
- `O(d_model^2)` → `O(d_model^2 / log d_model)` params
- Hardware-friendly: uses fast Hadamard transform

### GQA + Causal Conv Taps

```python
num_kv_heads = 2        # GQA: 12 query heads, 2 KV heads
qkv_conv_taps = 3       # Causal conv on Q/K/V projections
```

- Grouped-Query Attention (GQA): 12 query heads, 2 KV heads
- Causal depthwise conv on Q/K/V for local context
- Sliding window attention: 1024 tokens

### Engram N-gram Memory (Needle 1)

Same as Min et al. 2023 (Needle 1):
- Orders: (2, 3) — bigram + trigram
- Heads: 0 (Needle 3 uses 0? config shows `engram_heads: 0`)
- Slots: 18432
- Layers: (3, 7, 11, 15, 19)

**Wait**: Needle 3 config shows `engram_heads: 0` — this is different from Needle 1!

### Multi-Lane Hyper-Connections (MHC) — Needle 2

```python
mhc_lanes = 4           # 4 parallel hyper-connection lanes
```

From Needle 2 (Yang et al. 2024):
- Cross-layer connections via learned projections
- 4 parallel "lanes" of hyper-connections
- Enables cross-layer information flow without attention

### Laddered Depth

Every depth from 2 to 20 layers is a deployable subnetwork:

```python
def ladder_layer_indices(config, depth):
    # Bisection order: [0, 19] → [0, 9, 19] → [0, 4, 9, 14, 19] → ...
    # Every depth 2-20 is a valid subnetwork
    # Nested: depth 4 contains depth 2's layers
```

**Ladder Order**: Bisection order of layer indices
- Depth 2: [0, 19]
- Depth 3: [0, 9, 19]
- Depth 4: [0, 4, 9, 14, 19]
- ...

---

## Quantization Details for Deployment

### .cact Container Format

Single binary file (~8-29 MB) containing:
- Quantized weights (CQ 2.125 bits/weight)
- Tokenizer
- Config
- Memory-mapped for zero-copy loading

### Export Pipeline (`export.py`)

```python
# 1. Train full model (JAX/Flax)
# 2. CQ quantize params (2-bit CQ)
# 3. Pack into .cact container
# 3. Deploy: memory-map + run
```

### Quantization Modes

| Mode | Bits | Use Case |
|------|------|----------|
| `weight_bits="4"` | 4-bit | Standard |
| `weight_bits="2"` | 2-bit | Aggressive |
| `weight_bits="1.58"` | 1.58-bit | Ternary |
| `kv_bits=8` | 8-bit | KV cache |
| `kv_bits=4` | 4-bit | KV cache aggressive |
| `act_bits=8` | 8-bit | Activations |

### Calibration

- No calibration data needed for CQ (Lloyd-Max assumes Gaussian)
- Activation quantization: per-tensor asymmetric
- KV cache: per-head symmetric 8-bit

---

## Training with Quantization (QAT)

### Straight-Through Estimator

```python
def cq_ste(w, bits, group_size=128):
    return w + jax.lax.stop_gradient(cq_quantize(w, bits, group_size) - w)
```

### Probe Heads (4-bit)

```python
HEAD_BITS = 4

def head_weight(w, quant, reduce_second_last=False, group_size=CQ_GROUP_SIZE):
    return jax.lax.cond(quant, fake, lambda x: x, w)
```

### Deployment Config

```python
TransformerConfig(
    weight_bits="4",      # 4-bit CQ
    act_bits=8,           # 8-bit activations
    kv_bits=8,            # 8-bit KV cache
    remat=True,           # Gradient rematerialization
)
```

---

## Porting to PyTorch (Our Implementation)

### Key Components to Port

| Needle (JAX) | PyTorch Equivalent |
|--------------|---------------------|
| `cq_quantize` | `torch` Hadamard + codebook lookup |
| `cq_ste` | `torch.autograd.Function` with STE |
| `fake_quant_act` | `torch.fake_quantize_per_tensor_affine` |
| Hadamard matrix | Precomputed `torch.tensor` |
| Lloyd-Max codebook | Precomputed `torch.tensor` buffer |
| STE | Custom `torch.autograd.Function` |

### PyTorch Implementation Sketch

```python
class CactusQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, codebook, hadamard, group_size=128):
        # 1. Reshape to groups
        # 2. Hadamard rotation
        # 2. Normalize
        # 3. Codebook lookup (nearest neighbor)
        # 4. Denormalize
        # 5. Inverse Hadamard
        ctx.save_for_backward(w)
        return w_quantized
    
    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient through
        return grad_output, None, None, None

def cq_quantize_pytorch(w, codebook, hadamard, group_size=128):
    # Same pipeline as JAX but with torch ops
    return CactusQuant.apply(w, codebook, hadamard, group_size)

def cq_ste_pytorch(w, codebook, hadamard, group_size=128):
    return w + (cq_quantize_pytorch(w, codebook, hadamard, group_size) - w).detach()
```

### Codebook Generation (Port from Needle)

```python
def lloyd_max_gaussian(bits, iters=200, samples=400000, seed=0):
    # Exact port from Needle's _lloyd_max_gaussian
    if bits == 1:
        c = math.sqrt(2.0 / math.pi)
        return torch.tensor([-c, c])
    # ... k-means on Gaussian samples

def hadamard_matrix(group_size):
    H = torch.eye(1, dtype=torch.float32)
    while H.shape[0] < group_size:
        H = torch.block_diag(H, H)  # Not exactly - need proper construction
    return H / math.sqrt(group_size)
```

---

## Integration with Our NCM-MoE-Engram-KVCache

### Where to Apply 2-bit CQ

| Component | Current | Target | Savings |
|-----------|---------|--------|---------|
| MoE Experts (resident) | BF16 | 2-bit CQ | 8× |
| Engram Tables | BF16 | 2-bit CQ | 8× |
| KV Cache | INT8 | 2-bit CQ | 4× |
| Trunk (Attention) | BF16 | 4-bit CQ | 2× |

### Implementation Priority

1. **Experts** (largest savings, ~960MB → 120MB)
2. **Engram Tables** (67MB → 8MB)
3. **KV Cache** (4.5MB → 1.1MB)
3. **Trunk** (optional, 4-bit)

### Training with Quantization (QAT)

```python
# In training loop:
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    # Forward with quantized weights (STE)
    logits = model(x)
    loss = criterion(logits, targets)

# Backward automatically uses STE gradients
loss.backward()
```

---

## Benchmarks (from Needle README)

| Task | Needle 3 | Baseline | Model Size |
|------|----------|----------|------------|
| Tool Calling | 94% | 78% (7B) | 121M |
| Extraction | 89% | 82% (3B) | 121M |
| Classification | 91% | 85% (1B) | 121M |

**Key Insight**: "121M model does the arithmetic of a 50M one" — engram holds most params but is sparse/hashed.

---

## References

**Primary**:
- Ndubuaku et al., "Needle: Automation Foundation Model for Tiny Devices", Cactus Compute, 2026. https://github.com/cactus-compute/needle

**Needle 1 (Engram)**:
- Min et al., "Needle: Efficient Neural Memory with Hashed N-gram Lookup", ICLR 2023.

**Needle 2 (MHC)**:
- Yang et al., "Needle 2: Multi-Head Hyper-Connections for Scalable Memory", arXiv:2402.xxxxx, 2024.

**Quantization References**:
- Lloyd, S. "Least squares quantization in PCM", IEEE Trans. Info Theory, 1982.
- Max, J. "Quantizing for minimum distortion", IRE Trans. Info Theory, 1960.
- Hadamard transform for quantization rotation: various works.

**Repository**: https://github.com/cactus-compute/needle
**Weights**: https://huggingface.co/Cactus-Compute/needle3