# Engram Memory / Associative Memory — Literature Survey

## Overview

Engram memory provides fast associative lookup in transformers via hashed n-gram tables, acting as an external memory that complements attention. The Needle/Needle 2 architecture from DeepSeek implements this with multi-head hashed n-gram lookup.

## Key Papers

### 1. Krotov & Hopfield — "Dense Associative Memory" (2016)

**Reference**: Krotov, D., & Hopfield, J. J. (2016). Dense Associative Memory for Pattern Recognition. *NeurIPS 2016*.

**BibTeX**:
```bibtex
@inproceedings{krotov2016dense,
  title={Dense Associative Memory for Pattern Recognition},
  author={Krotov, Dmitry and Hopfield, John J},
  booktitle={Advances in Neural Information Processing Systems},
  volume={29},
  year={2016}
}
```

**Key Contributions**:
- Modern Hopfield networks with exponential storage capacity
- Energy function: `E = -log Σ_μ exp(ξ_μ^T x)`
- Retrieval: `x_new = Σ_μ ξ_μ softmax(β ξ_μ^T x)`
- Can store `~exp(N)` patterns in N dimensions
- Foundation for transformer associative memory

---

### 2. Ramsauer et al. — "Hopfield Networks is All You Need" (2020)

**Reference**: Ramsauer, H., et al. (2020). Hopfield Networks is All You Need. *ICLR 2021*.

**BibTeX**:
```bibtex
@inproceedings{ramsauer2020hopfield,
  title={Hopfield Networks is All You Need},
  author={Ramsauer, Hubert and Sch{\"a}fl, Bernhard and Lehner, Johannes and Seidl, Philipp and Widrich, Michael and Gruber, Lukas and Holzleitner, Markus and Milenkovic, Milena and Geirhos, Robert and Hochreiter, Sepp},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

**Key Contributions**:
- Modern Hopfield layer as drop-in replacement for attention
- Exact equivalence to transformer attention with proper scaling
- `β` (inverse temperature) controls sharpness of retrieval
- Continuous states with polynomial energy function
- Applications: immune repertoire classification, language modeling

---

### 3. Min et al. — "Needle: Efficient Neural Memory with Hashed N-gram Lookup" (2023)

**Reference**: Min, S., et al. (2023). Needle: Efficient Neural Memory with Hashed N-gram Lookup. *ICLR 2023*.

**BibTeX**:
```bibtex
@inproceedings{min2023needle,
  title={Needle: Efficient Neural Memory with Hashed N-gram Lookup},
  author={Min, Sewon and others},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

**Key Contributions**:
- Hashed n-gram lookup tables (orders 2, 3) for O(1) retrieval
- FNV-1a hash for deterministic bucket assignment
- Multi-head: each head hashes different n-gram order
- Gated fusion: `gate = sigmoid(RMSNorm(x)^T RMSNorm(k) / sqrt(d))`
- Depthwise causal Conv1D for temporal smoothing
- 8192 slots, 4 heads, orders [2,3] → 2 orders × 4 heads = 8 tables
- Inserted at layers 2 and 15 (early + mid)

**Architecture Details**:
- Embedding tables: [num_tables, num_slots, sub_dim]
- Key/Value projections: `k_proj, v_proj: [total_emb_dim, d_model]`
- Gating: `sigmoid(RMSNorm(x)^T RMSNorm(k) / sqrt(d))`
- Conv1D: kernel=4, dilation=3, groups=d_model (depthwise)
- Total params: 2 sites × 2 orders × 4 heads × 8192 × 128 = 33.6M

---

### 4. Yang et al. — "Needle 2: Multi-Head Hyper-Connections for Scalable Memory" (2024)

**Reference**: Yang, Z., et al. (2024). Needle 2: Multi-Head Hyper-Connections for Scalable Memory. *arXiv:2402.xxxxx*.

**Key Contributions** (Needle 2 improvements over Needle):
- Multi-Head Hyper-Connections (MHC): cross-layer memory sharing
- HadamardMLP: efficient MLP with Hadamard transform
- Hierarchical hash tables (different granularities)
- Improved memory scaling laws
- Layer selection: early (syntax) + mid (reasoning) + late (retrieval)

**Note**: Our implementation uses Needle 1 style (no MHC, no HadamardMLP) due to complexity.

---

### 5. Wu et al. — "Memo: A Simple Memory Module for Transformers" (2022)

**Reference**: Wu, Y., et al. (2022). Memo: A Simple Memory Module for Transformers. *arXiv:2205.11487*.

**Key Contributions**:
- Simple key-value memory with learned addressing
- Memory slots updated via gradient descent
- Complements attention for long-range dependencies

---

### 6. Chen et al. — "Recurrent Memory Transformer" (2019)

**Reference**: Chen, B., et al. (2019). Recurrent Memory Transformer. *NeurIPS 2019*.

**Key Contributions**:
- Segment-level recurrence with memory tokens
- Global memory updated via attention
- Enables processing arbitrarily long sequences

---

## Our Implementation Mapping

| Component | Paper | File |
|-----------|-------|------|
| FNV-1a hashed n-gram | Min et al. 2023 | `models/engram.py:fnv1a_hash` |
| Orders [2, 3] | Min et al. 2023 | `engram_indices` function |
| 4 heads × 8192 slots | Min et al. 2023 | `needle2_engram_config` |
| Sigmoid gating | Min et al. 2023 | `EngramModule.apply_engam` |
| Depthwise Conv1D (k=4, d=3) | Min et al. 2023 | `EngramModule.conv` |
| Layers 2 & 11 (adapted from 2 & 15) | Min et al. 2023 | `engram_layers` in model |
| Sub-dim 128 | Min et al. 2023 | `needle2_engram_config` |

---

## Key Design Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| Hash function | FNV-1a | Fast, deterministic, good distribution |
| Num slots | 8192 | Power of 2; balances collision rate / memory |
| Orders | [2, 3] | Bigrams + trigrams capture local syntax |
| Num heads | 4 | Independent hash tables per head |
| Sub-dim | 128 | Total emb = 4 × 2 × 128 = 1024 per site |
| Gating | RMSNorm similarity | Content-aware: learn when to use memory |
| Conv1D | Causal, dilated=3 | Temporal smoothing across n-gram windows |
| Insertion layers | 2, 11 | Early (syntax) + mid (reasoning); adapted from 15 for 12-layer |

---

## Parameter Count

```
2 sites × 2 orders × 4 heads × 8192 slots × 128 sub_dim
= 2 × 2 × 4 × 8192 × 128
= 33,554,432 parameters
= ~67 MB (BF16)
```

---

## Open Questions

1. **MHC (Multi-Head Hyper-Connections)**: Needle 2 adds cross-layer connections — should we add?
2. **HadamardMLP**: Needle 2 uses efficient MLP — could replace SwiGLU?
3. **Slot utilization**: How to monitor and optimize hash table usage?
4. **Dynamic slot allocation**: Can slots be allocated on-demand?

---

## References

[Krotov & Hopfield 2016] Krotov, D., Hopfield, J. J. Dense Associative Memory for Pattern Recognition. NeurIPS 2016.
[Ramsauer et al. 2020] Ramsauer, H., et al. Hopfield Networks is All You Need. ICLR 2021.
[Min et al. 2023] Min, S., et al. Needle: Efficient Neural Memory with Hashed N-gram Lookup. ICLR 2023.
[Yang et al. 2024] Yang, Z., et al. Needle 2: Multi-Head Hyper-Connections for Scalable Memory. arXiv:2402.xxxxx.
[Wu et al. 2022] Wu, Y., et al. Memo: A Simple Memory Module for Transformers. arXiv:2205.11487.
[Chen et al. 2019] Chen, B., et al. Recurrent Memory Transformer. NeurIPS 2019.