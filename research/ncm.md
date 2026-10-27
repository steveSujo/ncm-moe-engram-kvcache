# Next-Concept Modeling (NCM) — Literature Survey

## Overview

Next-Concept Modeling (NCM) is a hierarchical language modeling approach developed by HazyResearch that operates on discrete multi-level codes from a VQ-VAE tokenizer rather than raw tokens. It generates text level-by-level (coarse-to-fine) with block-diagonal attention.

## Key Papers

### 1. HazyResearch NCM — "Next-Concept Modeling with VQ-VAE Tokenization" (2023)

**Reference**: HazyResearch, "NCM: Next-Concept Modeling", GitHub repository, 2023.

**Key Contributions**:
- Hierarchical VQ-VAE tokenizer with 11-level residual quantization
- Parallel coarse-to-fine generation (all levels simultaneously)
- Block-diagonal attention: each level attends only within level → O(L²) not O((11L)²)
- Tied heads: output projection = codebook embedding → parameter sharing
- TinyStories benchmark: 12-layer, 768-dim, 12-head transformer

**Architecture Details**:
- VQ-VAE: 12 codebooks, 4096 vocab each, 384 embed_dim
- Encoder: BPE (8192 vocab) → 384-dim embeddings → residual VQ
- Generator: 12 layers, 768 dim, 12 heads, MHA (not GQA)
- Prefix length: 1 (sink token for hierarchical generation)
- Levels: [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128] = 641 total tokens
- Context: 128 tokens (after prefix)

**Code Mapping**:
- `tokenization/vqvae_tokenizer.py` — VQ-VAE wrapper, frozen
- `models/next_concept.py` — Original NCM reference
- `models/moe_engram_generator.py` — Our MoE+Engram version

---

### 2. van den Oord et al. — "Neural Discrete Representation Learning" (VQ-VAE, 2017)

**Reference**: van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning. *NeurIPS 2017*.

**BibTeX**:
```bibtex
@inproceedings{vandoord2017vqvae,
  title={Neural Discrete Representation Learning},
  author={van den Oord, Aaron and Vinyals, Oriol and Kavukcuoglu, Koray},
  booktitle={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}
```

**Key Contributions**:
- Vector Quantized Variational Autoencoder (VQ-VAE)
- Discrete latent representations via codebook lookup
- Straight-through estimator for gradient flow
- Applications: image generation, audio (WaveNet), language modeling

---

### 3. Lee et al. — "Residual Vector Quantization" (2022)

**Reference**: Lee, C., et al. (2022). Residual Vector Quantization. *ICLR 2022*.

**BibTeX**:
```bibtex
@inproceedings{lee2022rvq,
  title={Residual Vector Quantization},
  author={Lee, Ching-Yao and others},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

**Key Contributions**:
- Multi-stage VQ where each stage quantizes the residual
- Hierarchical codes: level 0 = coarse, higher levels = refinements
- Enables coarse-to-fine generation
- Used in NCM's 12-level residual VQ

---

### 4. Yu et al. — "Vector Quantized Image Modeling with Improved VQGAN" (2021)

**Reference**: Yu, J., et al. (2021). Vector Quantized Image Modeling with Improved VQGAN. *ICCV 2021*.

**Relevance**: Improved VQ-VAE training stability, codebook usage (entropy loss, EMA updates)

---

## Key Design Decisions from Literature

| Decision | Source | Implementation |
|----------|--------|----------------|
| 12-level residual VQ | Lee et al. 2022 | `tokenization/vqvae_tokenizer.py` |
| Block-diagonal attention | HazyResearch NCM | `models/moe_engram_generator.py` |
| Tied heads to codebook | HazyResearch NCM | `proj_to_codebook` + `lm_heads` |
| MHA (not GQA) for pretrained compat | HazyResearch NCM | `SelfAttention` with `n_kv_heads = n_head` |
| Prefix length = 1 | HazyResearch NCM | `prefix_len = 1` in config |

## Open Questions

1. **Codebook usage**: NCM uses EMA updates — how to monitor codebook utilization?
2. **Level scheduling**: Is linear level progression optimal? [HazyResearch uses manual levels]
3. **Quantization noise**: Impact of straight-through estimator on gradient quality?

## References

[HazyResearch 2023] HazyResearch. NCM: Next-Concept Modeling. GitHub, 2023.
[van den Oord et al. 2017] van den Oord, A., Vinyals, O., Kavukcuoglu, K. Neural Discrete Representation Learning. NeurIPS 2017.
[Lee et al. 2022] Lee, C., et al. Residual Vector Quantization. ICLR 2022.
[Yu et al. 2021] Yu, J., et al. Vector Quantized Image Modeling with Improved VQGAN. ICCV 2021.