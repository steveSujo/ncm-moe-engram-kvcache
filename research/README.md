# Research Notes — NCM-MoE-Engram-KVCache

This directory contains research notes and literature citations for the NCM-MoE-Engram-KVCache project.

## Project Overview

**Goal**: Train a Next-Concept Modeling (NCM) generator on 4GB VRAM by combining four complementary sparsity techniques:
1. Frozen NCM tokenizer (HazyResearch VQ-VAE + residual VQ)
2. MoE paging (64 experts/layer, 32 resident on VRAM)
3. Engram memory (Needle 2 style, orders [2,3], 8192 slots, layers 2 & 11)
4. Bounded KV cache (256 window + 1 sink + INT8 quantization)

## Research Topics

| Topic | File | Description |
|-------|------|-------------|
| Next-Concept Modeling (NCM) | `ncm.md` | HazyResearch NCM architecture, VQ-VAE tokenizer, parallel coarse-to-fine generation |
| Mixture of Experts (MoE) | `moe.md` | MoE architectures, routing, expert paging, capacity factors |
| Engram Memory | `engram.md` | Needle/Needle 2, hashed n-gram lookup, associative memory in transformers |
| KV Cache Optimization | `kv_cache.md` | Sliding window, attention sinks, INT8 quantization |
| Memory-Constrained Training | `memory_training.md` | 4GB VRAM training techniques, gradient accumulation, offloading |

## Citation Format

All citations use `[Author Year]` format with full references in each topic file.

## BibTeX Database

See `citations.bib` for complete bibliography in BibTeX format.