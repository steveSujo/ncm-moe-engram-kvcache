# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-XX-XX

### Added
- **Core Architecture**: NCM-MoE-Engram-KVCache model combining four sparsity techniques
- **NCM Tokenizer** (frozen): Pre-trained HazyResearch VQ-VAE + multiscale residual VQ (TinyStories)
- **NCM Generator**: Parallel coarse-to-fine transformer (11 levels, block-diagonal attention, tied heads)
- **MoE Paging** (mini-AGI style): 64 experts/layer on SSD, 32 resident in VRAM, dynamic expert loading
- **Engram Key-Value Memory** (Needle 2 / DeepSeek): Hashed n-gram lookup tables (2-gram + 3-gram) at layers 2 & 15
- **Bounded KV Cache** (Needle 2): 256-token sliding window + pinned attention sinks + INT8 quantization
- **Two-Timescale Optimizer**: Trunk LR 3e-4, Engram tables LR 1e-4, Experts LR 3e-3 (10× trunk)
- **K-means Expert Initialization**: Distillation from dense NCM generator checkpoint

### Components
- `models/engram.py` - EngramModule with FNV-1a hashing, sigmoid-gated attention, depthwise causal Conv1D
- `models/kv_cache.py` - BoundedKVCache with INT8 quantization, sliding window, attention sinks
- `models/moe_block.py` - MoEBlock with Top-2 routing, entropy loss, SwiGLU experts
- `models/common.py` - Base NCM components: Block, SelfAttention (QK-Norm, RoPE), MLP, RMSNorm
- `models/moe_engram_generator.py` - Integrated MoEEngramNextConceptTransformer
- `utils/expert_io.py` - MoEPager + ExpertIndex for SSD-backed expert storage with LRU residency
- `tokenization/vqvae_tokenizer.py` - Frozen VQ-VAE tokenizer wrapper

### Training & Evaluation
- `training/init_experts.py` - K-means distillation from dense NCM checkpoint (12 layers × 64 experts)
- `training/train_moe_engram.py` - Two-timescale training loop with periodic checkpointing
- `training/smoke_tests.py` - 7 comprehensive verification tests
- `training/create_dummy_dataset.py` - Dummy dataset creator

### Documentation
- `README.md` - Complete project documentation with architecture, quick start, config reference
- `QUICKSTART.md` - 5-minute getting started guide
- `ARCHITECTURE.md` - Deep technical documentation
- `Makefile` - Automation targets (install, download, init-experts, test, train, generate)

### Configuration
- Target: ~200M sparse params, ~32M active params/forward
- Fits 4GB VRAM (BF16 weights + INT8 KV/activations)
- 12 layers, 12 heads, 768 dim, 11 levels × 4096 vocab
- Engram: orders [2,3], 4 heads, 8192 slots, 128 sub-dim
- MoE: 64 experts/layer, Top-2 routing, capacity 1.25, 32 resident slots
- KV Cache: 256 window, 1 sink token, INT8 per-channel

---

## Commit Message Convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, etc)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools
- `build`: Changes that affect the build system or external dependencies
- `ci`: Changes to CI configuration files and scripts

### Examples
```
feat(enggram): add sigmoid-gated attention for Engram fusion
fix(kv_cache): resolve dtype mismatch in attention computation
docs(readme): update install instructions to use requirements.txt
refactor(moe): simplify expert loading logic
test(smoke): add training step test (skipped on CPU)
```