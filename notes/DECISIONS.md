# Key Architectural Decisions — NCM-MoE-Engram-KVCache

This document records major design decisions, trade-offs, and rationale for future LLM agents working on this codebase.

---

## 1. Overall Architecture: Hybrid Sparsity Stack

**Decision**: Combine 5 sparsity techniques in a single model:
1. Frozen NCM tokenizer (VQ-VAE + residual VQ)
2. NCM generator (parallel coarse-to-fine, block-diagonal attention)
3. MoE paging (mini-AGI style: 64 experts/layer on SSD, 32 resident in VRAM)
4. Engram memory (Needle 2 / DeepSeek: hashed n-gram lookup at layers 2 & 11)
5. Bounded KV cache (256-token window + pinned sink + INT8)

**Why**: Single technique insufficient for 4GB VRAM budget. Each addresses different bottleneck:
- MoE paging → FFN parameter explosion (largest memory consumer)
- Engram → static knowledge reconstruction in early/mid layers
- Bounded KV cache → O(T²) attention memory during generation
- NCM tokenizer → discrete hierarchical codes (semantic compression)
- NCM generator → tied heads, block-diagonal attention (structure exploitation)

**Alternative considered**: Train from scratch with single technique → rejected (insufficient capacity, no pretrained init)

---

## 2. Frozen NCM Tokenizer (Not Trained)

**Decision**: Use HazyResearch's pre-trained TinyStories VQ-VAE + multiscale residual VQ tokenizer. All params `requires_grad=False`.

**Why**:
- Training VQ-VAE from scratch requires massive compute + unstable convergence
- HazyResearch checkpoint provides 11-level hierarchy with 4096 vocab/level
- Tokenizer is the foundation; errors compound through generator
- TinyStories domain matches target (simple narratives)

**Trade-off**: Cannot adapt tokenizer to new domains. Acceptable for TinyStories scope.

**Location**: `tokenization/vqvae_tokenizer.py` → `VQVAETokenizer.from_config()`

---

## 3. NCM Generator: Parallel Coarse-to-Fine with Tied Heads

**Decision**: Use NCM's parallel level generation (all 11 levels simultaneously) with block-diagonal attention and heads tied to codebook embeddings.

**Why**:
- Original NCM design — proven on TinyStories
- Block-diagonal attention: each level attends only within level → O(L²) not O((11L)²)
- Tied heads: output projection = codebook embedding → parameter sharing, semantic consistency
- 12 layers, 12 heads, 768 dim → fits VRAM when combined with MoE

**Critical**: Must use MHA (not GQA) to match pretrained NCM weights. `n_kv_heads = n_head = 12`.

**Location**: `models/next_concept.py` → `NextConceptTransformer`, `generate()`

---

## 4. MoE Paging: 64 Experts/Layer, 32 Resident, SSD-Backed

**Decision**: Replace each layer's MLP with MoE (SwiGLU experts, Top-2 routing, capacity 1.25). All 768 experts stored as safetensors on SSD; LRU pool of 32 experts in VRAM.

**Why**:
- Dense FFN: 3072 hidden × 768 × 3 (gate/up/down) × 12 layers ≈ 85M params → ~170 MB BF16 just for FFN
- With MoE: 64 experts × 3072 hidden → same total params but only 2 active/token
- Active params: 2 × 3072 × 768 × 3 × 12 / 64 ≈ 2.7M active/forward
- SSD storage: 768 experts × ~14 MB = ~11 GB (cheap NVMe)
- VRAM: 32 resident × 14 MB = ~450 MB BF16

**Routing**: Top-2 with entropy loss (0.01×) for load balancing. Router always resident (Linear 768→64, ~0.2 MB).

**Expert init**: K-means distillation from dense NCM generator checkpoint (concatenate fc + proj^T, cluster 3072 hidden units into 64 experts).

**Location**: `models/moe_block.py` → `MoEBlock`, `Top2Router`; `utils/expert_io.py` → `MoEPager`, `ExpertIndex`; `training/init_experts.py`

**Trade-offs**:
- SSD I/O latency during routing → mitigated by LRU + async prefetch
- Non-deterministic expert assignment → entropy loss stabilizes
- Windows mmap limitations → use `safetensors` with explicit load

---

## 5. Engram Memory: Needle 2 Style Hashed N-gram Lookup

**Decision**: Insert Engram modules at layers 2 and 11 (0-indexed). Each: multi-head FNV-1a hashed n-gram lookup (orders [2,3], 4 heads, 8192 slots, 128 sub_dim), sigmoid-gated attention, depthwise causal Conv1D (kernel=4, dilation=3).

**Why**:
- Early layer (2): captures local n-gram statistics (syntax, common phrases)
- Mid layer (11): supports reasoning with longer context patterns
- Hashed lookup: O(1) deterministic, no learned indices → can prefetch from host RAM
- Gating: `sigmoid(RMSNorm(h)^T RMSNorm(k) / sqrt(d))` → learns when to use memory vs. attention
- Conv1D: smoothes retrieved values across time (causal, dilation=max(order))

**Params**: 2 sites × 2 orders × 4 heads × 8192 slots × 128 dim = 33.6M params (BF16: ~67 MB VRAM)

**NOT implemented**: Needle 2's MHC (multi-head hyper-connections) and HadamardMLP — use standard residual + SwiGLU MoE instead.

**Location**: `models/engram.py` → `EngramModule`, `engram_indices`, `engram_geometry`, `needle2_engram_config`

---

## 6. Bounded KV Cache: 256-Token Sliding Window + Pinned Sink + INT8

**Decision**: Custom KV cache with 256-token rolling window, 1 pinned sink token (NCM prefix_len=1), INT8 per-channel quantization.

**Why**:
- Full KV cache: 12 layers × 12 heads × 128 tokens × 64 dim × 2 (K+V) × 2 BF16 bytes = ~2.4 MB per step → grows O(T)
- Generation up to 11 levels × 128 tokens = 1408 tokens → would OOM
- Sliding window: bounds memory to constant
- Sink token: preserves attention to BOS/prefix (critical for NCM hierarchical generation)
- INT8: 2× compression for KV storage; per-channel scale preserves quality

**INT8 quantization**: `quantize_int8(x) → (x_q, scale)`, `dequantize_int8(x_q, scale) → x_dq`. Applied per-head per-channel.

**Causal mask**: Dynamically constructed for sink + window length.

**Location**: `models/kv_cache.py` → `BoundedKVCache`, `quantize_int8`, `dequantize_int8`

---

## 7. Two-Timescale Optimizer

**Decision**: Three parameter groups with different LRs:
- Trunk (attention, embeddings, heads, engram projections, routers): LR 3e-4, WD 0.1
- Engram tables: LR 1e-4, WD 0.01 (slower — static memory should be stable)
- MoE experts: LR 3e-3, WD 0.1 (10× trunk — sparse, need faster adaptation)

**Why**:
- MoE experts see fewer tokens per step (Top-2) → need higher LR to learn
- Engram tables are large embedding tables → slow LR prevents catastrophic forgetting
- Trunk is dense, sees every token → standard LR

**Implementation**: `get_optimizer()` in `training/train_moe_engram.py` groups by parameter name patterns.

---

## 8. Training Loop: Gradient Accumulation + Periodic Checkpointing

**Decision**: Batch size 4, gradient accumulation 8 (effective batch 32). Checkpoints every 1000 steps. Resident experts checkpointed every 1000 steps.

**Why**:
- 4GB VRAM limits batch size; accumulation simulates larger batch
- 50k steps target → ~15 epochs on TinyStories
- Expert checkpointing: SSD experts updated during training; save resident pool periodically
- Mixed precision BF16 + GradScaler

**Loss**: Cross-entropy per level + 0.01 × router entropy loss.

**Location**: `training/train_moe_engram.py` → `train_step()`, `save_checkpoint()`, `main()`

---

## 9. Expert Initialization: K-means Distillation

**Decision**: Initialize MoE experts by k-means clustering on dense NCM generator's MLP weights (concatenated fc + proj^T), not random.

**Why**:
- Random init → router chaos, experts unspecialized, training instability
- K-means on dense weights → experts start as meaningful partitions of FFN subspace
- Cluster centroids → expert base weights; add noise for diversity
- Done once offline (CPU, ~5 min/layer); saved to SSD

**Process**:
1. Load `generator.pt` from HazyResearch checkpoint
2. For each layer: extract `c_fc.weight` [3072, 768], `c_proj.weight` [768, 3072]
3. Concatenate → [3072, 1536]
4. KMeans(64) on 3072 points → 64 centroids [64, 1536]
5. Split → fc_centroids [64, 768], proj_centroids [768, 64]
6. Expand to expert_hidden (3072) with noise
7. Save as safetensors per expert

**Location**: `training/init_experts.py` → `init_experts_from_dense()`

---

## 10. Logging & Resumability

**Decision**: All long-running scripts use Python `logging` module with file + console handlers. Progress logged every N iterations (16 experts, 500 steps). Skip completed layers on restart.

**Why**:
- Expert init takes ~5 min/layer × 12 = 60 min on CPU; must survive interruptions
- Training runs hours; need visibility without stdout polling
- Log files: `expert_init.log`, `training.log`
- Skip logic: check `expert_index.json` + file existence before processing layer

**Location**: `training/init_experts.py`, `training/train_moe_engram.py` — both have logging setup at module level.

---

## 11. Disk Space Guard Clause

**Decision**: Minimum 10GB free on C: (uv cache), 15GB free on expert/checkpoint drive. LLM agents MUST NOT run cleanup commands.

**Why**:
- Expert storage ~11 GB, checkpoints ~2 GB, artifacts ~578 MB
- uv/pip cache can consume 5-10 GB during installs
- Windows C: drive often constrained
- LLM cleanup attempts often delete wrong things

**Enforcement**: `.omp/RULES.md` has mandatory pre-flight check.

---

## 12. Testing Strategy

**Decision**: Unit tests for each component + integration smoke tests. Run on CPU for CI, GPU for full training.

**Tests**:
- `training/smoke_tests.py` — 7 tests: Engram, KV cache, MoE pager, Top-2 router, MoE block, integrated model, training step (skipped on CPU)
- `training/test_init_experts.py` — k-means shapes, pager save/load, expert index, full init single layer
- `training/test_train.py` — dataset, optimizer groups, scheduler, train_step, checkpoint save

**Why**: Isolate failures to specific components. Smoke tests verify wiring before GPU time.

---

## 13. File Structure Conventions

```
ncm_model/
├── artifacts/           # Downloaded checkpoints (not in git)
├── checkpoints/         # Training checkpoints (not in git)
├── data/                # Datasets (not in git)
├── experts/             # MoE expert weights (not in git, ~11 GB)
├── models/              # Core architecture
│   ├── common.py        # Base NCM: Block, SelfAttention, MLP, RMSNorm
│   ├── engram.py        # EngramModule, hashed n-gram lookup
│   ├── kv_cache.py      # BoundedKVCache, INT8 quantization
│   ├── moe_block.py     # MoEBlock, Top2Router, MoEBlockNoPager
│   ├── moe_engram_generator.py  # Integrated MoEEngramNextConceptTransformer
│   └── next_concept.py  # Original NCM (reference)
├── tokenization/        # Frozen tokenizer wrapper
├── training/            # Training scripts + tests
├── utils/               # Expert I/O, paging
├── .omp/RULES.md        # LLM agent rules
└── notes/               # This file
```

---

## 14. Known Limitations / Deferred Work

1. **Sub-3-bit Cactus Quants**: Register-level expansion for inference — deferred
2. **Grammar-guided FSM cache skipping**: For deterministic token sequences — deferred
3. **Needle 2 MHC / HadamardMLP**: Not implemented — use standard residual
4. **Real dataset**: `hazyresearch/ncm-datasets` is gated (needs auth); using dummy for now
5. **Distributed MoE**: Single GPU only; expert parallelism not implemented
6. **Windows mmap**: `safetensors` load may not mmap on Windows; explicit load used

---

## 15. Quick Reference: Key Files to Modify

| Change | Files to Edit |
|--------|---------------|
| Engram layers/positions | `models/moe_engram_generator.py` (line ~80: `engram_layers = [2, 11]`) |
| MoE config (experts, top-k) | `models/moe_block.py`, `models/moe_engram_generator.py` |
| KV cache window/sink | `models/kv_cache.py` (`BoundedKVCache.__init__`) |
| Optimizer LRs | `training/train_moe_engram.py` (`get_optimizer`) |
| Expert init (k, hidden) | `training/init_experts.py` |
| Training steps/batch | `training/train_moe_engram.py` (`main`) |
| Logging frequency | Both training scripts (logger.info calls) |

---

## 16. Commit History Reference

- `433a580` — feat: initial commit (all core components)
- `8ade65d` — feat(training): add logging to init_experts and train_moe_engram scripts

---

*Generated for future LLM agents. Keep updated as architecture evolves.*