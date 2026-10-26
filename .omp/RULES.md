# NCM-MoE-Engram-KVCache Project Rules (for LLM agents)

## Git & Commits
- Conventional Commits format: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `build`, `ci`
- Commit after logical units of work
- Push to origin/main after commits

## Code Style
- Python 3.10+, type hints required
- Black formatting (line-length 100)
- Ruff linting (select: E,F,I,W,UP,B,C4,SIM,T20)
- No dead code, no unused imports

## Architecture
- 4GB VRAM budget (BF16 weights + INT8 KV/activations)
- Frozen NCM tokenizer (HazyResearch VQ-VAE, TinyStories)
- 12-layer generator, 11 levels, 4096 vocab/level
- MoE: 64 experts/layer, Top-2, 32 resident slots, SSD paging
- Engram: Needle 2 style, orders [2,3], 8192 slots, 4 heads, layers 2 & 11
- KV Cache: 256-token sliding window + 1 sink token, INT8
- Two-timescale optimizer: trunk 3e-4, engram 1e-4, experts 3e-3

## Training
- Batch 4, accum 8, 50k steps target
- Checkpoints every 1000 steps
- Expert checkpoints every 1000 steps
- Mixed precision BF16, gradient clipping 1.0

## Logging
- All long-running scripts write to log files (`training.log`, `expert_init.log`)
- Use `logging` module with timestamps, file + console handlers
- Progress logs every N iterations (16 experts, 500 steps)
- Resume from checkpoint/logs on restart

## Dependencies
- uv for package management (`uv pip install -r requirements.txt`)
- PyTorch 2.5+ CUDA 12.1
- No system pip

## Disk Space — GUARD CLAUSE (MANDATORY PRE-FLIGHT)
BEFORE any install/build/download/training operation:
1. CHECK free space on target drive (C: for uv cache, A: for experts/checkpoints)
2. REQUIRE minimum 10GB free on C: drive
3. REQUIRE minimum 15GB free on expert/checkpoint drive
4. If FAIL → STOP, report to user, **DO NOT ATTEMPT TO FREE SPACE YOURSELF**
5. **LLM AGENTS: NEVER run cleanup commands (rm, del, Remove-Item, etc.) to free space**
6. **LLM AGENTS: Only report the error and wait for user action**

Check command (Windows):
```powershell
Get-PSDrive C | Select-Object Used,Free
Get-PSDrive A | Select-Object Used,Free
```

## Files Not in Repo
- `artifacts/` (downloaded checkpoints)
- `experts/` (generated expert weights)
- `checkpoints/` (training checkpoints)
- `data/` (datasets)
- `.venv/` (virtual env)
- `*.log` (log files)