# psudo-Engine Quickstart

Get the psudo-Engine running in 5 minutes.

## 1. Environment Setup (30s)

```bash
# Clone and enter
git clone https://github.com/steveSujo/ncm-moe-engram-kvcache.git
cd ncm_model

# Switch to the psudo-engine branch
git checkout psudo-engine-rewrite

# Install dependencies
uv venv .venv
uv pip install -r requirements.txt

# Optional: for synthetic data generation
uv pip install numpy
```

## 2. Quick Test (CPU)

```bash
# Run a generation demo (uses random weights - output will be garbage)
PYTHONPATH=. .venv/Scripts/python.exe -c "
from psudo_engine.inference.cli import main
import sys
sys.argv = ['cli.py', '--prompt', 'fn main() {', '--max-tokens', '20']
main()
"
```

**Expected**: Generates garbage text (model is randomly initialized).

## 3. Run Tests

```bash
# All tests (13 integration tests)
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/ -v

# Integration tests only (13 tests)
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/test_integration.py -v

# Unit tests
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/test_weight_paging.py -v
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/test_context.py -v
PYTHONPATH=. .venv/Scripts/python.exe -m pytest psudo_engine/tests/test_model.py -v
```

**Expected**: 10/13 integration tests pass (3 expected failures without trained model)

## 3. Run Benchmarks

```bash
PYTHONPATH=. .venv/Scripts/python.exe psudo_engine/benchmarks/benchmark.py
```

Expected output:
```
small (256d, 2L): 21 MB VRAM, 47 tok/s, 21ms/token
medium (512d, 12L): 286 MB VRAM, 11 tok/s, 89ms/token
```

## 4. CLI Demo

```bash
# Generate
PYTHONPATH=. .venv/Scripts/python.exe psudo_engine/inference/cli.py generate --prompt "fn hello() -> string {" --checkpoint PATH_TO_CHECKPOINT

# Chat mode (interactive)
PYTHONPATH=. .venv/Scripts/python.exe psudo_engine/inference/cli.py chat --checkpoint PATH_TO_CHECKPOINT

# Model info
PYTHONPATH=. .venv/Scripts/python.exe psudo_engine/inference/cli.py info --config psudo_engine/config/model.yaml --checkpoint PATH_TO_CHECKPOINT

# Tokenize
PYTHONPATH=. .venv/Scripts/python.exe psudo_engine/inference/cli.py tokenize --text "fn test() { return 42 }" --encode
```

## Training (Requires Corpus)

```bash
# Prepare data (generates synthetic data if no corpus)
PYTHONPATH=. .venv/Scripts/python.exe -m psudo_engine.training.train --prepare-data

# Train
PYTHONPATH=. .venv/Scripts/python.exe -m psudo_engine.training.train

# Resume from checkpoint
PYTHONPATH=. .venv/Scripts/python.exe -m psudo_engine.training.train --resume checkpoints/checkpoint_step_1000.pt

# Override config
PYTHONPATH=. .venv/Scripts/python.exe -m psudo_engine.training.train --max-steps 1000 --batch-size 8
```

## Project Structure

```
psudo_engine/
├── config/                 # YAML configs
├── weight_paging/          # Weight paging subsystem
├── context/                # Context/retrieval subsystem
├── model/                  # Model architecture
├── training/               # Training pipeline
├── inference/              # Inference pipeline
├── benchmarks/             # Benchmark suite
├── tests/                  # Test suite
├── data/                   # Data + shared schema (psudo-spec.json)
└── docs/                   # Documentation
```

## Key Files

| File | Purpose |
|------|---------|
| `config/model.yaml` | Model architecture |
| `config/training.yaml` | Training hyperparameters |
| `config/transpiler.yaml` | Transpiler config |
| `psudo_engine/data/psudo-spec.json` | Shared schema (grammar, builtins, AST) |
| `docs/sync_protocol.md` | Sync protocol docs |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run from repo root with `PYTHONPATH=.` |
| `CUDA out of memory` | Reduce `resident_slots` in config |
| `numpy not found` | `uv pip install numpy` (for synthetic data) |
| Windows file locking | Run on Linux/WSL or close file handles |
| `test_full_generation_pipeline` fails | Expected - model is untrained |

## Next Steps

1. **Train a model**: Need psudo code corpus + `uv pip install numpy`
2. **Evaluate**: Run benchmarks with trained checkpoint
3. **Deploy**: Build `psudo-harness` (separate repo) for transpilation